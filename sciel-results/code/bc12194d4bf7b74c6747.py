"""Pack 300 equal circles in a unit disk using hex lattice + force relaxation + local search."""
import numpy as np
import time
import math

N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def _hex_lattice_init(n, container_radius=1.0):
    """
    Generate n points on a hexagonal lattice, clipped to the unit disk.
    This is the natural close-packing geometry and gives a much better
    starting point than spiral/Fibonacci layouts.
    """
    # Estimate spacing: area per point ~ pi/(n) * container_radius^2
    # For hex packing, area per point = (sqrt(3)/2) * s^2 where s is spacing
    # So s ~ sqrt(2*pi*container_radius^2 / (n * sqrt(3)))
    s = math.sqrt(2.0 * math.pi * container_radius**2 / (n * math.sqrt(3.0)))
    # Add a small margin so we don't exceed the disk
    s *= 0.98
    
    # Generate hex lattice points
    # Hex lattice: rows offset by s/2 in x, spacing s in y, s*sqrt(3)/2 in y for alternating rows
    # Actually for hex close packing:
    # Point (i, j) -> x = s * (i + 0.5 * (j % 2)), y = s * sqrt(3)/2 * j
    pts = []
    # Estimate how many rows/cols needed
    max_extent = container_radius / s + 2
    for j in range(-int(max_extent), int(max_extent) + 1):
        for i in range(-int(max_extent), int(max_extent) + 1):
            x = s * (i + 0.5 * (j % 2))
            y = s * math.sqrt(3.0) / 2.0 * j
            if x*x + y*y <= container_radius**2:
                pts.append((x, y))
    
    pts = np.array(pts, dtype=np.float64)
    
    if len(pts) < n:
        # Need more points - reduce spacing
        for attempt in range(20):
            s *= 0.97
            pts = []
            max_extent = container_radius / s + 2
            for j in range(-int(max_extent), int(max_extent) + 1):
                for i in range(-int(max_extent), int(max_extent) + 1):
                    x = s * (i + 0.5 * (j % 2))
                    y = s * math.sqrt(3.0) / 2.0 * j
                    if x*x + y*y <= container_radius**2:
                        pts.append((x, y))
            pts = np.array(pts, dtype=np.float64)
            if len(pts) >= n:
                break
    
    if len(pts) > n:
        # Pick the n points closest to a uniform distribution
        # Simply take first n (they're in a deterministic order)
        pts = pts[:n]
    
    return pts

def _compute_radius(points):
    """Compute the feasible common radius for given points."""
    n = len(points)
    if n == 0:
        return 0.0
    # Boundary constraint
    dists_to_center = np.linalg.norm(points, axis=1)
    boundary_min = 1.0 - np.max(dists_to_center)
    if boundary_min < 0:
        return 0.0
    # Pairwise constraint
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist_matrix, np.inf)
    min_pair = np.min(dist_matrix)
    return min(boundary_min, min_pair / 2.0)

def _force_relaxation(points, time_limit=45.0, max_iters=50000):
    """
    Force-based relaxation: each point feels repulsive forces from all others
    (stronger when close) and an attractive force toward the center.
    Uses 1/d^3 weighting for repulsion to strongly penalize close pairs.
    """
    points = points.copy().astype(np.float64)
    n = len(points)
    start_time = time.time()
    
    # Estimate target radius to scale forces
    r_est = _compute_radius(points)
    if r_est < 1e-6:
        r_est = 0.05
    
    # Force parameters
    repulsion_strength = 1.0
    attraction_strength = 0.1
    damping = 0.5
    
    best_r = _compute_radius(points)
    best_points = points.copy()
    velocity = np.zeros_like(points)
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Compute forces
        forces = np.zeros_like(points)
        
        # Pairwise repulsion: F_ij = repulsion_strength * (r_est / d_ij)^3 * (x_i - x_j) / d_ij
        # Vectorized for speed
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dist_sq, np.inf)
        dist = np.sqrt(dist_sq)
        
        # Force magnitude: (r_est / d)^3 / d = r_est^3 / d^4
        # But we want it to blow up when d < r_est, so:
        # F = repulsion_strength * max(0, r_est - d + small) * (x_i - x_j) / d
        # Actually, use: F = repulsion_strength * (r_est / max(d, eps))^3 * (x_i - x_j) / d
        # This gives F ~ 1/d^4 direction (x_i - x_j)/d, so total ~ (x_i-x_j)/d^4
        
        # Safer: use soft repulsion that's strong at short range
        # F_ij direction = (x_i - x_j) / d_ij
        # Magnitude = repulsion_strength * (r_est / d_ij)^2 * (1 if d_ij < 2*r_est else 0)
        # Let's use: magnitude = repulsion_strength * r_est^3 / d_ij^3 for d_ij < 3*r_est
        
        mask = dist < 3.0 * r_est
        # Compute force for close pairs
        # F_i += sum_j: repulsion * (r_est^3 / d^3) * (x_i - x_j) / d = repulsion * r_est^3 * (x_i - x_j) / d^4
        # But this is huge. Let's normalize: use (r_est/d) as the ratio
        ratio = np.zeros_like(dist)
        ratio[mask] = r_est / dist[mask]
        # Force magnitude = repulsion_strength * ratio^3
        # Force vector = repulsion_strength * ratio^3 * (x_i - x_j) / d
        # = repulsion_strength * (r_est/d)^3 * (x_i - x_j)/d
        # = repulsion_strength * r_est^3 * (x_i - x_j) / d^4
        
        # Compute using broadcasting
        # force_contrib[i,j] = ratio[i,j]^3 * diff[i,j] / dist[i,j]
        # But diff[i,j] = x_i - x_j, and we want force on i to push away from j
        # So force on i from j: + ratio^3 * (x_i - x_j) / d
        # = ratio^3 * diff[i,j] / dist[i,j]
        
        # ratio^3 * diff / dist = (r_est/d)^3 * (x_i - x_j) / d = r_est^3 * (x_i-x_j) / d^4
        # Let's just compute: coeff = ratio^3 / dist
        coeff = np.zeros_like(dist)
        coeff[mask] = (ratio[mask]**3) / dist[mask]
        # Force on i = sum_j coeff[i,j] * diff[i,j]
        forces = np.sum(coeff[:, :, np.newaxis] * diff, axis=1) * repulsion_strength
        
        # Boundary attraction: pull points toward center if they're near the edge
        dists_to_center = np.linalg.norm(points, axis=1)
        near_boundary = dists_to_center > 0.85
        # Force toward center: -attraction * (x_i / |x_i|) * (|x_i| - 0.85) / 0.15
        for i in range(n):
            if dists_to_center[i] > 1e-10:
                pull = attraction_strength * max(0.0, (dists_to_center[i] - 0.85) / 0.15)
                forces[i] -= pull * points[i] / dists_to_center[i]
        
        # Update velocity with damping
        velocity = damping * velocity + (1.0 - damping) * forces
        # Limit step size
        step_size = np.min(np.linalg.norm(velocity, axis=1))
        max_step = 0.01
        if step_size > max_step:
            velocity *= max_step / step_size
        
        new_points = points + velocity
        
        # Project to unit disk
        new_dists = np.linalg.norm(new_points, axis=1)
        outside = new_dists > 1.0
        if np.any(outside):
            new_points[outside] *= (0.999 / new_dists[outside, np.newaxis])
        
        new_r = _compute_radius(new_points)
        if new_r > best_r:
            best_r = new_r
            best_points = new_points.copy()
        
        points = new_points
        
        # Cooldown: reduce forces over time
        if it % 1000 == 0 and it > 0:
            repulsion_strength *= 0.95
            attraction_strength *= 0.95
        
        # Early stop if converged
        if it > 1000 and new_r < best_r + 1e-10:
            if it > 5000:
                break
    
    return best_points, best_r

def _local_search(points, time_limit=30.0, max_iters=20000):
    """
    Targeted local search: find the most-constrained pair and try to move
    them apart. Also try to push boundary-constrained points inward.
    """
    points = points.copy().astype(np.float64)
    n = len(points)
    start_time = time.time()
    
    best_r = _compute_radius(points)
    best_points = points.copy()
    
    # Precompute pairwise distances
    def get_pairwise(points):
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dist_sq, np.inf)
        return np.sqrt(dist_sq)
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        dist_matrix = get_pairwise(points)
        np.fill_diagonal(dist_matrix, np.inf)
        
        # Find the minimum distance pair
        min_idx = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
        i, j = min_idx
        min_d = dist_matrix[i, j]
        
        # Also check boundary
        dists_to_center = np.linalg.norm(points, axis=1)
        max_boundary = np.max(dists_to_center)
        
        r = min(1.0 - max_boundary, min_d / 2.0)
        
        # Try to improve: move i and j apart
        if min_d < 2.0 * r + 1e-6:
            # They're too close - push apart
            direction = (points[i] - points[j]) / min_d
            # Move each by a small amount
            step = min(0.005, 0.1 * (2.0 * r - min_d))
            if step > 1e-8:
                new_points = points.copy()
                new_points[i] += direction * step
                new_points[j] -= direction * step
                # Project to disk
                new_dists = np.linalg.norm(new_points, axis=1)
                outside = new_dists > 1.0
                if np.any(outside):
                    new_points[outside] *= (0.999 / new_dists[outside, np.newaxis])
                new_r = _compute_radius(new_points)
                if new_r > best_r:
                    best_r = new_r
                    best_points = new_points.copy()
                    points = new_points
        elif max_boundary > 1.0 - r - 1e-6:
            # Boundary-constrained: push the outermost point inward
            outer_idx = np.argmax(dists_to_center)
            direction = -points[outer_idx] / dists_to_center[outer_idx]
            step = min(0.005, 0.1 * (1.0 - r - max_boundary))
            if step > 1e-8:
                new_points = points.copy()
                new_points[outer_idx] += direction * step
                new_dists = np.linalg.norm(new_points, axis=1)
                outside = new_dists > 1.0
                if np.any(outside):
                    new_points[outside] *= (0.999 / new_dists[outside, np.newaxis])
                new_r = _compute_radius(new_points)
                if new_r > best_r:
                    best_r = new_r
                    best_points = new_points.copy()
                    points = new_points
        else:
            # No obvious constraint - try random perturbation of a random point
            idx = np.random.randint(n)
            perturbation = np.random.normal(0, 0.002, size=2)
            new_points = points.copy()
            new_points[idx] += perturbation
            new_dists = np.linalg.norm(new_points, axis=1)
            outside = new_dists > 1.0
            if np.any(outside):
                new_points[outside] *= (0.999 / new_dists[outside, np.newaxis])
            new_r = _compute_radius(new_points)
            if new_r > best_r:
                best_r = new_r
                best_points = new_points.copy()
                points = new_points
            else:
                # Sometimes accept worse to escape local minima
                if np.random.random() < 0.05:
                    points = new_points
    
    return best_points, best_r

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    # Phase 1: Hex lattice initialization
    points = _hex_lattice_init(N)
    current_r = _compute_radius(points)
    
    best_points = points.copy()
    best_r = current_r
    
    # Phase 2: Force-based relaxation
    if time.time() < deadline - 60.0:
        opt_points, opt_r = _force_relaxation(points, time_limit=40.0)
        if opt_r > best_r:
            best_points = opt_points
            best_r = opt_r
    
    # Phase 3: Local search on most-constrained pairs
    if time.time() < deadline - 20.0:
        ls_points, ls_r = _local_search(best_points, time_limit=25.0)
        if ls_r > best_r:
            best_points = ls_points
            best_r = ls_r
    
    # Phase 4: Final force relaxation
    if time.time() < deadline - 5.0:
        final_points, final_r = _force_relaxation(best_points, time_limit=10.0)
        if final_r > best_r:
            best_points = final_points
            best_r = final_r
    
    # Phase 5: Final local search
    if time.time() < deadline - 2.0:
        final_ls_points, final_ls_r = _local_search(best_points, time_limit=5.0)
        if final_ls_r > best_r:
            best_points = final_ls_points
            best_r = final_ls_r
    
    return best_points
# EVOLVE-BLOCK-END
