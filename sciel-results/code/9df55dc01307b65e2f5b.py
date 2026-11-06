"""
Hexagonal lattice initialization with gradient-based optimization for circle packing.
N=300 equal circles in unit circle.
"""
import numpy as np
import time
from scipy.spatial import cKDTree

N = 300

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s limit
    
    def compute_score(pts):
        """Compute the packing score r for given points."""
        if len(pts) != N:
            return 0.0
        if not np.all(np.isfinite(pts)):
            return 0.0
        radii = np.hypot(pts[:, 0], pts[:, 1])
        if np.any(radii >= 1.0):
            return 0.0
        min_boundary = np.min(1.0 - radii)
        # Pairwise distances
        tree = cKDTree(pts)
        pairs, dists = tree.query(pts, k=2)
        # pairs[:, 1] is the nearest neighbor index, dists[:, 1] is the distance
        min_pair = np.min(dists[:, 1]) / 2.0
        return min(min_pair, min_boundary)
    
    def generate_hex_init():
        """Generate hexagonal lattice configuration clipped to unit circle."""
        # Hexagonal lattice spacing
        # For N circles in unit circle, estimate spacing
        # Area per circle ~ pi * R^2 / N, so spacing ~ sqrt(pi * R^2 / N)
        # But hexagonal packing has density pi/(2*sqrt(3))
        # Let's estimate: total area = pi, area per circle = pi/N
        # Circle radius r, so area per circle in hex lattice ~ r * 2 * sqrt(3) * r = 2*sqrt(3)*r^2
        # Actually for hex lattice with spacing s, each point has area s^2 * sqrt(3)/2
        # So s^2 * sqrt(3)/2 = pi/N => s = sqrt(2*pi/(N*sqrt(3)))
        
        s = np.sqrt(2.0 * np.pi / (N * np.sqrt(3.0)))
        
        # Generate hex lattice points
        # Hex lattice: two basis vectors
        # v1 = s * (1, 0)
        # v2 = s * (0.5, sqrt(3)/2)
        v1 = np.array([s, 0.0])
        v2 = np.array([0.5 * s, np.sqrt(3.0) / 2.0 * s])
        
        # Generate a grid of lattice points
        # The lattice points are i*v1 + j*v2 for integers i, j
        # We need to find all points within the unit circle (with some margin)
        
        # Estimate the range of i and j
        # Max |i*v1 + j*v2| < 1 + margin
        # |v1| = s, |v2| = s
        # The extent in each direction is roughly 1/s
        extent = int(np.ceil(1.0 / s)) + 2
        
        points = []
        for i in range(-extent, extent + 1):
            for j in range(-extent, extent + 1):
                p = i * v1 + j * v2
                if np.hypot(p[0], p[1]) < 1.0:
                    points.append(p)
        
        points = np.array(points)
        
        # We need exactly N points. If we have more, we need to select N.
        # If we have fewer, we need to add more.
        # Sort by distance from origin (prefer points closer to center)
        dists_from_origin = np.hypot(points[:, 0], points[:, 1])
        order = np.argsort(dists_from_origin)
        points = points[order]
        
        if len(points) > N:
            # Take the N closest to center
            pts = points[:N].copy()
        elif len(points) < N:
            # Need to add more points. Use a finer lattice or spiral
            # Let's use a spiral to fill in
            pts = points.copy()
            remaining = N - len(pts)
            if remaining > 0:
                # Add points on a spiral in the remaining space
                # Find the max radius used
                max_r = np.max(np.hypot(pts[:, 0], pts[:, 1])) if len(pts) > 0 else 0.0
                # Add points in a spiral from max_r to 1.0
                ga = np.pi * (3.0 - np.sqrt(5.0))
                for k in range(remaining):
                    t = (k + 1) / (remaining + 1)
                    r = max_r + t * (1.0 - max_r - 0.05)  # Leave some margin
                    angle = k * ga
                    p = np.array([r * np.cos(angle), r * np.sin(angle)])
                    pts = np.vstack([pts, p])
        else:
            pts = points.copy()
        
        return pts
    
    def gradient_optimize(pts, max_iters=20000, deadline_check=None):
        """
        Gradient ascent on the minimum distance metric.
        Objective: maximize r = min(min_pair_dist/2, min_boundary_margin)
        
        We use a smoothed objective: sum of log of distances to nearest neighbors
        and boundary margins.
        """
        pts = pts.copy()
        best_pts = pts.copy()
        best_score = compute_score(pts)
        
        step_size = 0.005
        patience = 0
        max_patience = 500
        lr_decay = 0.95
        
        for it in range(max_iters):
            if deadline_check is not None and time.time() > deadline_check:
                break
            
            # Compute pairwise distances using KDTree
            tree = cKDTree(pts)
            # Get 2 nearest neighbors (including self)
            pairs, dists = tree.query(pts, k=2)
            # dists[:, 0] = 0 (self), dists[:, 1] = distance to nearest neighbor
            nn_dists = dists[:, 1].copy()
            nn_indices = pairs[:, 1]
            
            # Boundary distances
            radii = np.hypot(pts[:, 0], pts[:, 1])
            boundary_dists = 1.0 - radii
            
            # Compute gradient of the objective
            # We want to maximize the minimum of all distances and boundary margins.
            # Use a smoothed version: sum of log of distances.
            
            # For pairwise term: for each point i, the nearest neighbor j contributes
            # grad_i += (pts[i] - pts[j]) / dist_ij^2
            # grad_j += (pts[j] - pts[i]) / dist_ij^2 = -grad_i contribution
            
            grad = np.zeros_like(pts)
            for i in range(N):
                j = nn_indices[i]
                d = nn_dists[i]
                if d > 1e-10:
                    diff = pts[i] - pts[j]
                    grad[i] += diff / (d * d)
                    grad[j] -= diff / (d * d)
            
            # Boundary term: grad_i += -pts[i] / (r_i * (1 - r_i))
            radii_safe = np.where(radii > 1e-10, radii, 1e-10)
            bound_factor = 1.0 / (radii_safe * (1.0 - radii_safe))
            grad -= pts * bound_factor[:, np.newaxis]
            
            # Update
            pts += grad * step_size
            
            # Project inside unit circle
            new_radii = np.hypot(pts[:, 0], pts[:, 1])
            outside = new_radii > 1.0
            if np.any(outside):
                pts[outside] = pts[outside] / new_radii[outside, np.newaxis]
            
            # Evaluate
            current_score = compute_score(pts)
            
            if current_score > best_score:
                best_score = current_score
                best_pts = pts.copy()
                patience = 0
                step_size = min(step_size * 1.02, 0.02)
            else:
                patience += 1
                step_size *= lr_decay
                if patience > max_patience:
                    # Reset with a kick
                    pts = best_pts.copy() + np.random.randn(N, 2) * 0.003
                    new_radii = np.hypot(pts[:, 0], pts[:, 1])
                    outside = new_radii > 1.0
                    if np.any(outside):
                        pts[outside] = pts[outside] / new_radii[outside, np.newaxis]
                    step_size = 0.005
                    patience = 0
                    max_patience = int(max_patience * 1.5)
            
            # Check deadline
            if deadline_check is not None and time.time() > deadline_check:
                break
        
        return best_pts, best_score
    
    def local_search(pts, max_iters=5000, deadline_check=None):
        """Local search: perturb individual points to improve the minimum distance."""
        pts = pts.copy()
        best_score = compute_score(pts)
        
        for it in range(max_iters):
            if deadline_check is not None and time.time() > deadline_check:
                break
            
            # Find the bottleneck: is it a pair or the boundary?
            tree = cKDTree(pts)
            pairs, dists = tree.query(pts, k=2)
            nn_dists = dists[:, 1]
            radii = np.hypot(pts[:, 0], pts[:, 1])
            boundary_dists = 1.0 - radii
            
            min_pair_dist = np.min(nn_dists)
            min_boundary = np.min(boundary_dists)
            
            if min_pair_dist / 2.0 < min_boundary:
                # Bottleneck is a pair
                # Find the pair with minimum distance
                min_idx = np.argmin(nn_dists)
                i = min_idx
                j = pairs[min_idx, 1]
                # Try to move i or j apart
                for move_idx in [i, j]:
                    for _ in range(10):
                        orig = pts[move_idx].copy()
                        # Move away from the other point
                        other = pts[j if move_idx == i else i]
                        direction = pts[move_idx] - other
                        norm = np.linalg.norm(direction)
                        if norm > 1e-10:
                            direction = direction / norm
                        else:
                            direction = np.random.randn(2)
                            direction = direction / np.linalg.norm(direction)
                        
                        pts[move_idx] += direction * 0.005
                        r = np.hypot(pts[move_idx, 0], pts[move_idx, 1])
                        if r > 1.0:
                            pts[move_idx] = pts[move_idx] / r
                        
                        new_score = compute_score(pts)
                        if new_score > best_score:
                            best_score = new_score
                            break
                        else:
                            pts[move_idx] = orig
            else:
                # Bottleneck is boundary
                min_idx = np.argmin(boundary_dists)
                i = min_idx
                # Move inward
                for _ in range(10):
                    orig = pts[i].copy()
                    direction = -pts[i]
                    norm = np.linalg.norm(direction)
                    if norm > 1e-10:
                        direction = direction / norm
                    else:
                        direction = np.random.randn(2)
                        direction = direction / np.linalg.norm(direction)
                    
                    pts[i] += direction * 0.005
                    new_score = compute_score(pts)
                    if new_score > best_score:
                        best_score = new_score
                        break
                    else:
                        pts[i] = orig
        
        return pts, best_score
    
    # Main execution
    # 1. Initialize with hexagonal lattice
    pts = generate_hex_init()
    best_pts = pts.copy()
    best_score = compute_score(pts)
    
    # 2. Gradient optimization
    pts, score = gradient_optimize(pts, max_iters=30000, deadline_check=deadline - 10)
    if score > best_score:
        best_score = score
        best_pts = pts
    
    # 3. Local search refinement
    remaining_time = deadline - time.time()
    if remaining_time > 5:
        pts, score = local_search(best_pts, max_iters=20000, deadline_check=deadline - 2)
        if score > best_score:
            best_score = score
            best_pts = pts
    
    # Final validation
    final_score = compute_score(best_pts)
    if final_score > best_score:
        best_score = final_score
    
    # Ensure validity
    radii = np.hypot(best_pts[:, 0], best_pts[:, 1])
    if np.any(radii >= 1.0) or not np.all(np.isfinite(best_pts)):
        scale = 0.99 / np.max(radii) if np.max(radii) > 0 else 1.0
        best_pts = best_pts * scale
    
    return best_pts
