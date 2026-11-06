"""Packing 300 equal circles in a unit circle - model-and-optimize with repulsion relaxation."""
import numpy as np
import time
from scipy.spatial.distance import pdist, cdist

N = 300

def _compute_radius(pts):
    """Compute the packing radius for a configuration."""
    if pts.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    # Boundary constraint
    dists_origin = np.hypot(pts[:, 0], pts[:, 1])
    boundary_min = 1.0 - np.max(dists_origin)
    if boundary_min <= 0:
        return 0.0
    # Pairwise distances
    min_pair = np.min(pdist(pts))
    return min(min_pair / 2.0, boundary_min)

def _compute_radius_fast(pts):
    """Faster radius computation using vectorized operations."""
    if pts.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    # Boundary constraint
    dists_origin = np.sqrt(np.sum(pts * pts, axis=1))
    boundary_min = 1.0 - np.max(dists_origin)
    if boundary_min <= 0:
        return 0.0
    # Pairwise distances using pdist (more efficient for this size)
    min_pair = np.min(pdist(pts))
    return min(min_pair / 2.0, boundary_min)

def _hex_initial(spacing, n_points=N):
    """Generate a hexagonal lattice configuration and select n_points."""
    sqrt3 = np.sqrt(3.0)
    k = int(1.0 / spacing) + 3
    
    points_list = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * sqrt3 / 2.0
            r = np.hypot(x, y)
            if r < 1.0:
                points_list.append((x, y))
    
    points = np.array(points_list)
    if len(points) < n_points:
        return None
    
    # Sort by distance from origin and keep central points
    dists = np.hypot(points[:, 0], points[:, 1])
    idx = np.argsort(dists, kind='stable')[:n_points]
    return points[idx]

def _repulsion_relax(pts, max_iter=5000, time_limit=None):
    """Force-based relaxation to push overlapping circles apart and boundary-violating circles inward."""
    if time_limit is None:
        time_limit = time.time() + 100.0
    
    current_pts = pts.copy()
    current_r = _compute_radius_fast(current_pts)
    best_pts = current_pts.copy()
    best_r = current_r
    
    # Force parameters
    repulsion_strength = 1.0
    boundary_strength = 2.0
    damping = 0.5
    step_scale = 0.01
    
    for iteration in range(max_iter):
        if time.time() >= time_limit:
            break
        
        # Compute forces on each point
        forces = np.zeros_like(current_pts)
        
        # Repulsion from other circles (only consider nearby pairs for efficiency)
        # For N=300, we can compute all pairs
        dists = cdist(current_pts, current_pts)
        np.fill_diagonal(dists, np.inf)
        
        # Find pairs that are too close (distance < 2*current_r)
        # Use a threshold based on current radius
        threshold = max(2 * current_r, 0.05)
        
        for i in range(N):
            # Find points too close to i
            close_mask = dists[i] < threshold
            if np.any(close_mask):
                close_indices = np.where(close_mask)[0]
                for j in close_indices:
                    d = dists[i, j]
                    if d < 1e-10:
                        # Coincident points - push in random direction
                        angle = np.random.uniform(0, 2 * np.pi)
                        forces[i, 0] += repulsion_strength * np.cos(angle)
                        forces[i, 1] += repulsion_strength * np.sin(angle)
                        forces[j, 0] -= repulsion_strength * np.cos(angle)
                        forces[j, 1] -= repulsion_strength * np.sin(angle)
                    else:
                        # Push apart along the line connecting them
                        direction = (current_pts[i] - current_pts[j]) / d
                        force_mag = repulsion_strength * (threshold - d) / threshold
                        forces[i] += force_mag * direction
                        forces[j] -= force_mag * direction
        
        # Boundary constraint: push points inside if too close to boundary
        dists_origin = np.sqrt(np.sum(current_pts * current_pts, axis=1))
        for i in range(N):
            if dists_origin[i] > 1.0 - current_r:
                # Push inward
                if dists_origin[i] > 1e-10:
                    inward = -current_pts[i] / dists_origin[i]
                    force_mag = boundary_strength * (dists_origin[i] - (1.0 - current_r))
                    forces[i] += force_mag * inward
        
        # Apply forces with damping
        current_pts += damping * step_scale * forces
        
        # Clamp to stay inside container
        dists_origin = np.sqrt(np.sum(current_pts * current_pts, axis=1))
        max_allowed = 1.0 - 1e-8
        mask = dists_origin > max_allowed
        if np.any(mask):
            current_pts[mask] *= max_allowed / dists_origin[mask]
        
        # Compute new radius
        new_r = _compute_radius_fast(current_pts)
        if new_r > best_r:
            best_r = new_r
            best_pts = current_pts.copy()
            current_r = new_r
        elif new_r < current_r * 0.9:
            # If radius dropped significantly, revert
            current_pts = best_pts.copy()
            current_r = best_r
    
    return best_pts, best_r

def _local_search(pts, max_iter=5000, time_limit=None):
    """Local search by perturbing individual points."""
    if time_limit is None:
        time_limit = time.time() + 80.0
    
    current_pts = pts.copy()
    current_r = _compute_radius_fast(current_pts)
    best_pts = current_pts.copy()
    best_r = current_r
    
    for iteration in range(max_iter):
        if time.time() >= time_limit:
            break
        
        # Pick a random point to move
        idx = np.random.randint(N)
        old_pos = current_pts[idx].copy()
        
        # Propose a new position: small random perturbation
        step = 0.003
        new_pos = old_pos + np.random.uniform(-step, step, 2)
        
        # Keep it inside the circle
        new_r_mag = np.hypot(new_pos[0], new_pos[1])
        if new_r_mag > 0.98:
            new_pos = new_pos / new_r_mag * 0.98
        
        current_pts[idx] = new_pos
        new_r = _compute_radius_fast(current_pts)
        
        if new_r > current_r:
            current_r = new_r
            best_pts = current_pts.copy()
            best_r = current_r
        else:
            current_pts[idx] = old_pos
    
    return best_pts, best_r

def _gradient_ascent(pts, max_iter=2000, time_limit=None):
    """Smooth gradient-based optimization using soft-min surrogate."""
    if time_limit is None:
        time_limit = time.time() + 60.0
    
    current_pts = pts.copy()
    current_r = _compute_radius_fast(current_pts)
    best_pts = current_pts.copy()
    best_r = current_r
    
    # Soft-min parameter
    beta = 50.0
    
    for iteration in range(max_iter):
        if time.time() >= time_limit:
            break
        
        # Compute soft-min of pairwise distances/2 and boundary distances
        dists_origin = np.sqrt(np.sum(current_pts * current_pts, axis=1))
        boundary_terms = 1.0 - dists_origin
        
        # Pairwise distances
        pair_dists = pdist(current_pts)
        pair_terms = pair_dists / 2.0
        
        # Combine all terms
        all_terms = np.concatenate([pair_terms, boundary_terms])
        
        # Soft-min using log-sum-exp
        max_term = np.max(all_terms)
        exp_terms = np.exp(-beta * (all_terms - max_term))
        soft_min = max_term - np.log(np.sum(exp_terms)) / beta
        
        # Gradient of soft-min with respect to points
        # d(soft_min)/d(point_i) = -sum_j exp(-beta*(term_ij - max)) / sum_k exp(-beta*(term_k - max)) * d(term_ij)/d(point_i)
        
        grad = np.zeros_like(current_pts)
        denom = np.sum(exp_terms)
        
        # Boundary gradient
        for i in range(N):
            if dists_origin[i] > 1e-10:
                # d(boundary_term_i)/d(point_i) = -point_i / |point_i|
                weight = exp_terms[N*(N-1)//2 + i] / denom
                grad[i] += weight * (-current_pts[i] / dists_origin[i])
        
        # Pairwise gradient
        pair_idx = 0
        for i in range(N):
            for j in range(i + 1, N):
                d = np.sqrt(np.sum((current_pts[i] - current_pts[j]) ** 2))
                if d > 1e-10:
                    weight = exp_terms[pair_idx] / denom
                    direction = (current_pts[i] - current_pts[j]) / d
                    grad[i] += weight * direction / 2.0
                    grad[j] -= weight * direction / 2.0
                pair_idx += 1
        
        # Update
        step_size = 0.001
        current_pts += step_size * grad
        
        # Clamp to stay inside container
        dists_origin = np.sqrt(np.sum(current_pts * current_pts, axis=1))
        max_allowed = 1.0 - 1e-8
        mask = dists_origin > max_allowed
        if np.any(mask):
            current_pts[mask] *= max_allowed / dists_origin[mask]
        
        # Compute new radius
        new_r = _compute_radius_fast(current_pts)
        if new_r > best_r:
            best_r = new_r
            best_pts = current_pts.copy()
            current_r = new_r
    
    return best_pts, best_r

def construct():
    deadline = time.time() + 140.0  # Safety margin before 150s kill
    
    def time_check():
        return time.time() < deadline
    
    # --- Stage 1: Construct a good initial configuration ---
    best_points = None
    best_radius = 0.0
    
    # Try multiple hexagonal lattice spacings
    for trial in range(30):
        if not time_check():
            break
        
        spacing = 0.09 + trial * 0.001
        
        pts = _hex_initial(spacing)
        if pts is None:
            continue
        
        r = _compute_radius_fast(pts)
        if r > best_radius:
            best_radius = r
            best_points = pts.copy()
    
    if best_points is None:
        # Fallback: uniform random in circle
        best_points = np.column_stack([
            np.cos(np.random.uniform(0, 2 * np.pi, N)) * np.sqrt(np.random.uniform(0, 1, N)),
            np.sin(np.random.uniform(0, 2 * np.pi, N)) * np.sqrt(np.random.uniform(0, 1, N))
        ])
        best_radius = 0.01
    
    # --- Stage 2: Repulsion relaxation ---
    if time_check():
        best_points, best_radius = _repulsion_relax(
            best_points, max_iter=3000, time_limit=time.time() + 50.0
        )
    
    # --- Stage 3: Local search ---
    if time_check():
        best_points, best_radius = _local_search(
            best_points, max_iter=3000, time_limit=time.time() + 40.0
        )
    
    # --- Stage 4: Gradient ascent polish ---
    if time_check():
        best_points, best_radius = _gradient_ascent(
            best_points, max_iter=1000, time_limit=time.time() + 30.0
        )
    
    # Final validation
    dists_origin = np.hypot(best_points[:, 0], best_points[:, 1])
    max_allowed = 1.0 - 1e-10
    mask = dists_origin >= max_allowed
    if np.any(mask):
        best_points[mask] *= (max_allowed / dists_origin[mask]) * 0.999
    
    # Ensure no coincident points
    for i in range(N):
        for j in range(i + 1, N):
            d = np.hypot(best_points[i, 0] - best_points[j, 0],
                         best_points[i, 1] - best_points[j, 1])
            if d < 1e-6:
                angle = np.random.uniform(0, 2 * np.pi)
                best_points[i, 0] += 0.01 * np.cos(angle)
                best_points[i, 1] += 0.01 * np.sin(angle)
                best_points[j, 0] -= 0.01 * np.cos(angle)
                best_points[j, 1] -= 0.01 * np.sin(angle)
    
    # Final check
    if not np.all(np.isfinite(best_points)) or best_points.shape != (N, 2):
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        radii = np.sqrt(np.linspace(0, 1, N))
        best_points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        best_points *= 0.99
    
    return best_points
