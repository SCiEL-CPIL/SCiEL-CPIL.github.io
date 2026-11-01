"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 50
_DEADLINE = time.time() + 140.0  # 10s safety margin before 150s kill

def _score(pts):
    """Compute the valid packing radius (or 0 if invalid)."""
    if len(pts) != N:
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    
    # Check if all points are inside the unit circle
    dists_origin = np.linalg.norm(pts, axis=1)
    if np.any(dists_origin > 1.0):
        return 0.0
    
    # Calculate minimum distance to boundary
    r_cont = 1.0 - np.max(dists_origin)
    if r_cont < 0:
        return 0.0
    
    if N < 2:
        return r_cont
        
    # Efficient pairwise distance computation
    # dist_mat[i, j] = ||pts[i] - pts[j]||
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_mat = np.linalg.norm(diff, axis=-1)
    
    # Ignore self-distances
    np.fill_diagonal(dist_mat, np.inf)
    
    # Minimum pairwise distance
    min_pair_dist = np.min(dist_mat)
    r_pair = 0.5 * min_pair_dist
    
    # The valid radius is the minimum of the boundary constraint and pairwise constraint
    return min(r_cont, r_pair)

def _hex_lattice_init():
    """Generate a hexagonal lattice packing as initial guess, optimized for N=50."""
    # Start with a spacing that likely yields enough points
    # For N=50, a spacing of around 0.12-0.15 is typical for unit circle
    # Let's try to find a good spacing by iterating
    best_pts = None
    best_score = -1
    
    # Try a range of spacings
    for spacing in np.linspace(0.10, 0.18, 10):
        k = int(1.0 / spacing) + 2
        # Generate grid points
        rows = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                rows.append((x, y))
        
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        
        # Filter points that are safely inside the unit circle
        # We want points that can fit a circle of radius r. 
        # Initially, we don't know r, but we can assume it's roughly spacing/2.
        # To be safe, we keep points with r < 1.0 - spacing/2
        mask = radius <= 1.0 - spacing / 2.0
        points = points[mask]
        radius = radius[mask]
        
        if len(points) >= N:
            # Select N points. 
            # Strategy: Sort by distance from origin and take the first N? 
            # This creates a compact core.
            idx = np.argsort(radius, kind="stable")[:N]
            candidate_pts = points[idx]
            
            # Evaluate this candidate
            sc = _score(candidate_pts)
            if sc > best_score:
                best_score = sc
                best_pts = candidate_pts
        
    if best_pts is None:
        # Fallback: Random initialization if lattice fails (unlikely)
        angles = np.random.uniform(0, 2 * np.pi, N)
        radii = np.sqrt(np.random.uniform(0, 0.81, N))
        best_pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        
    return best_pts

def _optimize(pts0, max_time=120.0):
    """
    Vectorized repulsion optimization with momentum and boundary projection.
    """
    pts = pts0.copy().astype(np.float64)
    best_pts = pts.copy()
    best_score = _score(pts)
    
    # Parameters
    step_size = 0.01
    momentum = 0.5
    v = np.zeros_like(pts)
    
    start_time = time.time()
    
    while time.time() < min(start_time + max_time, _DEADLINE):
        # 1. Compute Forces
        forces = np.zeros_like(pts)
        
        # Pairwise repulsion
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=-1)
        
        # Avoid division by zero
        np.fill_diagonal(dists, 1.0)
        
        # Force magnitude: 1/d^2 is a common choice for stability
        weights = 1.0 / (dists * dists)
        np.fill_diagonal(weights, 0.0)
        
        # Force on i is sum_j (p_i - p_j) * w_ij
        forces = np.sum(diff * weights[..., np.newaxis], axis=1)
        
        # 2. Update Velocity and Position
        v = momentum * v + (1 - momentum) * forces
        # Normalize velocity to prevent huge jumps
        v_norms = np.linalg.norm(v, axis=1, keepdims=True)
        v_norms[v_norms == 0] = 1.0
        v = v / v_norms
        
        new_pts = pts + step_size * v
        
        # 3. Project to Unit Disk
        new_radii = np.linalg.norm(new_pts, axis=1)
        mask = new_radii > 1.0
        if np.any(mask):
            new_pts[mask] = new_pts[mask] / new_radii[mask, np.newaxis]
            
        # 4. Evaluate
        new_score = _score(new_pts)
        
        if new_score > best_score:
            best_score = new_score
            best_pts = new_pts.copy()
            pts = new_pts.copy()
            # Slightly increase step size if improving significantly
            if new_score > best_score * 1.001:
                step_size = min(0.05, step_size * 1.05)
        else:
            # Decrease step size if not improving
            step_size *= 0.99
            # If stuck, try a small random perturbation
            if step_size < 0.001:
                pts = best_pts + np.random.normal(0, 0.01, (N, 2))
                radii = np.linalg.norm(pts, axis=1)
                mask = radii > 1.0
                if np.any(mask):
                    pts[mask] = pts[mask] / radii[mask, np.newaxis]
                v = np.zeros_like(pts)
                step_size = 0.01

    return best_pts

def construct():
    start_time = time.time()
    
    # 1. Get initial hexagonal lattice configuration
    pts0 = _hex_lattice_init()
    best_pts = pts0
    best_score = _score(pts0)
    
    # 2. Run optimization if there is enough time
    if time.time() < _DEADLINE - 10:
        try:
            optimized_pts = _optimize(pts0, max_time=130.0)
            opt_score = _score(optimized_pts)
            
            if opt_score > best_score:
                best_pts = optimized_pts
                best_score = opt_score
        except Exception:
            # If optimization fails, stick with the initial lattice
            pass

    # Final validity check
    if _score(best_pts) <= 0:
        # Fallback to a simple valid configuration if something went wrong
        # This should rarely happen
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        radii = np.ones(N) * 0.5
        best_pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])

    return best_pts
