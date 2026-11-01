import time
import numpy as np
import random

N = 50
# Safety margin: The evaluator kills at 150s. We stop at 145s.
DEADLINE = time.time() + 145.0

def _score(pts):
    """
    Compute the valid packing radius (or 0 if invalid).
    Returns the largest common radius r such that circles are disjoint and inside the unit circle.
    """
    if len(pts) != N:
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    
    # Check boundary
    dists_origin = np.linalg.norm(pts, axis=1)
    if np.any(dists_origin > 1.0):
        return 0.0
    r_cont = 1.0 - np.max(dists_origin)
    if r_cont < 0:
        return 0.0
    
    # Check pairwise distances
    if N < 2:
        return r_cont
        
    # Efficient pairwise distance computation
    # dx[i,j] = pts[i,0] - pts[j,0]
    dx = pts[:, 0:1] - pts[0:1, 0]
    dy = pts[:, 1:2] - pts[0:1, 1]
    dist_mat = np.sqrt(dx * dx + dy * dy)
    np.fill_diagonal(dist_mat, np.inf)
    r_pair = 0.5 * np.min(dist_mat)
    
    return min(r_cont, r_pair)

def _initialize_hex(spacing):
    """
    Generate a hexagonal lattice of points.
    Returns points that are strictly inside the unit circle (with margin spacing/2).
    """
    points = []
    k = int(2.0 / spacing) + 2
    for j in range(-k, k + 1):
        y = j * spacing * np.sqrt(3.0) / 2.0
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            r = np.hypot(x, y)
            # Keep points that can fit a circle of radius spacing/2
            if r + spacing / 2.0 <= 1.0:
                points.append((x, y))
    
    pts = np.array(points)
    if len(pts) < N:
        return None
    
    # Sort by distance from origin to take the most central N points
    radii = np.linalg.norm(pts, axis=1)
    idx = np.argsort(radii, kind='stable')[:N]
    return pts[idx]

def _initialize_random():
    """
    Generate N random points inside the unit circle.
    """
    angles = np.random.uniform(0, 2 * np.pi, N)
    radii = np.sqrt(np.random.uniform(0, 1.0, N))
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])

def _optimize(pts_init, max_time=10.0):
    """
    Optimize the packing using a force-directed method with simulated annealing.
    We minimize a potential energy function that penalizes overlaps and boundary violations.
    The potential is:
    E = sum_{i<j} max(0, 2r - d_ij)^2 + sum_i max(0, |c_i| - (1-r))^2
    However, r is variable. A simpler proxy that works well is to maximize the minimum distance.
    
    Strategy:
    1. Start with a small r.
    2. Calculate forces to increase r or move points to increase local clearance.
    3. Use a "soft" repulsion potential: V = sum 1/d_ij^2 for d_ij < 2r_est.
    
    Let's use a standard approach:
    Minimize E = sum_{i<j} 1/d_ij^2 + sum_i 1/(1-|c_i|)^2
    This pushes points apart and away from the boundary.
    """
    pts = pts_init.copy().astype(np.float64)
    best_pts = pts.copy()
    best_score = _score(pts)
    
    # Parameters
    T = 0.1  # Temperature
    cooling = 0.99
    step_scale = 0.01
    
    start_time = time.time()
    
    # Precompute indices for pairs? N=50 is small, full matrix is fine.
    # We will use a simple gradient descent on the "badness" function.
    # Badness = sum of penalties for close pairs and boundary violations.
    
    # To be more effective, we can use the specific metric:
    # We want to maximize min(r_pair, r_cont).
    # This is non-smooth. 
    # Instead, we minimize sum_{i<j} 1/d_ij^2 + lambda * sum_i 1/(1-|c_i|)^2.
    # As points move, the minimum distance increases.
    
    while time.time() < min(start_time + max_time, DEADLINE):
        # Compute gradients of the potential
        # V = sum_{i<j} d_ij^{-2} + sum_i (1 - |c_i|)^{-2}
        
        forces = np.zeros_like(pts)
        
        # Pairwise repulsion
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diff * diff, axis=-1)
        # Avoid division by zero
        np.fill_diagonal(dists_sq, 1.0)
        
        # Gradient of 1/d^2 w.r.t p_i is -2 * (p_i - p_j) / d^4
        # We want to minimize V, so move opposite to gradient?
        # Force = -grad V.
        # d(1/d^2)/dp_i = -2 (p_i - p_j) / d^4
        # So Force = 2 (p_i - p_j) / d^4
        
        # weights = 2 / dists_sq^2
        weights = 2.0 / (dists_sq * dists_sq)
        np.fill_diagonal(weights, 0.0)
        
        forces_rep = np.sum(diff * weights[..., np.newaxis], axis=1)
        
        # Boundary repulsion
        # V_b = sum_i 1 / (1 - |c_i|)^2
        # dV_b/dp_i = 2 * p_i / (|c_i| * (1 - |c_i|)^3)  (approx, assuming |c_i| > 0)
        # Actually, let's use a softer boundary potential to avoid singularities at origin.
        # V_b = sum_i 1 / (1 - |c_i|)^2
        # grad V_b_i = 2 * (p_i / |p_i|) / (1 - |p_i|)^3
        dists_origin = np.linalg.norm(pts, axis=1)
        # Avoid division by zero at origin
        dists_origin_safe = np.where(dists_origin < 1e-6, 1e-6, dists_origin)
        
        # Only apply boundary force if close to boundary (dist > 0.5) to avoid pushing from center?
        # Actually, we want to push OUT of the boundary if inside? No, we want to stay inside.
        # The potential 1/(1-r)^2 is high when r is close to 1.
        # Gradient points inwards (towards origin) to reduce r?
        # d/dr (1/(1-r)^2) = 2/(1-r)^3. This is positive.
        # So to minimize V, we should decrease r. Force is inwards.
        # This helps keep points inside.
        
        # Force direction: -grad V.
        # grad V points outwards (increases r). So -grad V points inwards.
        # We want to prevent points from going outside.
        # If a point is outside, we project it.
        
        # Let's use a simpler constraint: Project to disk after update.
        
        forces_boundary = np.zeros_like(pts)
        # Apply boundary force only for points close to edge to save computation?
        # For N=50, it's cheap.
        mask = dists_origin > 0.1
        # grad V_b = 2 * (p_i / |p_i|) / (1 - |p_i|)^3
        # Force = -grad V_b = -2 * (p_i / |p_i|) / (1 - |p_i|)^3
        # This points INWARDS.
        denom = (1.0 - dists_origin_safe[mask]) ** 3
        denom = np.where(denom < 1e-9, 1e-9, denom)
        forces_boundary[mask] = -2.0 * (pts[mask] / dists_origin_safe[mask, np.newaxis]) / denom[:, np.newaxis]
        
        total_forces = forces_rep + forces_boundary
        
        # Update
        # Normalize forces to avoid huge steps
        f_norms = np.linalg.norm(total_forces, axis=1, keepdims=True)
        f_norms[f_norms == 0] = 1.0
        total_forces = total_forces / f_norms
        
        # Add noise (Simulated Annealing)
        noise = np.random.normal(0, T, (N, 2))
        
        new_pts = pts + step_scale * (total_forces + noise)
        
        # Project to unit disk
        new_radii = np.linalg.norm(new_pts, axis=1)
        mask_out = new_radii > 1.0
        if np.any(mask_out):
            new_pts[mask_out] = new_pts[mask_out] / new_radii[mask_out, np.newaxis]
            
        new_score = _score(new_pts)
        
        # Accept/Reject (Metropolis)
        if new_score > best_score:
            best_score = new_score
            best_pts = new_pts.copy()
            pts = new_pts.copy()
            T *= cooling
        else:
            # Keep pts as is, or revert?
            # In Metropolis, we might accept worse states.
            # But for this specific metric, "worse" means smaller min radius.
            # Let's just stick to hill climbing for stability, but with noise.
            # If we accept worse states, we need to be careful not to drift out.
            # Let's try accepting worse states with probability exp(-delta/E)
            # But we don't have an energy scale.
            # Let's just do hill climbing with random restarts if stuck.
            pts = new_pts.copy() # Move anyway to explore, but keep best
            T *= cooling
            
        if T < 1e-6:
            break
            
    return best_pts, best_score

def construct():
    start_time = time.time()
    best_pts = None
    best_score = 0.0
    
    # 1. Try Hexagonal Lattice Initialization
    # The optimal radius is ~0.1258. Diameter ~0.2516.
    # Hex spacing should be around diameter.
    # Let's try a few spacings around the expected value.
    base_spacing = 0.2516
    for spacing_factor in [0.9, 0.95, 1.0, 1.05, 1.1]:
        spacing = base_spacing * spacing_factor
        pts_init = _initialize_hex(spacing)
        if pts_init is not None:
            # Ensure it's a valid start (it should be, by construction)
            score_init = _score(pts_init)
            if score_init > best_score:
                best_score = score_init
                best_pts = pts_init.copy()
                
            # Optimize
            if time.time() < DEADLINE - 5.0:
                opt_pts, opt_score = _optimize(pts_init, max_time=10.0)
                if opt_score > best_score:
                    best_score = opt_score
                    best_pts = opt_pts.copy()
                    
    # 2. Try Random Initialization with Annealing
    # Random starts are less likely to be good, but might escape local minima of hex.
    for _ in range(3):
        if time.time() > DEADLINE - 5.0:
            break
        pts_init = _initialize_random()
        # Project to disk
        radii = np.linalg.norm(pts_init, axis=1)
        mask = radii > 1.0
        if np.any(mask):
            pts_init[mask] = pts_init[mask] / radii[mask, np.newaxis]
            
        score_init = _score(pts_init)
        if score_init > best_score:
            best_score = score_init
            best_pts = pts_init.copy()
            
        opt_pts, opt_score = _optimize(pts_init, max_time=10.0)
        if opt_score > best_score:
            best_score = opt_score
            best_pts = opt_pts.copy()

    # 3. Final Polish
    # Run one more optimization on the best found so far
    if best_pts is not None and time.time() < DEADLINE - 2.0:
        opt_pts, opt_score = _optimize(best_pts, max_time=5.0)
        if opt_score > best_score:
            best_score = opt_score
            best_pts = opt_pts.copy()

    # Safety check
    if best_pts is None or _score(best_pts) <= 0:
        # Fallback to a very simple valid configuration
        # 50 circles. Just place them on a grid that fits.
        # Radius 0.1 is safe.
        # 10x5 grid?
        # Spacing 0.2.
        pts_fallback = []
        for i in range(10):
            for j in range(5):
                x = (i - 4.5) * 0.2
                y = (j - 2.0) * 0.2
                if np.hypot(x, y) < 0.9:
                    pts_fallback.append([x, y])
        if len(pts_fallback) < N:
             # Fill rest with random
             while len(pts_fallback) < N:
                 x, y = np.random.uniform(-0.8, 0.8, 2)
                 if np.hypot(x,y) < 0.8:
                     pts_fallback.append([x,y])
        best_pts = np.array(pts_fallback[:N])
        
    return best_pts

# EVOLVE-BLOCK-END
