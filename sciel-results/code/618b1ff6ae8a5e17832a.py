"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

def _compute_radius(pts):
    """Compute the valid radius for a configuration of points in [-0.5, 0.5]^3."""
    if len(pts) != N:
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    # Check bounds
    if np.any(pts < -0.5) or np.any(pts > 0.5):
        return 0.0
    # Wall distance
    wall = np.min(np.min(0.5 - np.abs(pts), axis=1), axis=0)
    # Pairwise distances
    from scipy.spatial.distance import cdist
    D = cdist(pts, pts)
    # Minimum pairwise distance (excluding diagonal)
    np.fill_diagonal(D, np.inf)
    min_pair = np.min(D)
    return min(wall, min_pair / 2.0)

def _soft_min_radius(pts, beta=50.0):
    """
    Soft surrogate for the radius: use log-sum-exp style soft-min.
    r = min(wall_dist, min_pair/2)
    We want to maximize r.
    
    Let f_i be the constraint values (wall dist for each point, pair dist/2 for each pair).
    We want max_t s.t. t <= f_i for all i.
    
    Soft version: use a smooth approximation.
    For pairs: soft_min of (d_ij/2) over all i<j
    For walls: soft_min of (0.5 - |x_ik|) over all i,k
    
    Actually, let's define:
    - For each point i and dimension k: wall_ik = 0.5 - |x_ik|
    - For each pair (i,j): pair_ij = ||x_i - x_j|| / 2
    
    r = min(all wall_ik, all pair_ij)
    
    Soft-min via log-sum-exp: soft_min(z) = -1/beta * log(sum(exp(-beta*z)))
    This is a smooth lower bound approximation.
    """
    pts = np.asarray(pts, dtype=np.float64)
    n = len(pts)
    
    # Wall distances: shape (n, 3)
    walls = 0.5 - np.abs(pts)  # (n, 3)
    walls_flat = walls.flatten()
    
    # Pair distances / 2
    from scipy.spatial.distance import cdist
    D = cdist(pts, pts)  # (n, n)
    np.fill_diagonal(D, np.inf)
    pairs = D / 2.0
    # Get upper triangle
    iu = np.triu_indices(n, k=1)
    pairs_flat = pairs[iu]
    
    # Combine all constraint values
    all_vals = np.concatenate([walls_flat, pairs_flat])
    
    # Soft min: -1/beta * log(sum(exp(-beta * vals)))
    # Use stable computation
    beta_safe = beta
    min_val = np.min(all_vals)
    exp_terms = np.exp(-beta_safe * (all_vals - min_val))
    soft_min = min_val - np.log(np.sum(exp_terms)) / beta_safe
    
    return soft_min

def _soft_min_radius_grad(pts, beta=50.0):
    """
    Compute soft-min radius and its gradient w.r.t. pts.
    Returns (value, grad) where grad has shape (n, 3).
    """
    pts = np.asarray(pts, dtype=np.float64)
    n = len(pts)
    
    # Wall distances
    walls = 0.5 - np.abs(pts)  # (n, 3)
    walls_flat = walls.flatten()
    
    # Pair distances / 2
    from scipy.spatial.distance import cdist
    D = cdist(pts, pts)
    np.fill_diagonal(D, np.inf)
    pairs = D / 2.0
    iu = np.triu_indices(n, k=1)
    pairs_flat = pairs[iu]
    
    all_vals = np.concatenate([walls_flat, pairs_flat])
    
    # Soft min computation
    min_val = np.min(all_vals)
    shifted = all_vals - min_val
    exp_terms = np.exp(-beta * shifted)
    sum_exp = np.sum(exp_terms)
    soft_min = min_val - np.log(sum_exp) / beta
    
    # Gradient
    # d(soft_min)/d(vals) = -exp_terms / sum_exp
    # Note: soft_min = min_val - log(sum_exp)/beta
    # But min_val is not differentiable in general. However, in the soft-min formulation:
    # soft_min(z) = -1/beta * log(sum(exp(-beta * z)))
    # d(soft_min)/d(z_i) = exp(-beta*z_i) / sum(exp(-beta*z_j)) = exp_terms[i] / sum_exp
    
    # Wait, let me re-derive:
    # f(z) = -1/beta * log(sum_j exp(-beta * z_j))
    # df/dz_i = -1/beta * (1/sum) * (-beta * exp(-beta*z_i)) = exp(-beta*z_i) / sum
    
    weights = exp_terms / sum_exp  # shape (len(all_vals),)
    
    grad = np.zeros_like(pts)
    
    # Wall gradient contribution
    # walls_flat[i*3 + k] = 0.5 - |pts[i,k]|
    # d(walls_flat[i*3+k])/d(pts[i,k]) = -sign(pts[i,k])
    # grad[pts] += sum_k weights[k] * d(wall)/d(pts)
    
    wall_weights = weights[:n*3].reshape(n, 3)
    # d(0.5 - |x|)/dx = -sign(x)
    grad -= wall_weights * np.sign(pts)
    # Handle x=0: subgradient, use 0
    zero_mask = (pts == 0)
    grad[zero_mask] = 0
    
    # Pair gradient contribution
    # pairs_flat[m] = ||pts[i_m] - pts[j_m]|| / 2
    # d(pairs_flat[m])/d(pts[i_m]) = (pts[i_m] - pts[j_m]) / (2 * ||pts[i_m] - pts[j_m]||)
    # d(pairs_flat[m])/d(pts[j_m]) = (pts[j_m] - pts[i_m]) / (2 * ||pts[i_m] - pts[j_m]||)
    
    pair_weights = weights[n*3:]
    i_idx = iu[0]
    j_idx = iu[1]
    diff = pts[i_idx] - pts[j_idx]  # (n_pairs, 3)
    dists = np.linalg.norm(diff, axis=1, keepdims=True)  # (n_pairs, 1)
    # Avoid division by zero
    dists = np.where(dists > 1e-12, dists, 1.0)
    
    # d(pair_m)/d(pts[i_m]) = diff_m / (2 * dist_m)
    # d(pair_m)/d(pts[j_m]) = -diff_m / (2 * dist_m)
    
    grad_pair_i = (pair_weights[:, np.newaxis] * diff) / (2.0 * dists)
    grad_pair_j = -grad_pair_i
    
    np.add.at(grad, i_idx, grad_pair_i)
    np.add.at(grad, j_idx, grad_pair_j)
    
    return soft_min, grad

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    np.random.seed(42)
    
    # Initialize with a reasonable starting configuration
    # Use FCC-like lattice or random with rejection
    best_pts = None
    best_r = -np.inf
    
    # Try multiple initializations
    for init_iter in range(20):
        if time.time() > deadline - 60:
            break
        
        # Random initialization with some structure
        if init_iter == 0:
            # Simple grid-like start
            m = 5
            ticks = (np.arange(m) + 0.5) / m - 0.5
            grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
            pts = grid[:N].copy()
        else:
            # Random initialization
            pts = np.random.uniform(-0.45, 0.45, (N, 3))
        
        # Ensure valid initial config (push inside if needed)
        pts = np.clip(pts, -0.49, 0.49)
        
        # Optimize using gradient ascent on soft-min radius
        beta_schedule = [20.0, 50.0, 100.0, 200.0, 500.0]
        for beta in beta_schedule:
            if time.time() > deadline - 30:
                break
            
            lr = 0.01
            for step in range(500):
                if time.time() > deadline - 10:
                    break
                
                val, grad = _soft_min_radius_grad(pts, beta=beta)
                
                # Gradient ascent
                pts = pts + lr * grad
                
                # Project back to cube
                pts = np.clip(pts, -0.49, 0.49)
                
                # Compute actual radius
                r = _compute_radius(pts)
                if r > best_r:
                    best_r = r
                    best_pts = pts.copy()
        
        # Local refinement with smaller beta for precision
        if time.time() > deadline - 15:
            break
        
        for beta in [500.0, 1000.0]:
            lr = 0.001
            for step in range(200):
                if time.time() > deadline - 5:
                    break
                
                val, grad = _soft_min_radius_grad(pts, beta=beta)
                pts = pts + lr * grad
                pts = np.clip(pts, -0.49, 0.49)
                
                r = _compute_radius(pts)
                if r > best_r:
                    best_r = r
                    best_pts = pts.copy()
    
    # Final polish: coordinate descent / local search
    if best_pts is not None and time.time() < deadline - 2:
        pts = best_pts.copy()
        for polish_iter in range(100):
            if time.time() > deadline - 1:
                break
            
            improved = False
            for i in range(N):
                if time.time() > deadline - 1:
                    break
                for attempt in range(10):
                    # Try small random perturbations
                    delta = np.random.normal(0, 0.001, 3)
                    new_pts = pts.copy()
                    new_pts[i] += delta
                    new_pts[i] = np.clip(new_pts[i], -0.49, 0.49)
                    
                    r_new = _compute_radius(new_pts)
                    if r_new > best_r + 1e-12:
                        pts = new_pts
                        best_r = r_new
                        best_pts = pts.copy()
                        improved = True
                        break
            
            if not improved:
                break
    
    if best_pts is None:
        # Fallback: simple grid
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_pts = grid[:N]
    
    # Final validation
    r_final = _compute_radius(best_pts)
    if r_final <= 0:
        # Fallback
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_pts = grid[:N]
    
    return best_pts
# EVOLVE-BLOCK-END
