"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16
D = 2

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    # Objective: Minimize (max_dist / min_dist)^2
    # Strategy:
    # 1. The function f(x) = (max/min)^2 is non-smooth and has flat regions/gradients issues.
    # 2. A good proxy is to make all pairwise distances as equal as possible.
    #    If all distances are equal, max/min = 1, score = 1.
    #    We minimize the variance of the squared distances.
    #    Let d_ij^2 be squared distances.
    #    Objective: Sum (d_ij^2 - mean(d^2))^2
    #    This is smooth and differentiable.
    # 3. We use L-BFGS-B or Nelder-Mead. L-BFGS-B is faster if gradients are available.
    #    We can compute gradients analytically or use finite differences.
    #    Given 32 dimensions (16 points * 2), finite differences are acceptable if the function is fast.
    #    However, L-BFGS-B with numerical gradients might be slow if too many function calls.
    #    Let's try Nelder-Mead first as it is derivative-free and robust for low-dim.
    #    Actually, let's implement a custom gradient for the variance objective.
    
    def get_dists(pts):
        # Returns array of squared distances for i < j
        # pts: (N, 2)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        sq_dists = np.sum(diff**2, axis=2)
        iu = np.triu_indices(N, 1)
        return sq_dists[iu]

    def score_raw(pts):
        d_sq = get_dists(pts)
        d_min = np.sqrt(d_sq.min())
        d_max = np.sqrt(d_sq.max())
        if d_min < 1e-12:
            return 1e10
        return (d_max / d_min) ** 2

    def objective_variance(pts):
        # Minimize variance of squared distances
        d_sq = get_dists(pts)
        mean_d_sq = np.mean(d_sq)
        # Sum of squared deviations
        return np.sum((d_sq - mean_d_sq) ** 2)

    def gradient_variance(pts):
        # Analytical gradient of sum((d_ij^2 - mean)^2)
        # Let S = sum_k (d_k^2 - m)^2 where m = (1/M) sum_k d_k^2
        # dS/dp_i = sum_k 2(d_k^2 - m) * d(d_k^2)/dp_i
        # d(d_ij^2)/dp_i = 2(p_i - p_j) for k=(i,j)
        # d(d_ij^2)/dp_j = 2(p_j - p_i) for k=(i,j)
        # dm/dp_i = (1/M) sum_k d(d_k^2)/dp_i
        
        d_sq = get_dists(pts)
        m = np.mean(d_sq)
        M = len(d_sq)
        
        # We need to map indices back to point pairs
        iu = np.triu_indices(N, 1)
        i_idx, j_idx = iu
        
        grad = np.zeros_like(pts)
        
        # Term 1: sum_k 2(d_k^2 - m) * 2(p_i - p_j)
        # For each pair k, add to grad[i] and grad[j]
        # coeff_k = 4 * (d_sq[k] - m)
        coeffs = 4.0 * (d_sq - m)
        
        # Accumulate gradients
        # np.add.at is good for this
        np.add.at(grad, i_idx, coeffs[:, np.newaxis] * (pts[i_idx] - pts[j_idx]))
        np.add.at(grad, j_idx, coeffs[:, np.newaxis] * (pts[j_idx] - pts[i_idx]))
        
        # Term 2: - sum_k 2(d_k^2 - m) * (1/M) * sum_l d(d_l^2)/dp_i
        # This is effectively: - (2/M) * (sum_k (d_k^2 - m)) * (sum_l d(d_l^2)/dp_i)
        # Note: sum_k (d_k^2 - m) = 0. 
        # Wait, sum(d_sq - m) is exactly 0 by definition of mean.
        # So the gradient of the mean term is zero?
        # Let's check.
        # f = sum (d_k^2 - m)^2
        # df/dp_i = sum 2(d_k^2 - m) * (2(d_k^2)' ) - 2 * sum(d_k^2 - m) * (dm/dp_i)
        # Since sum(d_k^2 - m) = 0, the second term vanishes.
        # So the gradient is just the first part.
        
        return grad

    # Initial setup: 
    # A regular grid is a decent start, but symmetric structures like hexagonal packing or 
    # specific polytopes might be better.
    # Let's try a few random restarts and a grid start.
    
    best_pts = None
    best_score = 1e10
    best_var_score = 1e10
    
    rng = np.random.default_rng(42)
    
    def try_optimize(init_pts):
        nonlocal best_pts, best_score, best_var_score
        
        # Normalize initial points to unit scale to help optimization
        init_pts = init_pts.copy()
        init_pts -= np.mean(init_pts, axis=0)
        scale = np.sqrt(np.mean(np.sum(init_pts**2, axis=1)))
        if scale > 1e-9:
            init_pts /= scale
            
        # Use L-BFGS-B with analytical gradient
        # Bounds: keep points in a reasonable box to prevent explosion
        # Since we re-center, scale is somewhat controlled, but let's bound coords to [-10, 10]
        bounds = [(-10, 10)] * (N * 2)
        
        try:
            res = minimize(
                objective_variance,
                init_pts.flatten(),
                jac=gradient_variance,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if res.success or res.fun < best_var_score:
                # Reconstruct points
                cand_pts = res.x.reshape(N, 2)
                # Re-center and scale to unit variance for consistent scoring
                cand_pts -= np.mean(cand_pts, axis=0)
                scale = np.sqrt(np.mean(np.sum(cand_pts**2, axis=1)))
                if scale > 1e-9:
                    cand_pts /= scale
                
                var_score = res.fun
                raw_score = score_raw(cand_pts)
                
                if raw_score < best_score:
                    best_score = raw_score
                    best_pts = cand_pts.copy()
                    best_var_score = var_score
                    
        except Exception:
            pass

    # Generate diverse starting points
    # 1. Grid
    side = int(np.ceil(N ** (1 / D)))
    grid = []
    for i in range(side):
        for j in range(side):
            grid.append([i, j])
            if len(grid) == N:
                break
        if len(grid) == N:
            break
    try_optimize(np.array(grid, dtype=float))
    
    # 2. Random points in a circle
    for _ in range(10):
        if time.time() > deadline:
            break
        angles = rng.uniform(0, 2 * np.pi, N)
        radii = rng.uniform(0, 1, N)
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        try_optimize(pts)
        
    # 3. Random points in a square
    for _ in range(10):
        if time.time() > deadline:
            break
        pts = rng.uniform(-1, 1, (N, 2))
        try_optimize(pts)

    # If optimization didn't improve much, fallback to a known good configuration?
    # For 16 points, the optimal is likely related to a hexagonal packing subset or a specific lattice.
    # The 4x4 grid score is ~18.
    # A hexagonal packing of 16 points might be better.
    # Let's try to construct a hexagonal packing explicitly as a seed if needed.
    
    if best_pts is None:
        # Fallback to grid
        side = int(np.ceil(N ** (1 / D)))
        grid = []
        for i in range(side):
            for j in range(side):
                grid.append([i, j])
                if len(grid) == N:
                    break
            if len(grid) == N:
                break
        best_pts = np.array(grid, dtype=float)
        best_pts -= np.mean(best_pts, axis=0)
        scale = np.sqrt(np.mean(np.sum(best_pts**2, axis=1)))
        if scale > 1e-9:
            best_pts /= scale
        best_score = score_raw(best_pts)

    return best_pts
# EVOLVE-BLOCK-END
