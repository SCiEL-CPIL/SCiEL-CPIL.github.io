"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
import itertools

def _score(P):
    """Calculate the radius for a given set of centers P (N, 3)."""
    # P is (N, 3)
    # Wall distances
    # min_i min_k (0.5 - |P[i, k]|)
    wall_dists = 0.5 - np.abs(P)
    min_wall = np.min(wall_dists)
    
    # Pairwise distances
    # To avoid O(N^2) memory for large N, but N=50 is small.
    # Compute all pairwise distances
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    # Zero out diagonal
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    return min(min_wall, min_pair)

def _smooth_hinge(x, eps=1e-4):
    """Smooth approximation of max(0, x)."""
    return 0.5 * (x + np.sqrt(x**2 + eps**2))

def _objective_and_grad(params, beta=10.0):
    """
    Objective function for minimization.
    params: [P (N*3), r]
    Minimize: -r + penalty(violations)
    """
    N_pts = len(params) - 1
    P = params[:N_pts].reshape(-1, 3)
    r = params[-1]
    
    # Wall violations: r + |c_ik| - 0.5 > 0
    wall_viol = r + np.abs(P) - 0.5
    
    # Pair violations: 2r - dist(i,j) > 0
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    pair_viol = 2.0 * r - dists
    
    # Apply smooth hinge
    # We sum the squared smooth hinges to make it differentiable and smooth
    # Actually, sum of smooth hinges is enough, squaring makes it stronger
    # Let's use sum of smooth hinges for stability
    penalty = np.sum(_smooth_hinge(wall_viol, beta)) + np.sum(_smooth_hinge(pair_viol, beta))
    
    val = -r + penalty
    
    # Gradient calculation
    # This is tricky with vectorized smooth hinge. 
    # For L-BFGS-B, we can use numerical gradients if analytical is too complex/slow to derive correctly in code.
    # But N=50, 151 vars. Numerical gradient is 151 function evals.
    # Function eval is O(N^2) = 2500.
    # Total 151 * 2500 = 377,500 ops per gradient step.
    # This is acceptable.
    # However, providing analytical gradient is faster.
    # Let's try numerical gradient first for correctness, then optimize if needed.
    # scipy.optimize.minimize with L-BFGS-B uses numerical gradients by default if jac is None.
    
    return val

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    best_score = -1.0
    best_P = None
    
    # Initial configurations
    # 1. Grid with jitter
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    base_grid = grid[:N]
    
    inits = []
    # Add jittered grid
    for scale in [0.01, 0.05, 0.1]:
        jitter = np.random.uniform(-scale, scale, size=base_grid.shape)
        P_init = np.clip(base_grid + jitter, -0.5, 0.5)
        inits.append(P_init)
        
    # Add random inits
    for _ in range(5):
        P_rand = np.random.uniform(-0.4, 0.4, size=(N, 3))
        inits.append(P_rand)

    # Optimization loop
    for i, P_init in enumerate(inits):
        if time.time() > deadline:
            break
            
        # Initial r estimate
        r_init = _score(P_init) * 0.9
        
        # Parameters: P flattened + r
        x0 = np.concatenate([P_init.flatten(), [r_init]])
        
        # Bounds: P in [-0.5, 0.5], r in [0, 0.5]
        bounds_P = [(-0.5, 0.5)] * (N * 3)
        bounds_r = [(0.0, 0.5)]
        bounds = bounds_P + bounds_r
        
        # Optimize
        try:
            # Use L-BFGS-B
            # Max iterations to prevent hanging
            res = minimize(
                _objective_and_grad, 
                x0, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-10, 'gtol': 1e-8}
            )
        except Exception:
            continue
            
        if res.success or res.status == 0:
            P_opt = res.x[:N*3].reshape(N, 3)
            r_opt = res.x[-1]
            
            # Refine: The optimization maximizes a penalized objective.
            # We should calculate the true score of this configuration.
            true_score = _score(P_opt)
            
            # Sometimes the penalty allows r to be slightly larger than the true min dist.
            # We can do a local "push" to ensure validity and potentially improve.
            # But _score is the ground truth.
            
            if true_score > best_score:
                best_score = true_score
                best_P = P_opt.copy()
                
    # Final polish: Simple local search on best_P
    # Try to nudge particles apart if they are too close
    if best_P is not None and time.time() < deadline - 5:
        P = best_P.copy()
        for _ in range(100):
            if time.time() > deadline - 5:
                break
            score = _score(P)
            # Find closest pair
            diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dists, np.inf)
            
            min_dist = np.min(dists)
            idx = np.unravel_index(np.argmin(dists), dists.shape)
            i, j = idx
            
            # Move i and j apart
            dir_vec = P[i] - P[j]
            norm = np.linalg.norm(dir_vec)
            if norm > 1e-6:
                dir_vec /= norm
                # Small step
                step = 0.001
                P[i] += dir_vec * step
                P[j] -= dir_vec * step
            else:
                # Random kick
                P[i] += np.random.uniform(-0.01, 0.01, 3)
                P[j] += np.random.uniform(-0.01, 0.01, 3)
            
            # Clip to cube
            P = np.clip(P, -0.5, 0.5)
            
            new_score = _score(P)
            if new_score > score:
                best_score = new_score
                best_P = P.copy()
            else:
                # Revert if worse? Or keep if it's a saddle point?
                # For max-min, if we move apart, min dist should increase unless another pair becomes smaller.
                # If new_score <= score, we might have broken another constraint.
                # Revert to be safe.
                P = best_P.copy()
                break

    if best_P is None:
        # Fallback to grid
        return base_grid
        
    return best_P

# EVOLVE-BLOCK-END
