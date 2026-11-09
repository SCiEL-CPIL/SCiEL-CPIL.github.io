"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill

    # Precompute pair indices for distance computation
    n_pairs = N * (N - 1) // 2
    pair_i = np.zeros(n_pairs, dtype=int)
    pair_j = np.zeros(n_pairs, dtype=int)
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            pair_i[idx] = i
            pair_j[idx] = j
            idx += 1

    def compute_radius(x):
        """Compute the radius for a given configuration x (flattened 2N array)."""
        pts = x.reshape(N, 2)
        # Pairwise distances
        di = pts[pair_i]
        dj = pts[pair_j]
        diffs = di - dj
        dists = np.sqrt(np.sum(diffs * diffs, axis=1))
        min_pair_dist = np.min(dists)
        
        # Wall distances
        wall_dists = np.min(np.stack([
            0.5 - np.abs(pts[:, 0]),
            0.5 - np.abs(pts[:, 1])
        ], axis=1), axis=1)
        min_wall_dist = np.min(wall_dists)
        
        return min(min_pair_dist / 2.0, min_wall_dist)

    def objective(x):
        """Negative radius to minimize."""
        return -compute_radius(x)

    def objective_grad(x):
        """Gradient of negative radius (approximate via finite differences for SLSQP)."""
        # For better performance, we'll use analytical gradient if possible,
        # but for now use numerical
        eps = 1e-7
        grad = np.zeros_like(x)
        f0 = objective(x)
        for k in range(len(x)):
            xk = x.copy()
            xk[k] += eps
            grad[k] = (objective(xk) - f0) / eps
        return grad

    # Initial grid configuration
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    x0 = grid[:N].flatten()
    
    # Add small random perturbation to break symmetry
    rng = np.random.default_rng(42)
    x0 = x0 + rng.normal(0, 0.001, size=x0.shape)
    
    # Clip to valid range
    x0 = np.clip(x0, -0.4999, 0.4999)
    
    best_x = x0.copy()
    best_r = compute_radius(x0)
    
    # Try multiple optimization approaches
    max_iter_per_run = 200
    
    # Approach 1: SLSQP with bounds
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    # Try a few restarts with different initial conditions
    for restart in range(5):
        if time.time() > deadline - 20.0:
            break
            
        if restart == 0:
            x_start = x0.copy()
        else:
            # Perturb the best solution found so far
            x_start = best_x + rng.normal(0, 0.01, size=2*N)
            x_start = np.clip(x_start, -0.4999, 0.4999)
        
        try:
            result = minimize(
                objective,
                x_start,
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': max_iter_per_run, 'ftol': 1e-10}
            )
            
            if result.success or result.fun < objective(best_x):
                r = compute_radius(result.x)
                if r > best_r:
                    best_r = r
                    best_x = result.x.copy()
        except Exception:
            pass
    
    # Approach 2: Nelder-Mead for local polish
    if time.time() < deadline - 10.0:
        try:
            result = minimize(
                objective,
                best_x,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            r = compute_radius(result.x)
            if r > best_r:
                best_r = r
                best_x = result.x.copy()
        except Exception:
            pass
    
    # Final polish with very tight tolerance
    if time.time() < deadline - 5.0:
        try:
            result = minimize(
                objective,
                best_x,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-14}
            )
            r = compute_radius(result.x)
            if r > best_r:
                best_r = r
                best_x = result.x.copy()
        except Exception:
            pass
    
    # Ensure final result is valid (clip to square)
    pts = best_x.reshape(N, 2)
    pts = np.clip(pts, -0.5, 0.5)
    
    return pts
# EVOLVE-BLOCK-END
