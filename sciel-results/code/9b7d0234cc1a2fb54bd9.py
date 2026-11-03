import numpy as np
import time
from scipy.optimize import minimize

N = 600
W = 1.0 / 1200.0

# EVOLVE-BLOCK-START
def _compute_ratio_fast(h):
    """
    Compute max(autoconv) / l2_sq for step function h.
    Uses FFT on a fine grid for accurate peak detection.
    """
    h = np.asarray(h, dtype=np.float64)
    h = np.clip(h, 0.0, None)
    if np.sum(h) < 1e-15:
        return float('inf')
    
    n_grid = 2048  # Power of 2 for fast FFT, sufficient resolution
    x_fine = np.linspace(-0.25, 0.25, n_grid, endpoint=False)
    cell_idx = np.clip(((x_fine + 0.25) / W).astype(np.int32), 0, N - 1)
    f_fine = h[cell_idx]
    
    n_fft = 2 * n_grid  # Next power of 2 >= 2*n_grid - 1
    F = np.fft.rfft(f_fine, n_fft)
    conv = np.fft.irfft(F * F, n_fft)[:2 * n_grid - 1]
    dx = 0.5 / n_grid
    conv *= dx
    max_conv = np.max(conv)
    
    l2_sq = np.sum(f_fine ** 2) * dx
    if l2_sq < 1e-15:
        return float('inf')
    return max_conv / l2_sq

def _gaussian_model(params, x_grid, K):
    """
    Evaluate sum of K Gaussians at x_grid.
    params: [A1, mu1, sigma1, A2, mu2, sigma2, ...]
    """
    h = np.zeros_like(x_grid)
    for j in range(K):
        A = params[3*j]
        mu = params[3*j+1]
        sigma = params[3*j+2]
        
        # Ensure sigma is positive and A is non-negative for non-negativity constraint
        # We use softplus or simple clipping in the optimizer by transforming params
        # Here we assume params are already transformed or we handle it in objective
        if sigma < 1e-6:
            sigma = 1e-6
        if A < 0:
            A = 0.0
            
        h += A * np.exp(-0.5 * ((x_grid - mu) / sigma) ** 2)
    return np.maximum(h, 0.0)

def _objective_gaussian(params, x_grid, K):
    """
    Objective function for scipy.optimize.
    Returns the ratio to be minimized.
    """
    h = _gaussian_model(params, x_grid, K)
    if np.sum(h) < 1e-15:
        return 1e6
    return _compute_ratio_fast(h)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    x_grid = np.linspace(-0.25, 0.25, N, endpoint=False) + W/2.0 # Centers of intervals
    
    best_h = np.ones(N)
    best_ratio = _compute_ratio_fast(best_h)
    
    # Try different numbers of Gaussians
    # K=1 is a single bump, K=2 two bumps, etc.
    # The optimal solution for this problem often involves disjoint support or specific shapes.
    # Gaussians are good for smooth bumps.
    
    np.random.seed(42)
    
    for K in [1, 2, 3, 5]:
        if time.time() > deadline - 20.0:
            break
            
        # Initial guesses for parameters
        # We try multiple random initializations to escape local minima
        for init_iter in range(5):
            if time.time() > deadline - 10.0:
                break
                
            # Initialize parameters
            # A: positive, mu: in [-0.25, 0.25], sigma: positive
            A_init = np.random.rand(K) * 1.0 + 0.1
            mu_init = np.random.uniform(-0.2, 0.2, K)
            sigma_init = np.random.uniform(0.02, 0.2, K)
            
            params_init = np.concatenate([A_init, mu_init, sigma_init])
            
            # Bounds: A in [0, 10], mu in [-0.25, 0.25], sigma in [0.001, 0.5]
            bounds = []
            for j in range(K):
                bounds.append((0.0, 10.0)) # A
                bounds.append((-0.25, 0.25)) # mu
                bounds.append((0.001, 0.5)) # sigma
            
            # Use Powell method which doesn't require gradients and is robust
            # Nelder-Mead is also good for small dimensions
            try:
                res = minimize(_objective_gaussian, params_init, args=(x_grid, K),
                               method='Powell', bounds=bounds,
                               options={'maxiter': 100, 'xatol': 1e-4, 'fatol': 1e-6})
                
                if res.success or res.fun < best_ratio:
                    h_opt = _gaussian_model(res.x, x_grid, K)
                    r_opt = _compute_ratio_fast(h_opt)
                    if r_opt < best_ratio:
                        best_ratio = r_opt
                        best_h = h_opt.copy()
            except Exception:
                pass

    # Local refinement using coordinate descent on the 600 heights
    # This polishes the discrete step function representation
    h = best_h.copy()
    step_size = max(0.01, np.max(h) * 0.01)
    
    # Use a smaller number of iterations since we start from a good parametric solution
    for iteration in range(200):
        if time.time() > deadline - 5.0:
            break
        
        improved = False
        # Try perturbing a subset of coordinates
        n_trials = 50
        idxs = np.random.choice(N, size=n_trials, replace=False)
        
        for idx in idxs:
            h_old = h[idx]
            
            # Try increasing
            h[idx] = h_old + step_size
            r_plus = _compute_ratio_fast(h)
            
            # Try decreasing
            h[idx] = h_old - step_size
            r_minus = _compute_ratio_fast(h)
            
            h[idx] = h_old
            
            if r_plus < best_ratio:
                h[idx] = h_old + step_size
                best_ratio = r_plus
                improved = True
            elif r_minus < best_ratio and h[idx] > 0:
                h[idx] = h_old - step_size
                best_ratio = r_minus
                improved = True
        
        if not improved:
            step_size *= 0.8
            if step_size < 1e-6:
                break
    
    # Final evaluation
    final_r = _compute_ratio_fast(h)
    if final_r < best_ratio:
        best_ratio = final_r
        best_h = h.copy()
    
    return best_h.tolist()
# EVOLVE-BLOCK-END
