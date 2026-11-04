import numpy as np
import time
from scipy.optimize import differential_evolution

N = 600
W = 1.0 / 1200.0

def boxcar_to_heights(params, N=600, W=1.0/1200.0):
    """Convert boxcar parameters to step heights.
    params: array of shape (3*K,) where each triplet is (center, half_width, amplitude)
    """
    K = len(params) // 3
    h = np.zeros(N)
    x = np.linspace(-0.25, 0.25, N, endpoint=False)
    for i in range(K):
        center = params[3*i]
        half_width = abs(params[3*i+1])
        amp = max(0.0, params[3*i+2])
        mask = np.abs(x - center) < half_width
        h[mask] += amp
    return np.clip(h, 0.0, None)

def compute_ratio(h, n_grid=2048):
    """Compute max(autoconv) / l2_sq for step function h."""
    h = np.asarray(h, dtype=np.float64)
    h = np.clip(h, 0.0, None)
    if np.sum(h) < 1e-15:
        return float('inf')

    # Upsample to a fine grid for accurate convolution peak detection
    x_fine = np.linspace(-0.25, 0.25, n_grid, endpoint=False)
    cell_idx = np.clip(((x_fine + 0.25) / W).astype(np.int32), 0, N - 1)
    f_fine = h[cell_idx]

    # Linear autoconvolution via FFT (zero-padded)
    n_fft = 1 << (2 * n_grid).bit_length()
    F = np.fft.rfft(f_fine, n_fft)
    conv = np.fft.irfft(F * F, n_fft)[:2 * n_grid - 1]
    dx = 0.5 / n_grid
    conv *= dx
    max_conv = np.max(conv)

    l2_sq = np.sum(f_fine ** 2) * dx
    if l2_sq < 1e-15:
        return float('inf')
    return max_conv / l2_sq

def objective_boxcar(params, deadline):
    """Objective function for boxcar optimization."""
    if time.time() > deadline:
        return 1e10
    h = boxcar_to_heights(params)
    return compute_ratio(h, n_grid=1024)

def construct():
    deadline = time.time() + 145.0  # 5s safety margin
    
    # Number of boxcars to try
    K = 3
    
    # Bounds for each parameter: (center, half_width, amplitude)
    # center in [-0.25, 0.25], half_width in [0.001, 0.25], amplitude in [0, 10]
    bounds = []
    for _ in range(K):
        bounds.append((-0.25, 0.25))   # center
        bounds.append((0.001, 0.25))   # half_width
        bounds.append((0.0, 10.0))     # amplitude
    
    # Initial best: constant function
    best_h = np.ones(N)
    best_ratio = compute_ratio(best_h)
    
    # Run differential evolution
    try:
        result = differential_evolution(
            objective_boxcar,
            bounds,
            args=(deadline,),
            maxiter=100,
            tol=1e-5,
            popsize=20,
            seed=42,
            polish=False,
            init='sobol'
        )
        
        if result.fun < best_ratio:
            h_opt = boxcar_to_heights(result.x)
            r = compute_ratio(h_opt, n_grid=2048)
            if r < best_ratio:
                best_ratio = r
                best_h = h_opt.copy()
    except Exception:
        pass
    
    # Try with different numbers of boxcars if time permits
    for K_try in [2, 4, 5]:
        if time.time() > deadline - 10:
            break
        
        bounds_try = []
        for _ in range(K_try):
            bounds_try.append((-0.25, 0.25))
            bounds_try.append((0.001, 0.25))
            bounds_try.append((0.0, 10.0))
        
        try:
            result_try = differential_evolution(
                objective_boxcar,
                bounds_try,
                args=(deadline,),
                maxiter=50,
                tol=1e-4,
                popsize=15,
                seed=42+K_try,
                polish=False,
                init='latinhypercube'
            )
            
            if result_try.fun < best_ratio:
                h_opt = boxcar_to_heights(result_try.x)
                r = compute_ratio(h_opt, n_grid=2048)
                if r < best_ratio:
                    best_ratio = r
                    best_h = h_opt.copy()
        except Exception:
            pass
    
    # Final refinement: local optimization of the best boxcar configuration
    if time.time() < deadline - 5:
        from scipy.optimize import minimize
        
        # Reconstruct initial params from best_h by fitting
        # For now, just do a local search around the best found
        if hasattr(result, 'x') and result.fun < best_ratio:
            def local_obj(params):
                if time.time() > deadline:
                    return 1e10
                h = boxcar_to_heights(params)
                return compute_ratio(h, n_grid=2048)
            
            try:
                res_local = minimize(
                    local_obj,
                    result.x,
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-7}
                )
                
                if res_local.fun < best_ratio:
                    h_opt = boxcar_to_heights(res_local.x)
                    r = compute_ratio(h_opt, n_grid=2048)
                    if r < best_ratio:
                        best_ratio = r
                        best_h = h_opt.copy()
            except Exception:
                pass
    
    return best_h.tolist()
