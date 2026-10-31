import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

N = 600
DT = (1/4 - (-1/4)) / N  # interval width = 1/600
X_GRID = -1/4 + (np.arange(N) + 0.5) * DT  # midpoints of intervals
TIME_LIMIT = 145.0  # safety margin before 150s kill

def _compute_score(h):
    """Compute normalized maximum autoconvolution."""
    h = np.maximum(h, 0.0)
    s = np.sum(h)
    if s < 1e-15:
        return 1e10
    hn = h / s
    F = np.fft.fft(hn, n=2*N)
    conv = np.fft.ifft(F * F).real[:2*N-1]
    return np.max(conv)

def _cosine_bump_heights(params, n_components):
    """Convert parameter vector to step heights using cosine bumps.
    
    params layout: [amp_1, center_1, width_1, amp_2, center_2, width_2, ...]
    Each bump is amp * cos^2(pi * (x - center) / width) for |x-center| <= width/2
    """
    h = np.zeros(N)
    for k in range(n_components):
        amp = params[3*k]
        center = params[3*k+1]
        width = params[3*k+2]
        if width < 0.01:
            width = 0.01
        width = min(width, 0.5)
        dist = np.abs(X_GRID - center)
        mask = dist <= width / 2.0
        if np.any(mask):
            h[mask] += amp * np.cos(np.pi * (X_GRID[mask] - center) / width) ** 2
    h = np.maximum(h, 0)
    s = np.sum(h)
    if s < 1e-15:
        return np.ones(N) / N
    return h / s

def _gaussian_mixture_heights(params, n_components):
    """Convert parameter vector to step heights using Gaussian mixture."""
    h = np.zeros(N)
    for k in range(n_components):
        amp = params[3*k]
        center = params[3*k+1]
        width = params[3*k+2]
        if width < 1e-6:
            width = 1e-6
        h += amp * np.exp(-0.5 * ((X_GRID - center) / width) ** 2)
    h = np.maximum(h, 0)
    s = np.sum(h)
    if s < 1e-15:
        return np.ones(N) / N
    return h / s

def _smooth_max(x, beta):
    """Smooth approximation to max using log-sum-exp."""
    x_max = np.max(x)
    if x_max < 1e-15:
        return 0.0
    shifted = x - x_max
    log_sum = np.log(np.sum(np.exp(beta * shifted))) / beta
    return x_max + log_sum

def _sensitivity_refinement(h, start_time, max_time, threshold_frac=0.95):
    """Local refinement using sensitivity-based peak reduction.
    
    At each step, identify lags where autoconvolution is near its max,
    compute how much each bin contributes to those peaks, and reduce bins
    that contribute more to peaks than to off-peak regions.
    """
    best_h = h.copy()
    best_val = _compute_score(h)
    current_h = h.copy()
    alpha = 0.05
    
    while time.time() - start_time < max_time:
        s = np.sum(current_h)
        if s < 1e-12:
            break
        # Normalize for stable computation
        current_h_norm = current_h / s
        
        F = np.fft.fft(current_h_norm, n=2*N)
        conv = np.fft.ifft(F * F).real[:2*N-1]
        max_c = np.max(conv)
        current_val = max_c
        
        if current_val < best_val:
            best_val = current_val
            best_h = current_h.copy()
        
        # Find lags where conv is close to max
        threshold = max_c * threshold_frac
        peak_indices = np.where(conv >= threshold)[0]
        
        if len(peak_indices) == 0:
            break
        
        # Compute sensitivity: for each bin j, how much does it contribute to peak lags?
        # conv[m] = sum_j hn[j] * hn[m-j] (circular, but we use linear approximation)
        # d(conv[m])/d(hn[j]) = hn[m-j]
        sensitivity = np.zeros(N)
        for m in peak_indices:
            # m ranges from 0 to 2N-2
            # j ranges such that m-j is in [0, N-1]
            j_min = max(0, m - (N - 1))
            j_max = min(N - 1, m)
            if j_max >= j_min:
                sensitivity[j_min:j_max+1] += current_h_norm[m - np.arange(j_min, j_max+1)]
        
        # Normalize sensitivity
        sens_mean = np.mean(sensitivity)
        sens_std = np.std(sensitivity) + 1e-12
        sens_norm = (sensitivity - sens_mean) / sens_std
        
        # Reduce bins with high sensitivity
        h_new = current_h * (1.0 - alpha * sens_norm)
        h_new = np.maximum(h_new, 0.0)
        
        s_new = np.sum(h_new)
        if s_new < 1e-12:
            break
        h_new = h_new / s_new
        
        val_new = _compute_score(h_new)
        
        if val_new < current_val:
            current_h = h_new
            alpha = min(alpha * 1.05, 0.5)
        else:
            alpha *= 0.85
            if alpha < 1e-8:
                break
    
    return best_h, best_val

def construct():
    start_time = time.time()
    
    def elapsed():
        return time.time() - start_time
    
    # Start with uniform baseline
    h_uniform = np.ones(N)
    best_h = h_uniform.copy()
    best_val = _compute_score(best_h)
    
    # Try some simple structural candidates first
    candidates = [
        np.ones(N),
        np.clip(1.0 - 4.0 * np.abs(X_GRID), 0, None),  # triangular
        np.exp(-10.0 * X_GRID**2),  # narrow Gaussian
        np.exp(-2.0 * X_GRID**2),   # wide Gaussian
        np.clip(1.0 - 2.0 * np.abs(X_GRID), 0, None),  # wide triangle
        np.cos(np.pi * X_GRID / 0.5)**2,  # cosine bump
    ]
    
    for h_cand in candidates:
        if elapsed() > TIME_LIMIT - 30:
            break
        h_cand = np.maximum(h_cand, 0)
        s = np.sum(h_cand)
        if s < 1e-15:
            continue
        h_cand = h_cand / s
        val = _compute_score(h_cand)
        if val < best_val:
            best_val = val
            best_h = h_cand.copy()
    
    # Phase 2: Differential evolution on cosine bump parameters
    n_components = 5
    n_params = 3 * n_components
    
    # Bounds: amplitude in [0, 3], center in [-0.25, 0.25], width in [0.02, 0.5]
    bounds = []
    for _ in range(n_components):
        bounds.append((0.0, 3.0))    # amplitude
        bounds.append((-0.25, 0.25)) # center
        bounds.append((0.02, 0.5))   # width
    
    def objective_cosine(params):
        if elapsed() > TIME_LIMIT - 10:
            return 1e10
        h = _cosine_bump_heights(params, n_components)
        return _compute_score(h)
    
    de_result = None
    try:
        result = differential_evolution(
            objective_cosine,
            bounds,
            maxiter=150,
            popsize=12,
            tol=1e-6,
            mutation=(0.5, 1.0),
            recombination=0.7,
            seed=42,
            polish=True
        )
        de_result = result
        
        h_result = _cosine_bump_heights(result.x, n_components)
        val_result = _compute_score(h_result)
        
        if val_result < best_val:
            best_val = val_result
            best_h = h_result.copy()
            
    except Exception:
        pass
    
    # Phase 3: Also try Gaussian mixture DE if time permits
    if elapsed() < TIME_LIMIT - 40:
        n_comp_g = 4
        n_params_g = 3 * n_comp_g
        bounds_g = []
        for _ in range(n_comp_g):
            bounds_g.append((0.0, 2.0))
            bounds_g.append((-0.25, 0.25))
            bounds_g.append((0.01, 0.5))
        
        def objective_gaussian(params):
            if elapsed() > TIME_LIMIT - 10:
                return 1e10
            h = _gaussian_mixture_heights(params, n_comp_g)
            return _compute_score(h)
        
        try:
            result_g = differential_evolution(
                objective_gaussian,
                bounds_g,
                maxiter=100,
                popsize=10,
                tol=1e-6,
                mutation=(0.5, 1.0),
                recombination=0.7,
                seed=123,
                polish=True
            )
            
            h_result_g = _gaussian_mixture_heights(result_g.x, n_comp_g)
            val_result_g = _compute_score(h_result_g)
            
            if val_result_g < best_val:
                best_val = val_result_g
                best_h = h_result_g.copy()
                
        except Exception:
            pass
    
    # Phase 4: Nelder-Mead refinement on the best parametric family
    if elapsed() < TIME_LIMIT - 25:
        if de_result is not None and de_result.success:
            x0 = de_result.x.copy()
            obj_func = objective_cosine
            n_comp = n_components
        else:
            # Random initialization
            rng = np.random.RandomState(42)
            x0 = np.concatenate([
                rng.uniform(0, 2, n_components),
                rng.uniform(-0.2, 0.2, n_components),
                rng.uniform(0.05, 0.3, n_components)
            ])
            obj_func = objective_cosine
            n_comp = n_components
        
        try:
            result2 = minimize(
                obj_func,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            
            h_result2 = _cosine_bump_heights(result2.x, n_comp)
            val_result2 = _compute_score(h_result2)
            
            if val_result2 < best_val:
                best_val = val_result2
                best_h = h_result2.copy()
                
        except Exception:
            pass
    
    # Phase 5: Sensitivity-based local refinement
    if elapsed() < TIME_LIMIT - 15:
        refined_h, refined_val = _sensitivity_refinement(
            best_h, start_time, TIME_LIMIT - 5
        )
        if refined_val < best_val:
            best_val = refined_val
            best_h = refined_h.copy()
    
    # Ensure nonnegative and return
    best_h = np.maximum(best_h, 0.0)
    return list(best_h)
