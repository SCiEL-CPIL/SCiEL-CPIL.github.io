import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

N = 600
DT = (1/4 - (-1/4)) / N  # interval width = 1/600
X_GRID = -1/4 + (np.arange(N) + 0.5) * DT  # midpoints of intervals

def _gaussian_mixture_heights(params, n_components=8):
    """Convert parameter vector to step heights using Gaussian mixture.
    
    params layout: [amp_1, center_1, width_1, amp_2, center_2, width_2, ...]
    """
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

def _compute_score(h):
    """Compute normalized maximum autoconvolution."""
    s = np.sum(h)
    if s < 1e-15:
        return 1e10
    hn = h / s
    F = np.fft.fft(hn, n=2*N)
    conv = np.fft.ifft(F * F).real[:2*N-1]
    return np.max(conv)

def construct():
    start_time = time.time()
    time_limit = 145.0
    
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
    ]
    
    for h_cand in candidates:
        if elapsed() > time_limit:
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
    
    # Phase 2: Differential evolution on Gaussian mixture parameters
    n_components = 8
    n_params = 3 * n_components
    
    # Bounds: amplitude in [0, 2], center in [-0.25, 0.25], width in [0.01, 0.5]
    bounds = []
    for _ in range(n_components):
        bounds.append((0.0, 2.0))   # amplitude
        bounds.append((-0.25, 0.25)) # center
        bounds.append((0.01, 0.5))   # width
    
    def objective(params):
        if elapsed() > time_limit:
            return 1e10
        h = _gaussian_mixture_heights(params, n_components)
        return _compute_score(h)
    
    # Run differential evolution
    try:
        result = differential_evolution(
            objective,
            bounds,
            maxiter=300,
            popsize=20,
            tol=1e-8,
            mutation=(0.5, 1.0),
            recombination=0.7,
            seed=42,
            polish=True
        )
        
        h_result = _gaussian_mixture_heights(result.x, n_components)
        val_result = _compute_score(h_result)
        
        if val_result < best_val:
            best_val = val_result
            best_h = h_result.copy()
            
    except Exception:
        pass
    
    # Phase 3: Fine-tune with local search on the best mixture
    if elapsed() < time_limit - 10:
        # Use Nelder-Mead on the mixture parameters
        if 'result' in locals() and result.success:
            x0 = result.x.copy()
        else:
            # Random initialization
            x0 = np.array([
                np.random.uniform(0, 1) for _ in range(n_components)
            ] + [
                np.random.uniform(-0.2, 0.2) for _ in range(n_components)
            ] + [
                np.random.uniform(0.05, 0.3) for _ in range(n_components)
            ])
        
        try:
            result2 = minimize(
                objective,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            
            h_result2 = _gaussian_mixture_heights(result2.x, n_components)
            val_result2 = _compute_score(h_result2)
            
            if val_result2 < best_val:
                best_val = val_result2
                best_h = h_result2.copy()
                
        except Exception:
            pass
    
    # Phase 4: Direct coordinate descent on heights for final polish
    if elapsed() < time_limit - 5:
        h_polish = best_h.copy()
        # Simple coordinate descent: adjust individual bins
        for _ in range(50):
            if elapsed() > time_limit - 5:
                break
            improved = False
            for i in range(0, N, 2):  # step by 2 for speed
                if elapsed() > time_limit - 5:
                    break
                old_val = best_val
                # Try increasing bin i
                h_test = h_polish.copy()
                h_test[i] += 0.01
                s = np.sum(h_test)
                if s < 1e-15:
                    continue
                h_test = h_test / s
                val_test = _compute_score(h_test)
                if val_test < best_val - 1e-12:
                    best_val = val_test
                    best_h = h_test.copy()
                    h_polish = h_test.copy()
                    improved = True
                # Try decreasing bin i
                h_test = h_polish.copy()
                h_test[i] -= 0.01
                if h_test[i] < 0:
                    h_test[i] = 0
                s = np.sum(h_test)
                if s < 1e-15:
                    continue
                h_test = h_test / s
                val_test = _compute_score(h_test)
                if val_test < best_val - 1e-12:
                    best_val = val_test
                    best_h = h_test.copy()
                    h_polish = h_test.copy()
                    improved = True
            if not improved:
                break
    
    return list(best_h)
