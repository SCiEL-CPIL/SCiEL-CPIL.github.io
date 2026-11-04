import numpy as np
import time
from scipy.optimize import minimize
from scipy.fft import rfft, irfft

N = 600
W = 1.0 / 1200.0

# Fine grid for accurate autoconvolution evaluation
# The support of the convolution is [-0.5, 0.5].
FINE_N = 12000
X_FINE = np.linspace(-0.5, 0.5, FINE_N, endpoint=False)
DX = 1.0 / FINE_N

# Map fine grid points to the 600 bins of the step function
# Step function is defined on [-0.25, 0.25].
CELL = np.full(FINE_N, -1, dtype=np.int64)
valid_mask = (X_FINE >= -0.25) & (X_FINE < 0.25)
idx = np.floor((X_FINE[valid_mask] + 0.25) / W).astype(np.int64)
idx = np.clip(idx, 0, N - 1)
CELL[valid_mask] = idx

N_FFT = 1 << (2 * FINE_N - 1).bit_length()

# Precompute bin centers
BIN_CENTERS = -0.25 + (np.arange(N) + 0.5) * W

def ratio_from_h(h):
    """
    Compute the normalized maximum of the autoconvolution.
    """
    h = np.asarray(h, dtype=np.float64)
    if h.size != N:
        h = np.resize(h, N)
    h = np.clip(h, 0.0, None)
    s = h.sum()
    if s < 1e-15:
        return float('inf')
    
    # Map step heights to fine grid
    f_fine = np.zeros(FINE_N)
    mask = CELL >= 0
    f_fine[mask] = h[CELL[mask]]
    
    # FFT-based autoconvolution
    F = rfft(f_fine, N_FFT)
    conv = irfft(F * F, N_FFT)[:2 * FINE_N - 1]
    conv *= DX  # scale by dx for continuous integral
    
    max_conv = np.max(conv)
    
    # Normalization: (integral f)^2 = (sum(h) * W)^2
    integral_sq = (s * W) ** 2
    if integral_sq < 1e-15:
        return float('inf')
    
    return max_conv / integral_sq

def smooth_ratio_from_h(h, tau=0.05):
    """
    Smooth approximation of the ratio for gradient-based optimization.
    """
    h = np.asarray(h, dtype=np.float64)
    if h.size != N:
        h = np.resize(h, N)
    h = np.clip(h, 0.0, None)
    s = h.sum()
    if s < 1e-15:
        return float('inf')
    
    f_fine = np.zeros(FINE_N)
    mask = CELL >= 0
    f_fine[mask] = h[CELL[mask]]
    
    F = rfft(f_fine, N_FFT)
    conv = irfft(F * F, N_FFT)[:2 * FINE_N - 1]
    conv *= DX
    
    g = conv / ((s * W) ** 2)
    
    g_max = np.max(g)
    if not np.isfinite(g_max):
        return float('inf')
    
    z = (g - g_max) / tau
    z = np.clip(z, -50, 50)
    val = np.log(np.sum(np.exp(z))) * tau + g_max
    
    return val

def gaussians_to_h(params):
    """
    Convert Gaussian superposition parameters to step function heights.
    """
    K = len(params) // 3
    h = np.zeros(N)
    
    for i in range(K):
        a_raw = params[3 * i]
        x_c = params[3 * i + 1]
        s_raw = params[3 * i + 2]
        
        a = np.log1p(np.exp(np.clip(a_raw, -50, 50)))
        s = np.exp(np.clip(s_raw, -10, 10))
        s = max(s, 1e-4)
        
        h += a * np.exp(-0.5 * ((BIN_CENTERS - x_c) / s) ** 2)
        
    return h

def objective_gauss(params):
    h = gaussians_to_h(params)
    val = ratio_from_h(h)
    if not np.isfinite(val):
        return 1e10
    return val

def construct():
    """
    Construct a nonnegative step function with 600 equal intervals on [-1/4, 1/4]
    to minimize the normalized maximum of its autoconvolution.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin before 120s kill
    
    # Initial best: uniform
    best_h = np.ones(N, dtype=np.float64)
    best_val = ratio_from_h(best_h)
    
    # Strategy 1: Parametric Global Search (Gaussians)
    K = 4
    bounds = [
        (-2.0, 5.0),  # amplitude log
        (-0.25, 0.25), # center
        (-2.0, 1.0),  # width log
    ] * K
    
    np.random.seed(42)
    
    # Run Differential Evolution on Gaussian parameters
    try:
        from scipy.optimize import differential_evolution
        result = differential_evolution(
            objective_gauss,
            bounds,
            maxiter=50,
            popsize=15,
            tol=1e-6,
            seed=42,
            polish=False,
            init='sobol'
        )
        
        if result.fun < best_val and np.isfinite(result.fun):
            h_candidate = gaussians_to_h(result.x)
            val = ratio_from_h(h_candidate)
            if val < best_val and np.isfinite(val):
                best_val = val
                best_h = h_candidate.copy()
    except Exception:
        pass
    
    # Strategy 2: Local Polish on Bin Heights
    def smooth_obj(h_flat):
        h = h_flat
        val = smooth_ratio_from_h(h, tau=0.05)
        if not np.isfinite(val):
            return 1e10
        return val

    try:
        res = minimize(
            smooth_obj,
            best_h,
            method='L-BFGS-B',
            bounds=[(0, None)] * N,
            options={'maxiter': 200, 'ftol': 1e-8}
        )
        
        if res.fun < best_val and np.isfinite(res.fun):
            h_candidate = res.x
            val = ratio_from_h(h_candidate)
            if val < best_val and np.isfinite(val):
                best_val = val
                best_h = h_candidate.copy()
    except Exception:
        pass
    
    # Strategy 3: Multi-start Local Polish
    for restart in range(3):
        if time.time() > deadline - 10.0:
            break
            
        h_start = best_h + np.random.randn(N) * 0.05
        h_start = np.clip(h_start, 0, None)
        
        try:
            res = minimize(
                smooth_obj,
                h_start,
                method='L-BFGS-B',
                bounds=[(0, None)] * N,
                options={'maxiter': 100, 'ftol': 1e-8}
            )
            
            if res.fun < best_val and np.isfinite(res.fun):
                h_candidate = res.x
                val = ratio_from_h(h_candidate)
                if val < best_val and np.isfinite(val):
                    best_val = val
                    best_h = h_candidate.copy()
        except Exception:
            pass

    return list(best_h)
