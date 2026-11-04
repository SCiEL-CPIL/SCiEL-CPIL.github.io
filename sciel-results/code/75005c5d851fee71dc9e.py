import numpy as np
import time
import scipy.optimize as opt

N = 600
W = 1.0 / 1200.0
X = np.linspace(-0.25, 0.25, N, endpoint=False)

# Precompute fine grid for fast convolution ratio
FINE_N = 8000
X_FINE = np.linspace(-0.25, 0.25, FINE_N, endpoint=False)
DX = 0.5 / FINE_N
CELL = np.clip(((X_FINE + 0.25) / W).astype(np.int64), 0, N - 1)
N_FFT = 1 << (2 * FINE_N - 1).bit_length()

# Coarse grid for global search
COARSE_N = 2000
X_COARSE = np.linspace(-0.25, 0.25, COARSE_N, endpoint=False)
DC = 0.5 / COARSE_N
CELL_C = np.clip(((X_COARSE + 0.25) / W).astype(np.int64), 0, N - 1)
N_FFT_C = 1 << (2 * COARSE_N - 1).bit_length()

def ratio_from_h(h):
    """Compute max(autoconv) / ||h||_2^2 using FFT on fine grid."""
    h = np.asarray(h, dtype=np.float64)
    if h.size != N:
        h = np.resize(h, N)
    h = np.clip(h, 0.0, None)
    s = h.sum()
    if s < 1e-15:
        return float('inf')
    f = h[CELL]
    F = np.fft.rfft(f, N_FFT)
    conv = np.fft.irfft(F * F, N_FFT)[:2 * FINE_N - 1] * DX
    l2 = np.sum(f * f) * DX
    if l2 < 1e-15:
        return float('inf')
    return float(np.max(conv) / l2)

def coarse_ratio_from_h(h):
    """Compute max(autoconv) / ||h||_2^2 using FFT on coarse grid."""
    h = np.asarray(h, dtype=np.float64)
    if h.size != N:
        h = np.resize(h, N)
    h = np.clip(h, 0.0, None)
    s = h.sum()
    if s < 1e-15:
        return float('inf')
    f = h[CELL_C]
    F = np.fft.rfft(f, N_FFT_C)
    conv = np.fft.irfft(F * F, N_FFT_C)[:2 * COARSE_N - 1] * DC
    l2 = np.sum(f * f) * DC
    if l2 < 1e-15:
        return float('inf')
    return float(np.max(conv) / l2)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Fourier parameterization
    K = 30 # Number of harmonics
    N_PARAMS = 2 * K + 1
    
    def fourier_to_h(coeffs):
        h = np.full(N, coeffs[0])
        for k in range(1, K + 1):
            a_k = coeffs[2*k - 1]
            b_k = coeffs[2*k]
            h += a_k * np.cos(2 * np.pi * k * np.arange(N) / N)
            h += b_k * np.sin(2 * np.pi * k * np.arange(N) / N)
        return np.clip(h, 0.0, None)
    
    def fourier_ratio_coarse(coeffs):
        h = fourier_to_h(coeffs)
        return coarse_ratio_from_h(h)
    
    def fourier_ratio_fine(coeffs):
        h = fourier_to_h(coeffs)
        return ratio_from_h(h)

    best_h = np.ones(N)
    best_val = ratio_from_h(best_h)
    
    def update_best(h, val):
        nonlocal best_h, best_val
        if val < best_val:
            best_val = val
            best_h = h.copy()

    rng = np.random.default_rng(42)
    
    # Phase 1: Global search with Differential Evolution on coarse grid
    # Bounds: c0 in [0, 2], ak, bk in [-1, 1]
    bounds = [(0, 2)] + [(-1, 1)] * (2 * K)
    
    n_de_runs = 5
    for i in range(n_de_runs):
        if time.time() > deadline - 30.0:
            break
        
        try:
            result = opt.differential_evolution(
                fourier_ratio_coarse,
                bounds,
                maxiter=100,
                popsize=15,
                seed=42 + i,
                tol=0.01,
                init='latinhypercube'
            )
            if result.fun < best_val:
                h_opt = fourier_to_h(result.x)
                v_fine = ratio_from_h(h_opt)
                update_best(h_opt, v_fine)
        except Exception:
            pass

    # Phase 2: Local refinement with L-BFGS-B on fine grid
    # Use numerical gradients
    n_refine_runs = 10
    for i in range(n_refine_runs):
        if time.time() > deadline - 10.0:
            break
        
        # Start from current best or a random perturbation
        if i == 0:
            # Start from best found so far
            # Inverse Fourier to get coeffs? Hard.
            # Just use a random start or the uniform one.
            # Let's try to initialize coeffs from the best_h if possible?
            # For now, random starts.
            coeffs = np.zeros(N_PARAMS)
            coeffs[0] = float(rng.uniform(0.5, 1.5))
            for k in range(1, K + 1):
                coeffs[2*k - 1] = float(rng.uniform(-0.5, 0.5))
                coeffs[2*k] = float(rng.uniform(-0.5, 0.5))
        else:
            coeffs = np.zeros(N_PARAMS)
            coeffs[0] = float(rng.uniform(0.5, 1.5))
            for k in range(1, K + 1):
                coeffs[2*k - 1] = float(rng.uniform(-0.5, 0.5))
                coeffs[2*k] = float(rng.uniform(-0.5, 0.5))
        
        try:
            result = opt.minimize(
                fourier_ratio_fine,
                coeffs,
                method='L-BFGS-B',
                options={'maxiter': 50, 'gtol': 1e-4}
            )
            if result.fun < best_val:
                h_opt = fourier_to_h(result.x)
                update_best(h_opt, result.fun)
        except Exception:
            pass

    # Phase 3: Coordinate descent on the 600 variables (polish)
    if time.time() < deadline - 5.0:
        cur_h = best_h.copy()
        cur_val = best_val
        rng2 = np.random.default_rng(777)
        
        for it in range(2000):
            if time.time() > deadline - 2.0:
                break
            
            idx = int(rng2.integers(0, N))
            # Try a few small changes
            for factor in [0.9, 1.1, 0.8, 1.2]:
                h_try = cur_h.copy()
                h_try[idx] = max(0.0, cur_h[idx] * factor)
                v = ratio_from_h(h_try)
                if v < cur_val:
                    cur_val = v
                    cur_h = h_try
                    update_best(cur_h, cur_val)
                    break # Accept and move to next index

    return list(best_h)
