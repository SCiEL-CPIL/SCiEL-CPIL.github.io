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

def fourier_to_h(coeffs):
    """
    Convert Fourier coefficients to step heights.
    coeffs = [a0, a1, b1, a2, b2, ...]
    h(x) = a0 + sum_{k=1}^{K} [a_k cos(2*pi*k*x) + b_k sin(2*pi*k*x)]
    """
    K = (len(coeffs) - 1) // 2
    h = np.full(N, coeffs[0])
    for k in range(1, K + 1):
        ak = coeffs[1 + 2*(k-1)]
        bk = coeffs[2 + 2*(k-1)]
        arg = 2.0 * np.pi * k * X
        h += ak * np.cos(arg) + bk * np.sin(arg)
    return np.clip(h, 0.0, None)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    best_h = np.ones(N)
    best_val = ratio_from_h(best_h)
    
    def update_best(h, val):
        nonlocal best_h, best_val
        if val < best_val:
            best_val = val
            best_h = h.copy()
            
    rng = np.random.default_rng(42)
    
    # === Phase 1: Fourier-domain optimization ===
    # Use a fixed number of Fourier modes (e.g., 20 modes = 41 params)
    # This is a smooth, low-dimensional parameterization that can represent
    # many useful shapes (Gaussian-like, cosine, etc.)
    
    n_modes = 20
    n_params = 1 + 2 * n_modes  # a0, a1, b1, ..., a20, b20
    
    # Bounds: a0 in [0, 5], ak/bk in [-5, 5]
    bounds = [(0.0, 5.0)]  # a0
    for _ in range(n_modes):
        bounds.append((-5.0, 5.0))  # ak
        bounds.append((-5.0, 5.0))  # bk
    
    def fourier_objective(params):
        h = fourier_to_h(params)
        return ratio_from_h(h)
    
    # Use differential evolution for global search in Fourier space
    try:
        result = opt.differential_evolution(
            fourier_objective,
            bounds,
            seed=42,
            maxiter=150,
            popsize=20,
            tol=1e-5,
            update_interval=1,
            polish=True
        )
        if result.fun < best_val:
            h_opt = fourier_to_h(result.x)
            update_best(h_opt, result.fun)
    except Exception:
        pass
    
    if time.time() > deadline - 30.0:
        return list(best_h)
    
    # === Phase 2: Local refinement in spatial domain ===
    # From the best Fourier solution, do local spatial optimization
    # using coordinate descent on random subsets
    
    for restart in range(3):
        if time.time() > deadline - 5.0:
            break
        
        h_cur = best_h.copy()
        val_cur = best_val
        
        # Add small noise to escape local minima
        noise = rng.normal(0, 0.05, N)
        h_cur = np.clip(h_cur + noise, 0, None)
        val_cur = ratio_from_h(h_cur)
        
        # Coordinate descent: iterate over random coordinates
        for it in range(300):
            if time.time() > deadline - 2.0:
                break
            
            # Pick a random coordinate
            idx = rng.integers(0, N)
            
            # Try to optimize this coordinate via a small 1D search
            # Sample a few values around the current value
            current_val = h_cur[idx]
            # Try values in a range around current
            candidates = [
                current_val * 0.5,
                current_val * 0.8,
                current_val * 1.2,
                current_val * 1.5,
                max(0.0, current_val - 0.1),
                max(0.0, current_val + 0.1),
                0.0,  # Try zeroing out
                1.0,  # Try setting to 1
            ]
            
            best_candidate = h_cur[idx]
            best_candidate_val = val_cur
            
            for cand in candidates:
                h_try = h_cur.copy()
                h_try[idx] = cand
                v_try = ratio_from_h(h_try)
                if v_try < best_candidate_val:
                    best_candidate_val = v_try
                    best_candidate = cand
            
            if best_candidate != h_cur[idx]:
                h_cur[idx] = best_candidate
                val_cur = best_candidate_val
        
        if val_cur < best_val:
            update_best(h_cur, val_cur)
    
    # === Phase 3: Try specific analytical shapes ===
    if time.time() < deadline - 2.0:
        x_norm = np.linspace(-1.0, 1.0, N)
        
        # Gaussian with different widths
        for width in [3.0, 5.0, 7.0, 10.0]:
            h_g = np.exp(-width * x_norm**2)
            update_best(h_g, ratio_from_h(h_g))
        
        # Cosine
        h_c = 0.5 + 0.5 * np.cos(np.pi * x_norm)
        update_best(h_c, ratio_from_h(h_c))
        
        # Raised Cosine (Blackman-ish)
        h_b = 0.42 + 0.5 * np.cos(np.pi * x_norm) + 0.08 * np.cos(2 * np.pi * x_norm)
        h_b = np.clip(h_b, 0, None)
        update_best(h_b, ratio_from_h(h_b))
        
        # Triangular
        h_t = 1.0 - np.abs(x_norm)
        update_best(h_t, ratio_from_h(h_t))
        
        # Flat top (constant)
        h_f = np.ones(N)
        update_best(h_f, ratio_from_h(h_f))
    
    return list(best_h)
