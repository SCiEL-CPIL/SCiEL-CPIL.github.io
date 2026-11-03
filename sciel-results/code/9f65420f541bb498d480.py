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
            
    # Precompute cosine basis for symmetric Fourier representation
    # h(x) = sum_{k=0}^K a_k cos(2 pi k x / 0.5)
    # x is in [-0.25, 0.25]
    # cos(2 pi k x / 0.5) = cos(4 pi k x)
    # We will use K up to 30 (31 coefficients)
    
    def fourier_sym_to_h(coeffs):
        """Construct h from symmetric Fourier coefficients."""
        h = np.zeros(N)
        for k in range(len(coeffs)):
            h += coeffs[k] * np.cos(4 * np.pi * k * X)
        return np.clip(h, 0.0, None)
    
    def soft_ratio_from_coeffs(coeffs, T=0.02):
        """Compute smoothed ratio for gradient-based optimization."""
        h = fourier_sym_to_h(coeffs)
        if h.sum() < 1e-15:
            return 1e10
        f = h[CELL]
        F = np.fft.rfft(f, N_FFT)
        conv = np.fft.irfft(F * F, N_FFT)[:2 * FINE_N - 1] * DX
        l2 = np.sum(f * f) * DX
        if l2 < 1e-15:
            return 1e10
        # SoftMax approximation of max
        # conv values are around 1
        # Shift to avoid overflow
        max_c = np.max(conv)
        exp_conv = np.exp((conv - max_c) / T)
        soft_max = max_c + T * np.log(np.sum(exp_conv))
        return float(soft_max / l2)
    
    rng = np.random.default_rng(42)
    
    # Try different numbers of harmonics
    K_options = [10, 20, 30, 50]
    
    for K in K_options:
        if time.time() > deadline - 20.0:
            break
            
        n_starts = 5
        for start in range(n_starts):
            if time.time() > deadline - 5.0:
                break
            
            # Initialize coefficients
            # DC term should be positive
            coeffs = np.zeros(K + 1)
            coeffs[0] = float(rng.uniform(0.5, 2.0))
            for k in range(1, K + 1):
                # Small random perturbations
                coeffs[k] = float(rng.uniform(-0.2, 0.2))
            
            try:
                result = opt.minimize(
                    soft_ratio_from_coeffs,
                    coeffs,
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-6, 'gtol': 1e-5}
                )
                
                # Evaluate the true (non-soft) ratio for the optimized coefficients
                h_opt = fourier_sym_to_h(result.x)
                val = ratio_from_h(h_opt)
                
                if val < best_val:
                    update_best(h_opt, val)
                    
            except Exception:
                pass
                
    # Fallback: Simple coordinate descent if time remains
    if time.time() < deadline - 5.0:
        rng2 = np.random.default_rng(777)
        cur_h = best_h.copy()
        cur_val = best_val
        
        for it in range(2000):
            if time.time() > deadline - 2.0:
                break
            
            idxs = rng2.choice(N, size=5, replace=False)
            h_try = cur_h.copy()
            for idx in idxs:
                factor = float(rng2.uniform(0.9, 1.1))
                h_try[idx] = max(0.0, cur_h[idx] * factor)
            
            v = ratio_from_h(h_try)
            if v < cur_val:
                cur_val = v
                cur_h = h_try
                update_best(cur_h, cur_val)
                
    return list(best_h)
