import numpy as np
import time
import scipy.optimize as opt

N = 600
W = 1.0 / 1200.0
X = np.linspace(-0.25, 0.25, N, endpoint=False)

# Precompute for fast convolution ratio
FINE_N = 20000
X_FINE = np.linspace(-0.25, 0.25, FINE_N, endpoint=False)
DX = 0.5 / FINE_N
CELL = np.clip(((X_FINE + 0.25) / W).astype(np.int64), 0, N - 1)
N_FFT = 1 << (2 * FINE_N - 1).bit_length()

def ratio_from_h(h):
    """Compute normalized max autoconvolution ratio using FFT."""
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

def fourier_to_h(c0, a, b):
    """
    Reconstruct h from Fourier coefficients.
    c0: DC offset
    a: cosine coefficients
    b: sine coefficients
    Returns nonnegative h.
    """
    h = np.full(N, c0, dtype=np.float64)
    x = np.linspace(-0.25, 0.25, N, endpoint=False)
    # Map x to [0, 1] for standard Fourier basis? 
    # The domain is [-1/4, 1/4].
    # Let's use basis functions defined on this domain.
    # k-th harmonic: cos(2*pi*k*x / (0.5)), sin(2*pi*k*x / (0.5))
    # Period T = 0.5.
    # x in [-0.25, 0.25].
    # cos(2*pi*k*x/0.5) = cos(4*pi*k*x)
    # sin(2*pi*k*x/0.5) = sin(4*pi*k*x)
    
    # To be efficient, we can precompute the basis matrix or compute on the fly.
    # Since N is small (600) and K is small (20), on-the-fly is fine.
    # But we call this many times in optimization.
    # Precomputing basis is better.
    
    # Let's assume global precomputed basis
    for i in range(len(a)):
        k = i + 1
        h += a[i] * np.cos(4 * np.pi * k * x)
    for i in range(len(b)):
        k = i + 1
        h += b[i] * np.sin(4 * np.pi * k * x)
    
    return np.clip(h, 0.0, None)

# Precompute Fourier basis for speed
X_BASIS = np.linspace(-0.25, 0.25, N, endpoint=False)
K_MAX = 20
COS_BASIS = np.zeros((N, K_MAX))
SIN_BASIS = np.zeros((N, K_MAX))
for k in range(1, K_MAX + 1):
    COS_BASIS[:, k-1] = np.cos(4 * np.pi * k * X_BASIS)
    SIN_BASIS[:, k-1] = np.sin(4 * np.pi * k * X_BASIS)

def fourier_to_h_fast(c0, a, b):
    h = c0 + a @ COS_BASIS.T + b @ SIN_BASIS.T
    return np.clip(h, 0.0, None)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Initial best: uniform
    best_h = np.ones(N)
    best_val = ratio_from_h(best_h)
    
    # Phase 1: Fourier Domain Optimization
    # Parameters: c0, a[0..K-1], b[0..K-1]
    # Total params: 1 + 2*K_MAX
    
    def objective(params):
        c0 = params[0]
        a = params[1:1+K_MAX]
        b = params[1+K_MAX:1+2*K_MAX]
        h = fourier_to_h_fast(c0, a, b)
        if h.sum() < 1e-12:
            return 1e6
        return ratio_from_h(h)
    
    # Try a few random initializations in Fourier space
    rng = np.random.default_rng(42)
    
    # Start with a Gaussian-like profile in Fourier space?
    # Or just random small perturbations around 0.
    # A constant function has c0=1, a=0, b=0.
    # Let's start near constant but allow shape changes.
    
    best_fourier_h = best_h.copy()
    best_fourier_val = best_val
    
    n_starts = 10
    for i in range(n_starts):
        if time.time() > deadline - 30.0:
            break
            
        # Random initialization
        # c0 around 0.5-1.0
        c0_init = rng.uniform(0.2, 1.0)
        # a, b small random values, decaying with k
        a_init = np.zeros(K_MAX)
        b_init = np.zeros(K_MAX)
        for k in range(K_MAX):
            scale = 1.0 / (k + 1)
            a_init[k] = rng.normal(0, 0.1 * scale)
            b_init[k] = rng.normal(0, 0.1 * scale)
            
        x0 = np.concatenate([[c0_init], a_init, b_init])
        
        # Nelder-Mead optimization
        try:
            res = opt.minimize(objective, x0, method='Nelder-Mead', 
                               options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-6})
            if res.success or res.fun < best_fourier_val:
                h_opt = fourier_to_h_fast(res.x[0], res.x[1:1+K_MAX], res.x[1+K_MAX:1+2*K_MAX])
                v_opt = ratio_from_h(h_opt)
                if v_opt < best_fourier_val:
                    best_fourier_val = v_opt
                    best_fourier_h = h_opt
        except Exception:
            pass

    if best_fourier_val < best_val:
        best_val = best_fourier_val
        best_h = best_fourier_h

    # Phase 2: Local Refinement in Spatial Domain
    # Use the best Fourier result as starting point for spatial local search.
    if time.time() < deadline - 10.0:
        rng2 = np.random.default_rng(12345)
        current_h = best_h.copy()
        current_val = best_val
        
        for it in range(2000):
            if time.time() > deadline - 5.0:
                break
            
            # Random block perturbation
            mode = int(rng2.integers(0, 3))
            if mode == 0:
                # Additive bump
                c = int(rng2.integers(0, N))
                w = int(rng2.integers(5, 30))
                lo = max(0, c - w // 2)
                hi = min(N, lo + w)
                amp = float(rng2.normal(0.0, 0.1))
                h_try = current_h.copy()
                # Smooth bump
                xs = np.linspace(-1, 1, hi - lo)
                bump = np.exp(-xs**2)
                h_try[lo:hi] += amp * bump
                h_try = np.clip(h_try, 0.0, None)
            elif mode == 1:
                # Multiplicative scale
                c = int(rng2.integers(0, N))
                w = int(rng2.integers(5, 30))
                lo = max(0, c - w // 2)
                hi = min(N, lo + w)
                f = float(rng2.uniform(0.8, 1.2))
                h_try = current_h.copy()
                h_try[lo:hi] *= f
                h_try = np.clip(h_try, 0.0, None)
            else:
                # Smoothing / Blurring
                h_try = current_h.copy()
                # Simple box blur on a random segment
                c = int(rng2.integers(0, N))
                w = int(rng2.integers(3, 15))
                lo = max(0, c - w // 2)
                hi = min(N, lo + w)
                if hi - lo > 2:
                    h_try[lo:hi] = np.convolve(h_try[lo:hi], np.ones(3)/3, mode='same')
            
            v = ratio_from_h(h_try)
            if v < current_val:
                current_val = v
                current_h = h_try
                if current_val < best_val:
                    best_val = current_val
                    best_h = current_h.copy()
            else:
                # If it didn't improve, maybe try a smaller step? 
                # For simplicity, just reject.
                pass

    return list(best_h)
