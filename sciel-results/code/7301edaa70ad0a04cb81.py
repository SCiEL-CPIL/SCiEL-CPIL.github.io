import numpy as np
import time
import scipy.optimize as opt

N = 600
W = 1.0 / 1200.0
X = np.linspace(-0.25, 0.25, N, endpoint=False)

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

def softplus(x):
    return np.where(x > 30, x, np.log1p(np.exp(x)))

def fourier_softplus_to_h(params):
    """
    h(x) = softplus( sum_{k=0}^K a_k cos(2*pi*k*x) )
    """
    K = len(params) - 1
    h = np.full(N, params[0])
    for k in range(1, K + 1):
        ak = params[k]
        arg = 2.0 * np.pi * k * X
        h += ak * np.cos(arg)
    return softplus(h)

def objective(params):
    h = fourier_softplus_to_h(params)
    return ratio_from_h(h)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_h = np.ones(N)
    best_val = ratio_from_h(best_h)
    
    def update_best(h, val):
        nonlocal best_h, best_val
        if val < best_val:
            best_val = val
            best_h = h.copy()
            
    rng = np.random.default_rng(42)
    
    # Use Fourier series with K=30 harmonics (31 parameters)
    # Symmetric (cosines only)
    K = 30
    n_params = K + 1
    
    # Bounds
    # a0 positive to ensure function is not zero
    bounds = [(0.0, 5.0)]
    for _ in range(K):
        bounds.append((-2.0, 2.0))
        
    n_restarts = 100
    
    for i in range(n_restarts):
        if time.time() > deadline - 10.0:
            break
            
        # Random initial point
        x0 = np.zeros(n_params)
        x0[0] = rng.uniform(0.5, 2.0)
        for k in range(1, n_params):
            x0[k] = rng.uniform(-1.0, 1.0)
            
        try:
            result = opt.minimize(
                objective,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            h_opt = fourier_softplus_to_h(result.x)
            val_opt = objective(result.x)
            
            update_best(h_opt, val_opt)
            
        except Exception:
            pass
            
    return list(best_h)
