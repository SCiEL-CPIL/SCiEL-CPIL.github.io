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

def fourier_to_h(params):
    """
    Convert Fourier coefficients to step heights.
    params = [a0, a1, b1, a2, b2, ..., aK, bK]
    h(x) = a0 + sum_{k=1}^K [a_k cos(2*pi*k*x) + b_k sin(2*pi*k*x)]
    """
    K = (len(params) - 1) // 2
    h = np.full(N, params[0])
    for k in range(1, K + 1):
        ak = params[1 + 2*(k-1)]
        bk = params[2 + 2*(k-1)]
        arg = 2.0 * np.pi * k * X
        h += ak * np.cos(arg) + bk * np.sin(arg)
    return np.clip(h, 0.0, None)

def fourier_objective(params):
    h = fourier_to_h(params)
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
    
    # === Phase 1: Global Optimization with Differential Evolution on Fourier Coefficients ===
    # We try different numbers of Fourier modes: 2, 3, 4, 5
    # For K modes, we have 1 + 2*K parameters
    
    for K in [2, 3, 4, 5]:
        n_params = 1 + 2 * K
        # Base term can be large to keep h nonnegative
        bounds = [(0.5, 5.0)]  # a0
        for _ in range(K):
            bounds.append((-2.0, 2.0))  # a_k
            bounds.append((-2.0, 2.0))  # b_k
            
        try:
            # Differential Evolution
            result = opt.differential_evolution(
                fourier_objective,
                bounds,
                seed=42,
                maxiter=80,
                popsize=12,
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
            break

    # === Phase 2: Local Refinement with Nelder-Mead on Fourier Coefficients ===
    # Start from the best solution found in Phase 1
    if time.time() < deadline - 15.0:
        # We need to recover approximate Fourier coefficients from best_h
        # Or just use Nelder-Mead directly on the best_h perturbed in Fourier space?
        # Let's just run Nelder-Mead on the Fourier parameters that gave best_h
        # But we didn't store them. Let's do a few random restarts of NM from random Fourier inits.
        
        for restart in range(10):
            if time.time() > deadline - 5.0:
                break
            
            # Random initialization in Fourier space
            K = 3
            n_params = 1 + 2 * K
            x0 = np.zeros(n_params)
            x0[0] = 1.0  # Base
            for i in range(1, n_params):
                x0[i] = rng.uniform(-0.5, 0.5)
                
            try:
                result = opt.minimize(
                    fourier_objective,
                    x0,
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8}
                )
                if result.fun < best_val:
                    h_opt = fourier_to_h(result.x)
                    update_best(h_opt, result.fun)
            except Exception:
                pass

    # === Phase 3: Try Specific Analytical Shapes ===
    if time.time() < deadline - 2.0:
        x_norm = np.linspace(-1.0, 1.0, N)
        
        # Gaussian
        h1 = np.exp(-5.0 * x_norm**2)
        update_best(h1, ratio_from_h(h1))
        
        # Cosine
        h2 = 0.5 + 0.5 * np.cos(np.pi * x_norm)
        update_best(h2, ratio_from_h(h2))
        
        # Raised Cosine (Blackman-ish)
        h3 = 0.42 + 0.5 * np.cos(np.pi * x_norm) + 0.08 * np.cos(2 * np.pi * x_norm)
        h3 = np.clip(h3, 0, None)
        update_best(h3, ratio_from_h(h3))
        
        # Triangular
        h4 = 1.0 - np.abs(x_norm)
        update_best(h4, ratio_from_h(h4))
        
        # Sinc-like
        h5 = np.sinc(x_norm / 2.0)
        h5 = np.clip(h5, 0, None)
        update_best(h5, ratio_from_h(h5))

    return list(best_h)
