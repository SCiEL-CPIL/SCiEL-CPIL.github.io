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
    return h

def objective_fourier(params):
    h = fourier_to_h(params)
    neg = h[h < 0]
    if len(neg) > 0:
        penalty = 50.0 * np.sum(neg**2)
    else:
        penalty = 0.0
    r = ratio_from_h(h)
    return r + penalty

def local_refine(h, max_iters=20, tol=1e-5):
    """Refine step heights locally using coordinate descent."""
    h = np.clip(h.copy(), 0.0, None)
    best_val = ratio_from_h(h)
    
    for it in range(max_iters):
        improved = False
        for i in range(N):
            old_val = h[i]
            # Try small perturbations
            for delta in [-0.1, -0.05, 0.05, 0.1, 0.2, -0.2]:
                h[i] = max(0.0, old_val + delta)
                val = ratio_from_h(h)
                if val < best_val - tol:
                    best_val = val
                    improved = True
                    break
                h[i] = old_val
        if not improved:
            break
    return h, best_val

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill
    
    best_h = np.ones(N)
    best_val = ratio_from_h(best_h)
    
    def update_best(h, val):
        nonlocal best_h, best_val
        if val < best_val:
            best_val = val
            best_h = h.copy()
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Global search with Fourier series
    K = 15
    n_params = 1 + 2 * K
    
    bounds = [(0.0, 3.0)]  # a0: keep nonnegative
    for _ in range(K):
        bounds.append((-1.5, 1.5))  # a_k
        bounds.append((-1.5, 1.5))  # b_k
    
    # Try multiple optimization methods and restarts
    n_restarts = 30
    
    for i in range(n_restarts):
        if time.time() > deadline - 30.0:
            break
        
        # Random initial point
        x0 = np.array([rng.uniform(b[0], b[1]) for b in bounds])
        
        try:
            # Use L-BFGS-B for better convergence
            result = opt.minimize(
                objective_fourier,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-6, 'gtol': 1e-5}
            )
            
            h_opt = fourier_to_h(result.x)
            h_clipped = np.clip(h_opt, 0, None)
            true_val = ratio_from_h(h_clipped)
            update_best(h_clipped, true_val)
            
        except Exception:
            pass
    
    # Phase 2: Local refinement on best solution
    if time.time() < deadline - 10.0:
        h_refined, val_refined = local_refine(best_h, max_iters=30, tol=1e-6)
        update_best(h_refined, val_refined)
    
    # Phase 3: Try a few more Fourier restarts with different K if time permits
    if time.time() < deadline - 20.0:
        K2 = 20
        n_params2 = 1 + 2 * K2
        bounds2 = [(0.0, 3.0)]
        for _ in range(K2):
            bounds2.append((-1.0, 1.0))
            bounds2.append((-1.0, 1.0))
        
        for i in range(10):
            if time.time() > deadline - 5.0:
                break
            x0 = np.array([rng.uniform(b[0], b[1]) for b in bounds2])
            try:
                result = opt.minimize(
                    objective_fourier,
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds2,
                    options={'maxiter': 100, 'ftol': 1e-6}
                )
                h_opt = fourier_to_h(result.x)
                h_clipped = np.clip(h_opt, 0, None)
                true_val = ratio_from_h(h_clipped)
                update_best(h_clipped, true_val)
            except Exception:
                pass
    
    return list(best_h)
