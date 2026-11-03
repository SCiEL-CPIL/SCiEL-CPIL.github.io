import numpy as np
import time
import scipy.optimize as opt

N = 600
W = 1.0 / 1200.0
X = np.linspace(-0.25, 0.25, N, endpoint=False)

# Precompute fine grid for fast convolution ratio
FINE_N = 12000
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

def objective(params):
    h = fourier_to_h(params)
    neg = h[h < 0]
    if len(neg) > 0:
        penalty = 100.0 * np.sum(neg**2)
    else:
        penalty = 0.0
    r = ratio_from_h(h)
    return r + penalty

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_h = np.ones(N)
    best_val = ratio_from_h(best_h)
    best_params = None
    
    def update_best(h, val, params=None):
        nonlocal best_h, best_val, best_params
        if val < best_val:
            best_val = val
            best_h = h.copy()
            best_params = params.copy() if params is not None else None
            
    rng = np.random.default_rng(42)
    
    # Use Fourier series with K=10 harmonics (21 parameters)
    K = 10
    n_params = 1 + 2 * K
    
    # Bounds for Fourier coefficients
    bounds = [(-1.0, 5.0)]  # a0
    for _ in range(K):
        bounds.append((-2.0, 2.0)) # a_k
        bounds.append((-2.0, 2.0)) # b_k
        
    # Try multiple restarts of L-BFGS-B
    n_restarts = 50
    
    for i in range(n_restarts):
        if time.time() > deadline - 10.0:
            break
            
        # Random initial point within bounds
        x0 = np.array([rng.uniform(b[0], b[1]) for b in bounds])
        
        try:
            result = opt.minimize(
                objective,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            
            h_opt = fourier_to_h(result.x)
            val_opt = objective(result.x)
            
            # Re-evaluate with clipping to be safe
            h_clipped = np.clip(h_opt, 0, None)
            true_val = ratio_from_h(h_clipped)
            update_best(h_clipped, true_val, result.x)
            
        except Exception:
            pass
            
    # Final polish with tighter tolerance on the best found params
    if best_params is not None and time.time() < deadline - 5.0:
        try:
            result2 = opt.minimize(
                objective,
                best_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
            )
            h_polished = fourier_to_h(result2.x)
            val_polished = objective(result2.x)
            h_polished_clipped = np.clip(h_polished, 0, None)
            true_polished_val = ratio_from_h(h_polished_clipped)
            update_best(h_polished_clipped, true_polished_val, result2.x)
        except Exception:
            pass
            
    # Additional polish: try to reduce the max by adjusting the function
    # If we have time, do a local spatial optimization
    if time.time() < deadline - 5.0:
        # Simple local search in spatial domain
        current_h = best_h.copy()
        current_val = best_val
        step_size = 0.01
        for _ in range(20):
            if time.time() > deadline - 5.0:
                break
            improved = False
            for idx in range(N):
                if time.time() > deadline - 5.0:
                    break
                original = current_h[idx]
                current_h[idx] = max(0, original + step_size)
                val_plus = ratio_from_h(current_h)
                if val_plus < current_val:
                    current_val = val_plus
                    improved = True
                else:
                    current_h[idx] = max(0, original - step_size)
                    val_minus = ratio_from_h(current_h)
                    if val_minus < current_val:
                        current_val = val_minus
                        improved = True
                    else:
                        current_h[idx] = original
            if not improved:
                step_size *= 0.5
                if step_size < 1e-4:
                    break
            update_best(current_h, current_val)
            
    return list(best_h)
