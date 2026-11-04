import numpy as np
from scipy.optimize import minimize
import time

N = 50

def _compute_ratio(h):
    """Compute ||f||_2^2 / (||f||_1 * ||f*f||_inf) for nonnegative h."""
    h = np.asarray(h, dtype=np.float64)
    l2_sq = float(np.sum(h * h))
    l1 = float(np.sum(h))
    if l1 < 1e-15:
        return 0.0
    n_fft = 2 * N
    F = np.fft.rfft(h, n=n_fft)
    conv = np.fft.irfft(F * F.conj(), n=n_fft)
    conv_inf = float(np.max(np.abs(conv)))
    if conv_inf < 1e-15:
        return 0.0
    return l2_sq / (l1 * conv_inf)

def _neg_ratio(params):
    h = np.exp(params)
    return -_compute_ratio(h)

def _compute_ratio_normalized_g(g):
    """
    Compute ratio for h = |g|^2, normalized such that sum(h) = 1.
    g is a complex vector of length N.
    """
    h = np.abs(g) ** 2
    l1 = float(np.sum(h))
    if l1 < 1e-15:
        return 0.0
    h_norm = h / l1
    
    l2_sq = float(np.sum(h_norm * h_norm))
    
    n_fft = 2 * N
    H = np.fft.rfft(h_norm, n=n_fft)
    conv = np.fft.irfft(H * H.conj(), n=n_fft)
    conv_inf = float(np.max(np.abs(conv)))
    
    if conv_inf < 1e-15:
        return 0.0
        
    return l2_sq / (1.0 * conv_inf)

def _neg_ratio_complex(params):
    """
    Objective function for L-BFGS-B.
    params: real vector of length 2*N.
    First N are real parts, next N are imaginary parts.
    """
    g = params[:N] + 1j * params[N:]
    return -_compute_ratio_normalized_g(g)

def construct():
    deadline = time.time() + 140.0
    best_h = np.ones(N)
    best_score = _compute_ratio(best_h)
    
    # Initializations for g
    g0_1 = np.ones(N, dtype=complex)
    g0_2 = np.zeros(N, dtype=complex)
    g0_2[N // 2] = 1.0 + 0j
    g0_3 = np.random.randn(N) + 1j * np.random.randn(N)
    
    candidates = [g0_1, g0_2, g0_3]
    
    for g in candidates:
        h = np.abs(g) ** 2
        score = _compute_ratio(h)
        if score > best_score:
            best_score = score
            best_h = h
    
    n_restarts = 10
    for r in range(n_restarts):
        if time.time() > deadline:
            break
        
        if r < len(candidates):
            g_start = candidates[r]
        else:
            # Perturb best found
            g_start = np.sqrt(best_h) + 0j
            g_start += (np.random.randn(N) + 1j * np.random.randn(N)) * 0.1
            norm = np.linalg.norm(g_start)
            if norm > 0:
                g_start = g_start / norm
            else:
                g_start = candidates[0]

        params0 = np.concatenate([g_start.real, g_start.imag])
        
        try:
            result = minimize(
                _neg_ratio_complex,
                params0,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            g_opt = result.x[:N] + 1j * result.x[N:]
            h_opt = np.abs(g_opt) ** 2
            score = _compute_ratio(h_opt)
            if score > best_score:
                best_score = score
                best_h = h_opt
        except Exception:
            pass

    return [float(x) for x in best_h]
