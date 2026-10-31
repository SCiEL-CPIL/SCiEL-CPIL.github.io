import numpy as np
from scipy import optimize
from scipy.fft import fft, ifft
import time

N = 600

def _fft_convolve(h):
    """FFT-based autoconvolution for speed."""
    n = len(h)
    size = 2 * n
    H = fft(h, size)
    conv = np.real(ifft(H * H, size)[:2*n-1])
    return conv

def compute_autoconv_max(h):
    """Compute the normalized maximum of the autoconvolution of step function h."""
    h = np.asarray(h, dtype=float)
    conv = _fft_convolve(h)
    total = sum(h)
    if total < 1e-10:
        return 0.0
    return max(conv) / (total * total)

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Leave 5s margin for 150s limit
    
    # Initial guess: all ones
    h = np.ones(N)
    best_score = compute_autoconv_max(h)
    best_h = h.copy()
    
    # Soft-max objective
    def soft_objective(x, T):
        h = np.maximum(x, 0)
        conv = _fft_convolve(h)
        total = sum(h)
        if total < 1e-10:
            return 1e10
        conv_max = np.max(conv)
        shifted = (conv - conv_max) / T
        shifted = np.clip(shifted, -100, 100)
        soft_max = conv_max + T * np.log(np.sum(np.exp(shifted)))
        return soft_max / (total * total)
    
    np.random.seed(42)
    
    # List of initial guesses
    initial_guesses = [
        np.ones(N),
        1.0 + 0.1 * np.cos(2 * np.pi * np.linspace(0, 1, N, endpoint=False)),
        1.0 + 0.1 * np.sin(2 * np.pi * np.linspace(0, 1, N, endpoint=False)),
        1.0 + 0.05 * np.cos(4 * np.pi * np.linspace(0, 1, N, endpoint=False)),
        np.random.rand(N) + 0.1,
        1.0 + 0.2 * np.abs(np.linspace(-1, 1, N)),
        1.0 + 0.2 * np.exp(-5 * np.linspace(-1, 1, N)**2),
    ]
    
    for i, h0 in enumerate(initial_guesses):
        if time.time() > deadline - 10:
            break
        h0 = np.maximum(h0, 0.01)
        h0 = h0 / np.sum(h0) * np.sum(np.ones(N)) # Normalize to same scale
        
        # Stage 1: Coarse optimization
        T1 = 0.1
        try:
            res1 = optimize.minimize(
                lambda x: soft_objective(x, T1), h0,
                method='L-BFGS-B',
                bounds=[(0, None)] * N,
                options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            h1 = np.maximum(res1.x, 0)
            score1 = compute_autoconv_max(h1)
            if score1 < best_score:
                best_score = score1
                best_h = h1.copy()
                
            # Stage 2: Fine optimization
            if time.time() > deadline - 5:
                break
            T2 = 0.01
            res2 = optimize.minimize(
                lambda x: soft_objective(x, T2), h1,
                method='L-BFGS-B',
                bounds=[(0, None)] * N,
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            h2 = np.maximum(res2.x, 0)
            score2 = compute_autoconv_max(h2)
            if score2 < best_score:
                best_score = score2
                best_h = h2.copy()
                
        except Exception as e:
            pass
    
    # Final polish with very low temperature
    if time.time() < deadline - 2:
        T_final = 0.001
        try:
            res_final = optimize.minimize(
                lambda x: soft_objective(x, T_final), best_h,
                method='L-BFGS-B',
                bounds=[(0, None)] * N,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            h_final = np.maximum(res_final.x, 0)
            score_final = compute_autoconv_max(h_final)
            if score_final < best_score:
                best_score = score_final
                best_h = h_final.copy()
        except Exception:
            pass
            
    return list(best_h)
