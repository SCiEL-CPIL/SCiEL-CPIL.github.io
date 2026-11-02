import numpy as np
from scipy.optimize import minimize
import time

N = 600
START_TIME = time.time()
TIME_LIMIT = 140  # safety margin before 150s

# Use fewer fine samples per step for speed
K = 16
M_STEP = N * K  # 9600
M_FFT = 1 << 14  # 16384 (power of 2, >= M_STEP)

def compute_score(heights):
    """Compute the normalized maximum of autoconvolution."""
    heights = np.asarray(heights, dtype=np.float64)
    if np.any(heights < 0):
        return np.inf
    
    f_fine = np.zeros(M_FFT, dtype=np.float64)
    f_fine[:M_STEP] = np.repeat(heights, K)
    
    F = np.fft.rfft(f_fine)
    conv = np.fft.irfft(F * F, n=M_FFT)
    
    max_conv = np.max(conv)
    
    dx = 0.5 / M_STEP
    norm2 = np.sum(f_fine**2) * dx
    
    if norm2 < 1e-12:
        return np.inf
    
    return max_conv / norm2

def _objective_and_grad(heights, beta=500.0):
    """Compute soft-max objective and gradient w.r.t. step heights."""
    heights = np.asarray(heights, dtype=np.float64)
    
    f_fine = np.zeros(M_FFT, dtype=np.float64)
    f_fine[:M_STEP] = np.repeat(heights, K)
    
    F = np.fft.rfft(f_fine)
    conv = np.fft.irfft(F * F, n=M_FFT)
    
    c_max = np.max(conv)
    if c_max < 1e-15:
        c_max = 1e-15
    
    # Soft max: (1/beta) * log(sum(exp(beta*(conv - c_max))))
    scaled = beta * (conv - c_max)
    # Numerical stability: subtract max (already c_max, so max of scaled is 0)
    exp_terms = np.exp(scaled)
    Z = np.sum(exp_terms)
    soft_max = c_max + (1.0 / beta) * np.log(Z)
    
    dx = 0.5 / M_STEP
    norm2 = np.sum(f_fine**2) * dx
    
    if norm2 < 1e-12:
        return 1e10, np.zeros(N)
    
    obj = soft_max / norm2
    
    # Gradient
    # d(soft_max)/d(conv) = exp_terms / Z
    W = exp_terms / Z
    
    # d(conv)/d(f_fine): conv = ifft(fft(f)^2)
    # Perturbation: delta_conv = ifft(2*F*fft(delta_f))
    # So grad_f_fine = ifft(2*F*fft(W))
    W_fft = np.fft.rfft(W)
    grad_f_fine = np.fft.irfft(2 * F * W_fft, n=M_FFT)
    
    # d(norm2)/d(f_fine) = 2 * f_fine * dx
    # d(obj)/d(f_fine) = (grad_f_fine * norm2 - soft_max * 2 * f_fine * dx) / norm2^2
    grad_f = (grad_f_fine * norm2 - soft_max * 2.0 * f_fine * dx) / (norm2 ** 2)
    
    # Map to step heights: each height contributes to K fine samples
    grad_h = np.sum(grad_f[:M_STEP].reshape(N, K), axis=1)
    
    return obj, grad_h

def construct():
    global START_TIME
    
    def time_left():
        return TIME_LIMIT - (time.time() - START_TIME)
    
    # Initial guesses
    x = np.linspace(0, 1, N)
    
    # Guess 1: raised cosine
    h0 = 0.5 * (1 + np.cos(2 * np.pi * x))
    h0 = np.maximum(h0, 0.0)
    
    # Guess 2: uniform
    h1 = np.ones(N)
    
    # Guess 3: Gaussian-like
    xg = np.linspace(-1, 1, N)
    h2 = np.exp(-3.0 * xg**2)
    h2 = h2 / np.max(h2)
    
    # Guess 4: triangular
    h3 = np.maximum(1.0 - 2.0 * np.abs(x - 0.5), 0.0)
    
    # Guess 5: flat top
    h4 = np.ones(N)
    h4[:N//4] = 0.5
    h4[3*N//4:] = 0.5
    
    best_h = h0.copy()
    best_score = compute_score(h0)
    
    initial_guesses = [h0, h1, h2, h3, h4]
    
    # Try each initial guess with L-BFGS-B
    for idx, h_init in enumerate(initial_guesses):
        if time_left() < 20:
            break
        
        try:
            def obj_and_grad(h):
                h_clipped = np.maximum(h, 0.0)
                obj, grad = _objective_and_grad(h_clipped, beta=500.0)
                grad = np.where(h_clipped < 1e-10, 0.0, grad)
                return obj, grad
            
            res = minimize(
                obj_and_grad,
                h_init.copy(),
                method='L-BFGS-B',
                jac=True,
                bounds=[(0, None)] * N,
                options={'maxiter': 150, 'maxfun': 800, 'maxls': 50, 
                         'ftol': 1e-9, 'gtol': 1e-6, 'maxcor': 20}
            )
            
            if hasattr(res, 'x') and res.x is not None:
                h_opt = np.maximum(res.x, 0.0)
                score_opt = compute_score(h_opt)
                if score_opt < best_score:
                    best_h = h_opt
                    best_score = score_opt
                    
        except Exception:
            pass
    
    # Polish with higher beta
    if time_left() > 15:
        try:
            def obj_and_grad_polish(h):
                h_clipped = np.maximum(h, 0.0)
                obj, grad = _objective_and_grad(h_clipped, beta=2000.0)
                grad = np.where(h_clipped < 1e-10, 0.0, grad)
                return obj, grad
            
            res_polish = minimize(
                obj_and_grad_polish,
                best_h.copy(),
                method='L-BFGS-B',
                jac=True,
                bounds=[(0, None)] * N,
                options={'maxiter': 100, 'maxfun': 500, 'maxls': 50,
                         'ftol': 1e-10, 'gtol': 1e-7, 'maxcor': 30}
            )
            
            if hasattr(res_polish, 'x') and res_polish.x is not None:
                h_polish = np.maximum(res_polish.x, 0.0)
                score_polish = compute_score(h_polish)
                if score_polish < best_score:
                    best_h = h_polish
                    best_score = score_polish
                    
        except Exception:
            pass
    
    # Final exact polish
    if time_left() > 10:
        try:
            def obj_and_grad_exact(h):
                h_clipped = np.maximum(h, 0.0)
                obj, grad = _objective_and_grad(h_clipped, beta=10000.0)
                grad = np.where(h_clipped < 1e-10, 0.0, grad)
                return obj, grad
            
            res_exact = minimize(
                obj_and_grad_exact,
                best_h.copy(),
                method='L-BFGS-B',
                jac=True,
                bounds=[(0, None)] * N,
                options={'maxiter': 80, 'maxfun': 300, 'maxls': 50,
                         'ftol': 1e-11, 'gtol': 1e-8, 'maxcor': 40}
            )
            
            if hasattr(res_exact, 'x') and res_exact.x is not None:
                h_exact = np.maximum(res_exact.x, 0.0)
                score_exact = compute_score(h_exact)
                if score_exact < best_score:
                    best_h = h_exact
                    best_score = score_exact
                    
        except Exception:
            pass
    
    best_h = np.maximum(best_h, 0.0)
    return [max(0.0, float(v)) for v in best_h]
