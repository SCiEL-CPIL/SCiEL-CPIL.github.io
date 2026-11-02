import time
import numpy as np
from scipy.optimize import minimize

N = 600

def compute_score(h):
    """
    Compute the actual score: max_conv / (sum h)^2
    Lower is better.
    """
    s = np.sum(h)
    if s == 0:
        return 0.0
    # Use FFT for efficient convolution
    n_fft = 2 * N - 1
    H = np.fft.fft(h, n=n_fft)
    conv = np.real(np.fft.ifft(H * H))[:2*N-1]
    max_conv = np.max(conv)
    return max_conv / (s * s)

def smooth_max_surrogate(conv, beta=100.0):
    """
    Smooth approximation of max(conv) using log-sum-exp.
    """
    max_val = np.max(conv)
    shifted = beta * (conv - max_val)
    shifted = np.clip(shifted, -100, 100)
    log_sum = np.log(np.sum(np.exp(shifted)))
    return (max_val + log_sum / beta)

def objective(params, beta=100.0):
    """
    Objective function for optimization.
    params: flattened array of h values.
    """
    h = np.maximum(params, 0.0)
    s = np.sum(h)
    if s < 1e-10:
        return 1e10
    n_fft = 2 * N - 1
    H = np.fft.fft(h, n=n_fft)
    conv = np.real(np.fft.ifft(H * H))[:2*N-1]
    smoothed_peak = smooth_max_surrogate(conv, beta)
    return smoothed_peak / (s * s)

def objective_grad(params, beta=100.0):
    """
    Gradient of the objective function using FFT for efficiency.
    """
    h = np.maximum(params, 0.0)
    s = np.sum(h)
    if s < 1e-10:
        return np.zeros_like(params)
    
    n_fft = 2 * N - 1
    H = np.fft.fft(h, n=n_fft)
    conv = np.real(np.fft.ifft(H * H))[:2*N-1]
    
    # Compute softmax weights for the max surrogate
    max_val = np.max(conv)
    shifted = beta * (conv - max_val)
    shifted = np.clip(shifted, -100, 100)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    weights = exp_shifted / sum_exp
    
    # Gradient of smooth_max w.r.t h
    # smooth_max = max_val + (1/beta) log(sum exp(beta(conv - max_val)))
    # d(smooth_max)/d(h_i) = sum_k weights[k] * d(conv[k])/d(h_i)
    # conv[k] = sum_j h[j] h[k-j]
    # d(conv[k])/d(h_i) = h[k-i] + h[i-k] (with appropriate bounds, effectively correlation)
    # Using FFT: grad_h = 2 * correlate(weights, h)
    W = np.fft.fft(weights, n=n_fft)
    # Correlation via FFT: ifft(conj(W) * H) gives correlation
    grad_h_smooth = 2.0 * np.real(np.fft.ifft(np.conj(W) * H))[:N]
    
    # Gradient of objective: f = smooth_max / s^2
    smooth_max_val = smooth_max_surrogate(conv, beta)
    # d(f)/d(h_i) = (grad_h_smooth[i] * s^2 - smooth_max_val * 2 * s) / s^4
    #             = grad_h_smooth[i] / s^2 - 2 * smooth_max_val / s^3
    grad_obj = grad_h_smooth / (s * s) - 2 * smooth_max_val / (s * s * s)
    
    # Zero out gradient where h is at boundary (h=0) to respect constraint
    # Note: L-BFGS-B handles bounds, but zeroing gradient at boundary helps convergence
    grad_obj[h <= 1e-12] = 0.0
    
    return grad_obj

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    best_h = np.ones(N) / N
    best_score = compute_score(best_h)
    
    np.random.seed(42)
    
    # Define diverse initializations to explore different regions of the space
    initializations = []
    
    # 1. Uniform (Baseline)
    initializations.append(np.ones(N) / N)
    
    # 2. Two-block structure (Known to be better)
    # High on ends, low in middle? Or high in middle?
    # Let's try high on ends (U-shape)
    x = np.linspace(-1, 1, N)
    u_shape = 1.0 + x**2
    u_shape /= np.sum(u_shape)
    initializations.append(u_shape)
    
    # High in middle (Gaussian)
    gauss = np.exp(-x**2 / (2 * 0.3**2))
    gauss /= np.sum(gauss)
    initializations.append(gauss)
    
    # 3. Two distinct blocks (Left and Right)
    two_block = np.zeros(N)
    two_block[:N//3] = 1.0
    two_block[2*N//3:] = 1.0
    two_block /= np.sum(two_block)
    initializations.append(two_block)
    
    # 4. Random perturbation of uniform
    rand_h = np.ones(N) + 0.1 * np.random.randn(N)
    rand_h = np.maximum(rand_h, 0)
    rand_h /= np.sum(rand_h)
    initializations.append(rand_h)
    
    # 5. Cosine taper
    cos_taper = 0.5 * (1 + np.cos(np.pi * x))
    cos_taper = np.maximum(cos_taper, 1e-10)
    cos_taper /= np.sum(cos_taper)
    initializations.append(cos_taper)
    
    bounds = [(0, None)] * N
    
    # Optimize from each initialization
    for h0 in initializations:
        if time.time() > deadline - 20:
            break
        
        try:
            # Phase 1: Coarse optimization with lower beta
            res = minimize(
                objective, 
                h0, 
                method='L-BFGS-B', 
                jac=objective_grad, 
                args=(50.0,),
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-6}
            )
            
            h_opt = np.maximum(res.x, 0)
            
            # Phase 2: Fine optimization with higher beta if time permits
            if time.time() < deadline - 10:
                res2 = minimize(
                    objective, 
                    h_opt, 
                    method='L-BFGS-B', 
                    jac=objective_grad, 
                    args=(100.0,),
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                h_opt = np.maximum(res2.x, 0)
            
            score_opt = compute_score(h_opt)
            
            if score_opt < best_score:
                best_score = score_opt
                best_h = h_opt.copy()
                
        except Exception:
            pass

    # Final global check and polish
    if time.time() < deadline - 5:
        try:
            res_final = minimize(
                objective, 
                best_h, 
                method='L-BFGS-B', 
                jac=objective_grad, 
                args=(200.0,),
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            h_final = np.maximum(res_final.x, 0)
            score_final = compute_score(h_final)
            
            if score_final < best_score:
                best_score = score_final
                best_h = h_final.copy()
        except Exception:
            pass

    return best_h.tolist()
