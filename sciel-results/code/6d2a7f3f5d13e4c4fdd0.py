import time
import numpy as np
from scipy.optimize import minimize

N = 600

def compute_score(h):
    """
    Compute the normalized maximum of the autoconvolution.
    h: array of nonnegative values.
    Returns max_k (h*h)[k] / (sum h)^2
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
    Returns: (1/beta) * log(sum(exp(beta * conv)))
    """
    max_val = np.max(conv)
    shifted = beta * (conv - max_val)
    # Clip to avoid overflow in exp
    shifted = np.clip(shifted, -100, 100)
    log_sum = np.log(np.sum(np.exp(shifted)))
    return (max_val + log_sum / beta)

def objective_and_grad(params, beta=100.0):
    """
    Objective and gradient of the smooth-max surrogate objective.
    f(h) = smooth_max(conv(h)) / (sum h)^2
    where conv(h) = h * h (linear auto-convolution)
    
    Returns (f, grad_f)
    """
    h = np.maximum(params, 0.0)
    s = np.sum(h)
    if s < 1e-10:
        return 1e10, np.zeros_like(params)
    
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
    
    # smooth_max value
    smooth_max_val = max_val + np.log(sum_exp) / beta
    
    # Gradient of smooth_max w.r.t. h:
    # d(smooth_max)/d(h_i) = sum_k w_k * d(conv_k)/d(h_i)
    # conv_k = sum_j h_j * h_{k-j} (linear conv, zero-padded)
    # d(conv_k)/d(h_i) = h_{k-i} + h_{i-k} (with h_m = 0 for m outside [0,N-1])
    # This equals 2 * (w cross-correlate h)_i
    # Cross-correlation: (w ⋆ h)_i = sum_k w_k * h_{k-i}
    # In FFT: IFFT(conj(FFT(w)) * FFT(h)) gives cross-correlation
    W = np.fft.fft(weights, n=n_fft)
    # IFFT(conj(W) * H) = cross-correlation of w and h
    cross_corr = np.real(np.fft.ifft(np.conj(W) * H))[:N]
    grad_smooth = 2.0 * cross_corr
    
    # f = smooth_max / s^2
    # df/dh_i = grad_smooth_i / s^2 - 2 * smooth_max / s^3
    f = smooth_max_val / (s * s)
    grad_f = grad_smooth / (s * s) - 2.0 * smooth_max_val / (s * s * s)
    
    # Zero out gradient where h is at boundary (h=0) to respect constraint
    grad_f[h <= 1e-12] = 0.0
    
    return f, grad_f

def objective(params, beta=100.0):
    f, _ = objective_and_grad(params, beta)
    return f

def objective_grad(params, beta=100.0):
    _, g = objective_and_grad(params, beta)
    return g

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    best_h = np.ones(N) / N
    best_score = compute_score(best_h)

    np.random.seed(42)
    
    # Multiple initializations
    initializations = []
    
    # 1. Uniform
    initializations.append(np.ones(N) / N)
    
    # 2. Gaussian
    x = np.linspace(-1, 1, N)
    gauss = np.exp(-x**2 / (2 * 0.5**2))
    gauss /= np.sum(gauss)
    initializations.append(gauss)
    
    # 3. Linear ramp
    lin = np.linspace(0.1, 1.0, N)
    lin /= np.sum(lin)
    initializations.append(lin)
    
    # 4. Sine
    sine = np.sin(np.pi * (x + 1) / 2) ** 2
    sine /= np.sum(sine)
    initializations.append(sine)
    
    # 5. Step function (left half vs right half)
    step = np.zeros(N)
    step[:N//2] = 1.0
    step /= np.sum(step)
    initializations.append(step)
    
    # 6. Random
    rand_h = np.random.rand(N)
    rand_h /= np.sum(rand_h)
    initializations.append(rand_h)
    
    # 7. Triangular
    tri = np.abs(np.linspace(-1, 1, N))
    tri = 1.0 - tri
    tri /= np.sum(tri)
    initializations.append(tri)
    
    # 8. Two-block: high-low-high pattern
    two_block = np.zeros(N)
    two_block[:N//3] = 1.0
    two_block[2*N//3:] = 1.0
    two_block /= np.sum(two_block)
    initializations.append(two_block)
    
    # 9. Cosine bump
    cos_bump = np.cos(np.pi * x) ** 2
    cos_bump = np.maximum(cos_bump, 0)
    cos_bump /= np.sum(cos_bump)
    initializations.append(cos_bump)
    
    # 10. Sparse random: only ~30% nonzero
    sparse_rand = np.zeros(N)
    mask = np.random.rand(N) < 0.3
    sparse_rand[mask] = np.random.rand(np.sum(mask))
    sparse_rand /= np.sum(sparse_rand)
    initializations.append(sparse_rand)
    
    # 11. Alternating blocks
    alt = np.zeros(N)
    block_size = N // 6
    for b in range(0, N, 2 * block_size):
        alt[b:b+block_size] = 1.0
    alt /= np.sum(alt)
    initializations.append(alt)
    
    # 12. Concentrated center
    center = np.zeros(N)
    center[N//4:3*N//4] = 1.0
    center /= np.sum(center)
    initializations.append(center)

    bounds = [(0, None)] * N
    
    # Phase 1: Multi-start optimization with moderate beta
    for h0 in initializations:
        if time.time() > deadline - 40:
            break
            
        try:
            # Start with lower beta for smoother landscape
            res = minimize(
                objective, 
                h0, 
                method='L-BFGS-B', 
                jac=objective_grad, 
                args=(30.0,),
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-7}
            )
            
            h_opt = np.maximum(res.x, 0)
            
            # Refine with higher beta
            if time.time() < deadline - 35:
                res2 = minimize(
                    objective, 
                    h_opt, 
                    method='L-BFGS-B', 
                    jac=objective_grad, 
                    args=(80.0,),
                    bounds=bounds,
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-7}
                )
                h_opt = np.maximum(res2.x, 0)
            
            score_opt = compute_score(h_opt)
            
            if score_opt < best_score:
                best_score = score_opt
                best_h = h_opt.copy()
                
        except Exception:
            pass

    # Phase 2: More random restarts from perturbed best solution
    if time.time() < deadline - 25:
        for restart in range(5):
            if time.time() > deadline - 25:
                break
            try:
                noise = np.random.randn(N) * 0.1
                h0 = best_h + noise
                h0 = np.maximum(h0, 0)
                h0 /= np.sum(h0)
                
                res = minimize(
                    objective, 
                    h0, 
                    method='L-BFGS-B', 
                    jac=objective_grad, 
                    args=(50.0,),
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-7}
                )
                
                h_opt = np.maximum(res.x, 0)
                score_opt = compute_score(h_opt)
                
                if score_opt < best_score:
                    best_score = score_opt
                    best_h = h_opt.copy()
                    
            except Exception:
                pass

    # Phase 3: Final high-precision polish with very high beta
    if time.time() < deadline - 10:
        try:
            res = minimize(
                objective, 
                best_h, 
                method='L-BFGS-B', 
                jac=objective_grad, 
                args=(200.0,),
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            h_opt = np.maximum(res.x, 0)
            score_opt = compute_score(h_opt)
            if score_opt < best_score:
                best_score = score_opt
                best_h = h_opt.copy()
        except Exception:
            pass
    
    # Phase 4: Very high beta final polish
    if time.time() < deadline - 3:
        try:
            res = minimize(
                objective, 
                best_h, 
                method='L-BFGS-B', 
                jac=objective_grad, 
                args=(500.0,),
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            h_opt = np.maximum(res.x, 0)
            score_opt = compute_score(h_opt)
            if score_opt < best_score:
                best_score = score_opt
                best_h = h_opt.copy()
        except Exception:
            pass
    
    best_h = np.maximum(best_h, 0)
    return best_h.tolist()
# EVOLVE-BLOCK-END
