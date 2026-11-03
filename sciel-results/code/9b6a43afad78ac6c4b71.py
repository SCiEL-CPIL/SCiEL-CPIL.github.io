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
    n_fft = 2 * N - 1
    H = np.fft.rfft(h, n=n_fft)
    conv = np.fft.irfft(H * H, n=n_fft)[:2 * N - 1]
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
    """
    h = np.maximum(params, 0.0)
    s = np.sum(h)
    if s < 1e-10:
        return 1e10
    n_fft = 2 * N - 1
    H = np.fft.rfft(h, n=n_fft)
    conv = np.fft.irfft(H * H, n=n_fft)[:2 * N - 1]
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
    H = np.fft.rfft(h, n=n_fft)
    conv = np.fft.irfft(H * H, n=n_fft)[:2 * N - 1]
    
    # Compute softmax weights for the max surrogate
    max_val = np.max(conv)
    shifted = beta * (conv - max_val)
    shifted = np.clip(shifted, -100, 100)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    weights = exp_shifted / sum_exp
    
    # Gradient of smooth max w.r.t h
    # grad_max = 2 * IFFT(FFT(weights) * conj(H))[:N]
    W = np.fft.rfft(weights, n=n_fft)
    grad_max = 2.0 * np.real(np.fft.irfft(W * np.conj(H), n=n_fft))[:N]
    
    # Gradient of objective: f = smooth_max / s^2
    smooth_max_val = smooth_max_surrogate(conv, beta)
    grad_obj = grad_max / (s * s) - 2 * smooth_max_val / (s * s * s)
    
    # Zero out gradient where h is at boundary (h=0) to respect constraint
    grad_obj[h <= 1e-12] = 0.0
    
    return grad_obj

def log_objective(x, beta=100.0):
    """
    Objective in log-domain: h = exp(x).
    """
    # Clip x to avoid overflow/underflow
    x_clipped = np.clip(x, -30, 30)
    h = np.exp(x_clipped)
    s = np.sum(h)
    if s < 1e-10:
        return 1e10
    n_fft = 2 * N - 1
    H = np.fft.rfft(h, n=n_fft)
    conv = np.fft.irfft(H * H, n=n_fft)[:2 * N - 1]
    smoothed_peak = smooth_max_surrogate(conv, beta)
    return smoothed_peak / (s * s)

def log_objective_grad(x, beta=100.0):
    """
    Gradient of log-domain objective.
    h = exp(x_clipped)
    dh/dx = h (where not clipped)
    d_obj/dx = d_obj/dh * dh/dx
    """
    x_clipped = np.clip(x, -30, 30)
    h = np.exp(x_clipped)
    s = np.sum(h)
    if s < 1e-10:
        return np.zeros_like(x)
    
    n_fft = 2 * N - 1
    H = np.fft.rfft(h, n=n_fft)
    conv = np.fft.irfft(H * H, n=n_fft)[:2 * N - 1]
    
    max_val = np.max(conv)
    shifted = beta * (conv - max_val)
    shifted = np.clip(shifted, -100, 100)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    weights = exp_shifted / sum_exp
    
    W = np.fft.rfft(weights, n=n_fft)
    grad_max = 2.0 * np.real(np.fft.irfft(W * np.conj(H), n=n_fft))[:N]
    
    smooth_max_val = smooth_max_surrogate(conv, beta)
    grad_h = grad_max / (s * s) - 2 * smooth_max_val / (s * s * s)
    
    # d_obj/dx = grad_h * h
    grad_x = grad_h * h
    
    # Zero out where clipped
    grad_x[x < -30] = 0.0
    grad_x[x > 30] = 0.0
    
    return grad_x

def coordinate_descent(h, max_rounds=10, time_limit=None):
    """
    Coordinate descent with geometric line search including zero candidates.
    """
    start = time.time()
    h = h.copy()
    
    # Geometrically spaced candidate multipliers
    multipliers = [0.0]  # Always include zero
    for i in range(1, 12):
        multipliers.append(2.0 ** (-i))  # 0.5, 0.25, 0.125, ...
        multipliers.append(2.0 ** i)     # 2, 4, 8, ...
    multipliers = np.array(multipliers)
    
    for round_num in range(max_rounds):
        if time_limit is not None and time.time() - start > time_limit:
            break
            
        improved = False
        for i in range(N):
            if time_limit is not None and time.time() - start > time_limit:
                    break
                    
            old_val = h[i]
            best_new_val = old_val
            best_score = compute_score(h)
            
            # Try geometric multipliers
            for mult in multipliers:
                new_val = old_val * mult
                if new_val < 0:
                    continue
                h[i] = new_val
                s = np.sum(h)
                if s < 1e-10:
                    h[i] = old_val
                    continue
                score = compute_score(h)
                if score < best_score - 1e-12:
                    best_score = score
                    best_new_val = new_val
                    improved = True
            
            h[i] = best_new_val
        
        if not improved:
            break
    
    return h

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

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
    
    bounds = [(0, None)] * N
    
    # Adaptive Beta Schedule
    betas = [30.0, 100.0, 300.0]
    
    for h0 in initializations:
        if time.time() > deadline - 40:
            break
            
        h_current = h0.copy()
        
        for beta in betas:
            if time.time() > deadline - 20:
                break
                
            try:
                res = minimize(
                    objective, 
                    h_current, 
                    method='L-BFGS-B', 
                    jac=objective_grad, 
                    args=(beta,),
                    bounds=bounds,
                    options={'maxiter': 150, 'ftol': 1e-10, 'gtol': 1e-6}
                )
                
                h_opt = res.x
                h_opt = np.maximum(h_opt, 0)
                
                score_opt = compute_score(h_opt)
                
                if score_opt < best_score:
                    best_score = score_opt
                    best_h = h_opt.copy()
                    
                h_current = h_opt
                
            except Exception:
                pass
        
        # Log-domain optimization for this initialization
        if time.time() > deadline - 20:
            break
        
        # Convert h_current to log-domain
        h_pos = np.maximum(h_current, 1e-10)
        x0 = np.log(h_pos)
        
        for beta in betas:
            if time.time() > deadline - 20:
                break
                
            try:
                res = minimize(
                    log_objective, 
                    x0, 
                    method='L-BFGS-B', 
                    jac=log_objective_grad, 
                    args=(beta,),
                    options={'maxiter': 150, 'ftol': 1e-10, 'gtol': 1e-6}
                )
                
                x_opt = res.x
                h_opt = np.exp(np.clip(x_opt, -30, 30))
                h_opt = np.maximum(h_opt, 0)
                
                score_opt = compute_score(h_opt)
                
                if score_opt < best_score:
                    best_score = score_opt
                    best_h = h_opt.copy()
                    
                x0 = x_opt
                
            except Exception:
                pass

    # Coordinate descent refinement on best candidate
    if time.time() < deadline - 15:
        time_for_cd = min(10.0, deadline - time.time() - 5)
        h_cd = coordinate_descent(best_h, max_rounds=5, time_limit=time_for_cd)
        score_cd = compute_score(h_cd)
        if score_cd < best_score:
            best_score = score_cd
            best_h = h_cd.copy()

    # Final high-precision polish on the best candidate
    if time.time() < deadline - 8:
        try:
            res = minimize(
                objective, 
                best_h, 
                method='L-BFGS-B', 
                jac=objective_grad, 
                args=(500.0,),
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            h_opt = res.x
            h_opt = np.maximum(h_opt, 0)
            score_opt = compute_score(h_opt)
            
            if score_opt < best_score:
                best_score = score_opt
                best_h = h_opt.copy()
        except Exception:
            pass
        
        # Log-domain final polish
        if time.time() < deadline - 5:
            try:
                h_pos = np.maximum(best_h, 1e-10)
                x0 = np.log(h_pos)
                res = minimize(
                    log_objective, 
                    x0, 
                    method='L-BFGS-B', 
                    jac=log_objective_grad, 
                    args=(500.0,),
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                x_opt = res.x
                h_opt = np.exp(np.clip(x_opt, -30, 30))
                h_opt = np.maximum(h_opt, 0)
                score_opt = compute_score(h_opt)
                
                if score_opt < best_score:
                    best_score = score_opt
                    best_h = h_opt.copy()
            except Exception:
                pass

    # Second coordinate descent pass after polish
    if time.time() < deadline - 5:
        time_for_cd2 = min(5.0, deadline - time.time() - 3)
        h_cd2 = coordinate_descent(best_h, max_rounds=3, time_limit=time_for_cd2)
        score_cd2 = compute_score(h_cd2)
        if score_cd2 < best_score:
            best_score = score_cd2
            best_h = h_cd2.copy()

    # Sparse refinement: try zeroing out small values
    if time.time() < deadline - 3:
        h_try = best_h.copy()
        indices = np.argsort(h_try)
        
        num_zero = N // 10
        for i in range(num_zero):
            h_try[indices[i]] = 0.0
        
        if np.sum(h_try) > 0:
            score_try = compute_score(h_try)
            if score_try < best_score:
                best_score = score_try
                best_h = h_try.copy()
                
                num_zero = N // 5
                h_try2 = best_h.copy()
                indices = np.argsort(h_try2)
                for i in range(num_zero):
                    h_try2[indices[i]] = 0.0
                if np.sum(h_try2) > 0:
                    score_try2 = compute_score(h_try2)
                    if score_try2 < best_score:
                        best_score = score_try2
                        best_h = h_try2.copy()

    # Final cleanup
    best_h = np.maximum(best_h, 0)
    return best_h.tolist()
