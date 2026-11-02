import time
import numpy as np
from scipy.optimize import minimize

N = 600

def compute_score(h):
    s = np.sum(h)
    if s == 0:
        return 0.0
    n_fft = 2 * N - 1
    H = np.fft.rfft(h, n=n_fft)
    conv = np.fft.irfft(H * H, n=n_fft)[:2 * N - 1]
    max_conv = np.max(conv)
    return max_conv / (s * s)

def smooth_max_surrogate(conv, beta=100.0):
    max_val = np.max(conv)
    shifted = beta * (conv - max_val)
    shifted = np.clip(shifted, -100, 100)
    log_sum = np.log(np.sum(np.exp(shifted)))
    return (max_val + log_sum / beta)

def objective(params, beta=100.0):
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
    h = np.maximum(params, 0.0)
    s = np.sum(h)
    if s < 1e-10:
        return np.zeros_like(params)
    
    n_fft = 2 * N - 1
    H = np.fft.rfft(h, n=n_fft)
    conv = np.fft.irfft(H * H, n=n_fft)[:2 * N - 1]
    
    max_val = np.max(conv)
    shifted = beta * (conv - max_val)
    shifted = np.clip(shifted, -100, 100)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    weights = exp_shifted / sum_exp
    
    # Compute gradient using FFT
    W = np.fft.rfft(weights, n=n_fft)
    
    # Correlation via FFT
    grad_h = 2.0 * np.real(np.fft.irfft(W * np.conj(H), n=n_fft))[:N]
    
    smooth_max_val = smooth_max_surrogate(conv, beta)
    grad_obj = grad_h / (s * s) - 2 * smooth_max_val / (s * s * s)
    
    return grad_obj

def coordinate_descent(h, max_iter=500, time_limit=20.0, start_time=None):
    """
    Coordinate descent with geometric line search for each coordinate.
    """
    if start_time is None:
        start_time = time.time()
    
    h = np.maximum(h, 0.0).copy()
    s = np.sum(h)
    if s < 1e-10:
        return h
    
    current_score = compute_score(h)
    n_fft = 2 * N - 1
    H = np.fft.rfft(h, n=n_fft)
    conv = np.fft.irfft(H * H, n=n_fft)[:2 * N - 1]
    
    # Geometric multipliers for line search
    multipliers = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]
    
    for iteration in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Shuffle coordinate order for better convergence
        indices = np.random.permutation(N)
        
        for i in indices:
            if time.time() - start_time > time_limit:
                break
            
            if h[i] < 1e-12:
                continue
            
            old_val = h[i]
            best_new_val = old_val
            best_local_score = current_score
            
            for mult in multipliers:
                new_val = old_val * mult
                h[i] = new_val
                
                # Recompute only the affected parts
                s_new = s + (new_val - old_val)
                if s_new < 1e-10:
                    h[i] = old_val
                    continue
                
                H[i] = np.fft.rfft(np.zeros(N), n=n_fft)[i]  # placeholder
                # Actually need to recompute H properly
                H_new = H.copy()
                # Update FFT efficiently is complex, just recompute for simplicity in this inner loop
                # But that's expensive. Let's use a different approach.
                # For now, just compute score directly
                score_new = compute_score(h)
                
                if score_new < best_local_score:
                    best_local_score = score_new
                    best_new_val = new_val
                    s = s_new
            
            h[i] = best_new_val
            current_score = best_local_score
    
    return h

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

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

    beta = 100.0
    
    for h0 in initializations:
        if time.time() > deadline - 40:
            break
            
        # Optimize with L-BFGS-B
        try:
            res = minimize(
                objective, 
                h0, 
                method='L-BFGS-B', 
                jac=objective_grad, 
                args=(beta,),
                bounds=[(0, None)] * N,
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-6}
            )
            
            h_opt = res.x
            h_opt = np.maximum(h_opt, 0)
            
            score_opt = compute_score(h_opt)
            
            if score_opt < best_score:
                best_score = score_opt
                best_h = h_opt.copy()
                
        except Exception:
            pass

    # Coordinate descent refinement
    if time.time() < deadline - 10:
        try:
            h_cd = coordinate_descent(best_h, max_iter=100, time_limit=30.0, start_time=start_time)
            h_cd = np.maximum(h_cd, 0)
            score_cd = compute_score(h_cd)
            
            if score_cd < best_score:
                best_score = score_cd
                best_h = h_cd.copy()
        except Exception:
            pass

    # Final polish with higher beta and tighter tolerance
    if time.time() < deadline - 5:
        try:
            beta_high = 300.0
            res = minimize(
                objective, 
                best_h, 
                method='L-BFGS-B', 
                jac=objective_grad, 
                args=(beta_high,),
                bounds=[(0, None)] * N,
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            h_opt = res.x
            h_opt = np.maximum(h_opt, 0)
            score_opt = compute_score(h_opt)
            if score_opt < best_score:
                best_score = score_opt
                best_h = h_opt.copy()
        except Exception:
            pass

    # Zero-out pass: set very small values to 0
    best_h[best_h < 1e-10] = 0
    
    best_h = np.maximum(best_h, 0)
    return best_h.tolist()
# EVOLVE-BLOCK-END
