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
    if s < 1e-10:
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
    W = np.fft.rfft(weights, n=n_fft)
    grad_max = 2.0 * np.real(np.fft.irfft(W * np.conj(H), n=n_fft))[:N]
    
    # Gradient of objective: f = smooth_max / s^2
    smooth_max_val = smooth_max_surrogate(conv, beta)
    grad_obj = grad_max / (s * s) - 2 * smooth_max_val / (s * s * s)
    
    # Zero out gradient where h is at boundary (h=0) to respect constraint
    grad_obj[h <= 1e-12] = 0.0
    
    return grad_obj

def coordinate_descent(h, time_limit=10.0):
    """
    Coordinate descent with geometric line search including zero candidates.
    This is often more effective for non-smooth max-like objectives than gradient descent.
    """
    start = time.time()
    h = h.copy()
    current_score = compute_score(h)
    
    # Geometrically spaced multipliers for line search
    # Include 0 to allow sparsity
    multipliers = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 4.0]
    
    # We will iterate over indices. To avoid bias, we can shuffle or just go 0..N.
    # Given N=600, a full pass is 600 score evaluations.
    # Each score eval is ~ O(N log N). 600 * 600 * log(600) is manageable.
    
    no_improve_count = 0
    
    while time.time() - start < time_limit:
        improved = False
        
        for i in range(N):
            if time.time() - start > time_limit:
                break
            
            old_val = h[i]
            best_local_val = old_val
            best_local_score = current_score
            
            # Try different multipliers
            for m in multipliers:
                trial_val = old_val * m
                # Optimization: if trial_val is very close to old_val, skip
                if abs(trial_val - old_val) < 1e-12:
                    continue
                
                h[i] = trial_val
                # We don't normalize here because the score function handles normalization
                # But wait, the problem is scale invariant?
                # score(h) = max(h*h) / (sum h)^2.
                # score(c*h) = max(c^2 h*h) / (c sum h)^2 = score(h).
                # So we can work with unnormalized h.
                
                trial_score = compute_score(h)
                
                if trial_score < best_local_score - 1e-12:
                    best_local_score = trial_score
                    best_local_val = trial_val
            
            # Apply the best found value
            if best_local_val != old_val:
                h[i] = best_local_val
                current_score = best_local_score
                improved = True
                no_improve_count = 0
            else:
                no_improve_count += 1
        
        # If we went through a full pass without improvement, we might be stuck
        # in a local minimum where single-coordinate moves don't help.
        # We can break if no improvement for several passes, or just continue until time limit.
        # Given the time limit, continuing is safer to catch slow improvements.
        # However, if no_improve_count > N (one full pass), we might want to break early 
        # to save time for other initializations or polishes.
        if no_improve_count > N:
            break

    return h, current_score

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
    
    # Strategy:
    # 1. Brief L-BFGS to get into basin (low beta, smooth)
    # 2. Coordinate Descent for heavy refinement (exact max, discrete steps)
    # 3. Brief L-BFGS polish (high beta)
    
    for h0 in initializations:
        if time.time() > deadline - 20:
            break
            
        h_current = h0.copy()
        
        # Phase 1: Quick Gradient Descent to get close
        if time.time() < deadline - 10:
            try:
                res = minimize(
                    objective, 
                    h_current, 
                    method='L-BFGS-B', 
                    jac=objective_grad, 
                    args=(50.0,),
                    bounds=bounds,
                    options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-4}
                )
                h_current = res.x
                h_current = np.maximum(h_current, 0)
                score_current = compute_score(h_current)
                if score_current < best_score:
                    best_score = score_current
                    best_h = h_current.copy()
            except Exception:
                pass

        # Phase 2: Coordinate Descent Refinement
        # Allocate remaining time for this phase, but reserve some for final polish
        time_remaining = deadline - time.time() - 5.0
        if time_remaining > 1.0:
            h_cd, score_cd = coordinate_descent(h_current, time_limit=time_remaining)
            if score_cd < best_score:
                best_score = score_cd
                best_h = h_cd.copy()
            else:
                # Even if CD didn't improve global best, it might have improved this local one
                # But we only care about global best for the final return.
                pass
                
        # Phase 3: Final Polish with High Beta Gradient
        if time.time() < deadline - 2:
            try:
                res = minimize(
                    objective, 
                    best_h, 
                    method='L-BFGS-B', 
                    jac=objective_grad, 
                    args=(300.0,),
                    bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-6}
                )
                h_opt = res.x
                h_opt = np.maximum(h_opt, 0)
                score_opt = compute_score(h_opt)
                
                if score_opt < best_score:
                    best_score = score_opt
                    best_h = h_opt.copy()
            except Exception:
                pass

    # Final Sparse Refinement: Try zeroing out small values on the global best
    if time.time() < deadline - 1:
        h_try = best_h.copy()
        indices = np.argsort(h_try)
        
        # Try zeroing out the smallest 10%
        num_zero = N // 10
        for i in range(num_zero):
            h_try[indices[i]] = 0.0
        
        if np.sum(h_try) > 0:
            score_try = compute_score(h_try)
            if score_try < best_score:
                best_score = score_try
                best_h = h_try.copy()
                
                # Try zeroing out another 10% (total 20%)
                h_try2 = best_h.copy()
                indices = np.argsort(h_try2)
                num_zero2 = N // 5
                for i in range(num_zero2):
                    h_try2[indices[i]] = 0.0
                if np.sum(h_try2) > 0:
                    score_try2 = compute_score(h_try2)
                    if score_try2 < best_score:
                        best_score = score_try2
                        best_h = h_try2.copy()

    # Final cleanup
    best_h = np.maximum(best_h, 0)
    return best_h.tolist()
