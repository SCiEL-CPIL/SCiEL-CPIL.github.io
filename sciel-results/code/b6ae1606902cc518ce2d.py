import numpy as np
import time
import scipy.optimize as opt

N = 600
TIME_LIMIT = 140.0  # Safety margin before 150s kill

def compute_score(f):
    """Compute the normalized maximum of the autoconvolution."""
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-15):
        return np.inf
    f = np.clip(f, 0, None)
    norm = np.linalg.norm(f)
    if norm < 1e-15:
        return np.inf
    f = f / norm
    # Compute autoconvolution
    F = np.fft.fft(f)
    C = np.fft.ifft(F * F)
    c = np.real(C)
    return np.max(c)

def parametric_objective(coeffs):
    """Objective function for parametric optimization using cosine basis."""
    # Reconstruct f from coefficients
    # We use a cosine basis: f_k = sum_j c_j * cos(2*pi*j*k/N)
    # To ensure non-negativity, we use a softplus or just clip and normalize.
    # However, clipping during optimization can be non-differentiable.
    # Instead, we can optimize on the log-amplitude or just allow negative values
    # and project, but L-BFGS needs smoothness.
    # A better approach: optimize coefficients directly, construct f, clip, normalize.
    # The objective is smooth enough if we handle the normalization carefully.
    
    k = np.arange(N)
    # Basis functions: cos(2*pi*j*k/N) for j=0..M
    M = len(coeffs)
    # Construct f
    f = np.zeros(N)
    for j in range(M):
        f += coeffs[j] * np.cos(2 * np.pi * j * k / N)
    
    # Enforce non-negativity and normalization
    f = np.clip(f, 0, None)
    norm = np.linalg.norm(f)
    if norm < 1e-15:
        return 1e10
    f = f / norm
    
    return compute_score(f)

def local_refinement(f, time_limit):
    """Perform local refinement on the full 600-dim vector."""
    start_time = time.time()
    best_f = f.copy()
    best_val = compute_score(f)
    
    # Gradient descent with adaptive step size
    eta = 0.05
    beta = 10.0
    max_iters = 5000
    
    for i in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Compute autoconvolution
        F = np.fft.fft(f)
        C = np.fft.ifft(F * F)
        c = np.real(C)
        
        current_val = np.max(c)
        
        if current_val < best_val:
            best_val = current_val
            best_f = f.copy()
        
        # Compute weights for smoothed max
        c_shifted = c - np.max(c)
        w = np.exp(beta * c_shifted)
        w_sum = np.sum(w)
        if w_sum == 0:
            w = np.ones(N) / N
        else:
            w /= w_sum
            
        # Gradient
        W = np.fft.fft(w)
        grad_F = (2.0 / N) * F * W
        grad_f = np.real(np.fft.ifft(grad_F))
        
        # Gradient descent step
        f_new = f - eta * grad_f
        f_new = np.maximum(f_new, 0)
        norm = np.linalg.norm(f_new)
        if norm > 0:
            f_new /= norm
        else:
            f_new = f.copy()
            
        # Check if step was good
        new_val = compute_score(f_new)
        
        if new_val < current_val:
            f = f_new
            # Increase step size slightly if improving
            eta = min(eta * 1.05, 0.2)
        else:
            # Decrease step size
            eta *= 0.8
            
    return best_f, best_val

def construct():
    start_time = time.time()
    
    # Stage 1: Parametric Optimization
    # Use first 10 cosine modes
    M = 10
    # Initial guess: uniform (only c_0 non-zero)
    c0 = np.sqrt(N)  # To get norm 1, c_0 * cos(0) = c_0, norm is c_0 * sqrt(N)
    # Actually, if f = c_0, norm is c_0 * sqrt(N). So c_0 = 1/sqrt(N).
    # Let's just start with small random coefficients
    x0 = np.zeros(M)
    x0[0] = 1.0 / np.sqrt(N)
    
    # Optimize using L-BFGS-B
    # We need to be careful with time.
    t1_start = time.time()
    time_for_stage1 = 60.0
    
    try:
        res = opt.minimize(parametric_objective, x0, method='L-BFGS-B', 
                           options={'maxiter': 500, 'maxfun': 500, 'disp': False})
        best_coeffs = res.x
    except Exception:
        best_coeffs = x0
        
    t1_end = time.time()
    
    # Reconstruct f from best_coeffs
    k = np.arange(N)
    f_param = np.zeros(N)
    for j in range(M):
        f_param += best_coeffs[j] * np.cos(2 * np.pi * j * k / N)
    f_param = np.clip(f_param, 0, None)
    norm = np.linalg.norm(f_param)
    if norm > 0:
        f_param /= norm
    else:
        f_param = np.ones(N) / N
        f_param /= np.linalg.norm(f_param)
        
    current_val = compute_score(f_param)
    
    # Stage 2: Local Refinement
    time_elapsed = time.time() - start_time
    remaining_time = TIME_LIMIT - time_elapsed - 5.0 # 5s safety margin
    if remaining_time > 10:
        f_refined, val_refined = local_refinement(f_param, remaining_time)
        if val_refined < current_val:
            f_param = f_refined
            current_val = val_refined
            
    return list(f_param)
