"""
First autocorrelation inequality: minimize normalized max of autoconvolution
for a nonnegative step function with 600 equal intervals on [-1/4, 1/4].

Strategy:
1. Direct optimization on bin values using FFT-accelerated autoconvolution.
2. Subgradient method with soft-max surrogate for the peak objective.
3. Coordinate descent polish for per-bin optima.
4. Final high-precision refinement pass.
"""

import numpy as np
import time

N = 600
TIME_LIMIT = 135  # safety margin before 150s kill

def compute_score(f):
    """Compute the normalized maximum of the autoconvolution."""
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-15):
        return np.inf
    f = np.maximum(f, 0.0)
    conv = np.convolve(f, f, mode='full')
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    return np.max(conv) / (s * s)

def fft_autocorr(f):
    """Compute full autoconvolution using FFT."""
    n_fft = 2 * N
    F = np.fft.rfft(f, n=n_fft)
    power = F * np.conj(F)
    conv = np.fft.irfft(power, n=n_fft)[:2*N-1]
    return conv

def smooth_max(x, beta):
    """Smooth approximation to max using log-sum-exp."""
    x = np.asarray(x, dtype=np.float64)
    x_max = np.max(x)
    if x_max < 1e-15:
        return 0.0
    x_scaled = x / x_max
    shifted = x_scaled - np.max(x_scaled)
    log_sum = np.log(np.sum(np.exp(beta * shifted))) / beta
    return x_max * (1.0 + log_sum)

def objective_direct(f, beta=10.0):
    """Objective: smooth max of autoconvolution / (sum^2)."""
    conv = fft_autocorr(f)
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    return smooth_max(conv, beta) / (s * s)

def subgradient_step(f, beta=10.0, lr=0.01):
    """
    One subgradient step for minimizing smooth_max(conv)/(sum^2)
    subject to f >= 0.
    
    The gradient of smooth_max(conv, beta) w.r.t. f_j involves
    the derivative of each conv[m] w.r.t. f_j, weighted by softmax.
    """
    conv = fft_autocorr(f)
    s = np.sum(f)
    
    # Softmax weights for the smooth max
    x_max = np.max(conv)
    if x_max < 1e-15:
        return f.copy()
    shifted = conv / x_max - 1.0
    weights = np.exp(beta * shifted) / np.sum(np.exp(beta * shifted))
    
    # Gradient of smooth_max(conv) w.r.t. f_j:
    # d(smooth_max)/df_j = sum_m weights[m] * d(conv[m])/df_j
    # conv[m] = sum_j f[j]*f[m-j], so d(conv[m])/df_j = f[m-j] (with appropriate bounds)
    
    grad_conv = np.zeros(N)
    for m in range(2*N-1):
        w = weights[m]
        if w < 1e-12:
            continue
        j_min = max(0, m - (N - 1))
        j_max = min(N - 1, m)
        for j in range(j_min, j_max + 1):
            grad_conv[j] += w * f[m - j]
    
    # Objective = smooth_max(conv) / s^2
    # d(obj)/df_j = (grad_conv[j] * s^2 - smooth_max(conv) * 2*s) / s^4
    #            = grad_conv[j] / s^2 - smooth_max(conv) / s^3
    
    sm_val = smooth_max(conv, beta)
    grad_obj = grad_conv / (s * s) - sm_val / (s * s * s)
    
    # Projected gradient step with nonnegativity constraint
    f_new = f - lr * grad_obj
    f_new = np.maximum(f_new, 0.0)
    
    return f_new

def coordinate_descent_polish(f, max_iters=5000, start_time=None, time_limit=None):
    """
    Coordinate descent: for each bin j, find optimal value given others fixed.
    The contribution of f[j] to conv[m] is f[j]*f[m-j] for valid m-j.
    """
    best_f = f.copy()
    best_val = compute_score(f)
    current_f = f.copy()
    
    if start_time is None:
        start_time = time.time()
    if time_limit is None:
        time_limit = TIME_LIMIT - 5
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        improved = False
        for j in range(N):
            # Find optimal f[j] given others fixed
            # conv[m] = sum_{j'} f[j']*f[m-j']
            # Terms involving f[j]: f[j]*f[m-j] for each m where m-j is valid
            # = f[j] * (sum over valid m of f[m-j] * weight_m)
            # plus f[j]^2 term when m=j (self-convolution)
            
            # For simplicity, use a local quadratic approximation
            # Try a few candidate values
            f_old = current_f[j]
            if f_old < 1e-12:
                # Try small positive values
                candidates = [0.0, f_old + 0.01, f_old + 0.05, 0.1]
            else:
                candidates = [f_old * 0.5, f_old * 0.8, f_old, f_old * 1.2, f_old * 1.5]
            
            best_cand = f_old
            best_cand_val = compute_score(current_f)
            
            for c in candidates:
                c = max(0.0, c)
                current_f[j] = c
                val = compute_score(current_f)
                if val < best_cand_val:
                    best_cand_val = val
                    best_cand = c
            
            current_f[j] = best_cand
            if best_cand != f_old:
                improved = True
        
        if not improved:
            break
        
        val = compute_score(current_f)
        if val < best_val:
            best_val = val
            best_f = current_f.copy()
    
    return best_f, best_val

def subgradient_descent(f_init, start_time, max_time, beta_init=5.0, lr_init=0.05):
    """Subgradient descent with adaptive learning rate and beta annealing."""
    best_f = f_init.copy()
    best_val = compute_score(f_init)
    current_f = f_init.copy()
    beta = beta_init
    lr = lr_init
    
    # Normalize to sum = N
    s = np.sum(current_f)
    if s > 1e-12:
        current_f = current_f * (N / s)
    
    no_improve_count = 0
    
    while time.time() - start_time < max_time:
        # Adaptive beta: increase over time for sharper max approximation
        elapsed_frac = min(1.0, (time.time() - start_time) / max(max_time, 1.0))
        beta = beta_init + 20.0 * elapsed_frac
        
        f_new = subgradient_step(current_f, beta, lr)
        
        # Renormalize to maintain sum = N
        s_new = np.sum(f_new)
        if s_new > 1e-12:
            f_new = f_new * (N / s_new)
        
        val_new = compute_score(f_new)
        
        if val_new < best_val:
            best_val = val_new
            best_f = f_new.copy()
            no_improve_count = 0
            lr = min(lr * 1.02, 0.2)
        else:
            no_improve_count += 1
            lr *= 0.95
            if no_improve_count > 100:
                # Reset to best and try different direction
                current_f = best_f.copy()
                s = np.sum(current_f)
                if s > 1e-12:
                    current_f = current_f * (N / s)
                no_improve_count = 0
                lr = lr_init * 0.5
        
        current_f = f_new
        
        if lr < 1e-6:
            break
    
    return best_f, best_val

def construct():
    start_time = time.time()
    
    # Initial best: constant function
    best_f = np.ones(N)
    best_score = compute_score(best_f)
    
    # Multiple random initializations for subgradient descent
    rng = np.random.RandomState(42)
    
    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() - start_time > TIME_LIMIT - 30:
            break
        
        if restart == 0:
            # Constant
            f_init = np.ones(N)
        elif restart == 1:
            # Slightly perturbed constant
            f_init = np.ones(N) * (1.0 + 0.1 * rng.randn(N))
            f_init = np.maximum(f_init, 0.0)
        elif restart == 2:
            # Two-peak structure
            x = np.linspace(0, N, N)
            f_init = 1.0 + 0.5 * np.cos(2 * np.pi * x / N)
            f_init = np.maximum(f_init, 0.0)
        elif restart == 3:
            # Three-peak structure
            x = np.linspace(0, N, N)
            f_init = 1.0 + 0.3 * np.cos(2 * np.pi * x / N) + 0.2 * np.cos(4 * np.pi * x / N)
            f_init = np.maximum(f_init, 0.0)
        else:
            # Random smooth
            freqs = np.random.randn(N)
            f_init = np.abs(np.fft.irfft(freqs, n=N)) + 0.1
            f_init = np.maximum(f_init, 0.0)
        
        # Subgradient descent
        sg_f, sg_score = subgradient_descent(f_init, start_time, TIME_LIMIT - 15)
        if sg_score < best_score:
            best_f = sg_f
            best_score = sg_score
    
    # Coordinate descent polish
    if time.time() - start_time < TIME_LIMIT - 10:
        cd_f, cd_score = coordinate_descent_polish(
            best_f, max_iters=2000, 
            start_time=start_time, 
            time_limit=TIME_LIMIT - 5
        )
        if cd_score < best_score:
            best_f = cd_f
            best_score = cd_score
    
    # Final short subgradient pass for polish
    if time.time() - start_time < TIME_LIMIT - 3:
        sg_f2, sg_score2 = subgradient_descent(
            best_f, start_time, TIME_LIMIT - 1,
            beta_init=25.0, lr_init=0.01
        )
        if sg_score2 < best_score:
            best_f = sg_f2
            best_score = sg_score2
    
    # Ensure nonnegative
    best_f = np.maximum(best_f, 0.0)
    return [float(x) for x in best_f]

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
