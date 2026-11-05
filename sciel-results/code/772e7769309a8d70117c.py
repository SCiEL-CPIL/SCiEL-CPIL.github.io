"""Rewritten program for third autocorrelation inequality optimization."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _objective_hard(h):
    """Compute the hard objective value."""
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    conv = np.convolve(h, h, mode='full')
    return abs(2.0 * N * np.max(conv) / (s * s))

def _soft_max_obj(h, beta):
    """
    Soft-max approximation of the objective.
    Minimizes 2*N*soft_max(conv) / (sum(h))^2
    """
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    conv = np.convolve(h, h, mode='full')
    m = np.max(conv)
    # Numerically stable log-sum-exp
    shifted = beta * (conv - m)
    shifted = np.clip(shifted, -500, 500)
    log_sum_exp = m + np.log(np.sum(np.exp(shifted))) / beta
    val = 2.0 * N * log_sum_exp / (s * s)
    return val

def _soft_max_grad(h, beta):
    """
    Gradient of the soft-max objective.
    d/dh [2*N*soft_max(conv)/(sum(h))^2]
    """
    s = np.sum(h)
    if abs(s) < 1e-10:
        return np.zeros(N)
    conv = np.convolve(h, h, mode='full')
    m = np.max(conv)
    shifted = beta * (conv - m)
    shifted = np.clip(shifted, -500, 500)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    
    # soft_max = m + log(sum_exp)/beta
    # d(soft_max)/d(conv[j]) = exp_shifted[j] / sum_exp
    # d(conv[j])/dh[i] = h[i + (j - (N-1))] if valid index, else 0
    # conv has length 2N-1, indexed 0..2N-2
    # conv[j] = sum_{i} h[i] * h[i + j - (N-1)]
    
    # We need d(soft_max)/dh = sum_j d(soft_max)/d(conv[j]) * d(conv[j])/dh
    # d(soft_max)/d(conv[j]) = w_j where w_j = exp_shifted[j] / sum_exp
    
    # Let's compute the gradient via the chain rule
    # d(conv[j])/dh[i] = h[i + j - N + 1] + h[i - j + N - 1] (if indices valid)
    # Actually conv[j] = sum_k h[k] * h[k + j - N + 1]
    # d(conv[j])/dh[i] = h[i + j - N + 1] (if 0 <= i+j-N+1 < N) + h[i - j + N - 1] (if 0 <= i-j+N-1 < N)
    # But wait, conv is symmetric: conv[j] = conv[2N-2-j]
    # Let's use a more direct approach.
    
    # Let w = exp_shifted / sum_exp (weights for each conv entry)
    w = exp_shifted / sum_exp
    
    # grad_conv = sum_j w_j * d(conv[j])/dh
    # d(conv[j])/dh[i] = h[i + (j - N + 1)] if valid + h[i - (j - N + 1)] if valid
    # Let k = j - N + 1, so k ranges from -(N-1) to N-1
    # conv[j] = sum_i h[i] * h[i+k] where k = j - N + 1
    # d(conv[j])/dh[i] = h[i+k] + h[i-k] (with boundary handling)
    
    # This is getting complex. Let's use a different formulation.
    # Note that soft_max(conv) = max_k conv[k] approx
    # And conv = h * h (autoconvolution)
    
    # Alternative: use the fact that the gradient of max(conv) is the conv entry at the max.
    # For soft max, the gradient is the weighted average of the gradients of each conv entry.
    
    # Let's compute it numerically for safety, or use a known formula.
    # Actually, let's just use scipy's minimize with numerical gradients if needed, 
    # but that's slow. Let's try to derive it properly.
    
    # conv[j] = sum_{i=0}^{N-1} h[i] * h[i + j - N + 1]
    # Let's define the gradient of the soft max value with respect to h.
    # dV/dh[i] = 2*N * (1/s^2) * d(soft_max)/dh[i] - 2*N * soft_max * 2*s / s^4 * ds/dh[i]
    # ds/dh[i] = 1
    # d(soft_max)/dh[i] = sum_j w_j * d(conv[j])/dh[i]
    
    # d(conv[j])/dh[i] = h[i + j - N + 1] if 0 <= i+j-N+1 < N else 0
    #                 + h[i - j + N - 1] if 0 <= i-j+N-1 < N else 0
    # But since conv is symmetric, and w is symmetric (because conv is symmetric and soft max preserves symmetry),
    # we can simplify.
    
    # Let's just compute the gradient of the soft max of conv with respect to h.
    # This is equivalent to the convolution of w with h, in a specific way.
    
    # Let's use a simpler approach: since we're using L-BFGS-B which can do numerical gradients,
    # and N=400, it might be acceptable. But let's try to be efficient.
    
    # Actually, let's just return the objective and let scipy handle gradients numerically.
    # For N=400, this should be fine within the time budget.
    return None  # Signal to use numerical gradients

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    best_h = np.ones(N)
    best_val = _objective_hard(best_h)
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initializations
    initializations = []
    
    # 1. Random normal with positive mean
    for _ in range(15):
        h0 = rng.standard_normal(N)
        h0 += 0.5  # Ensure positive mean
        if abs(np.sum(h0)) < 1e-3:
            h0[0] += 1.0
        initializations.append(h0)
        
    # 2. Sign sequences with DC offset
    for _ in range(15):
        h0 = rng.choice([-1, 1], size=N).astype(float)
        h0 += 0.3 * rng.standard_normal()
        if abs(np.sum(h0)) < 1e-3:
            h0[0] += 1.0
        initializations.append(h0)
        
    # 3. Sine waves with various frequencies and phases
    for freq in [1, 2, 3, 5, 7, 10]:
        t = np.linspace(0, 2*np.pi*freq, N, endpoint=False)
        h0 = np.sin(t + rng.random()*2*np.pi)
        h0 += 0.8  # Positive mean
        initializations.append(h0)
        
    # 4. Polynomial perturbations
    for _ in range(10):
        h0 = np.ones(N)
        t = np.linspace(0, 1, N)
        h0 += 0.2 * (rng.random() * t**2 - 0.1)
        if abs(np.sum(h0)) < 1e-3:
            h0[0] += 1.0
        initializations.append(h0)
        
    # 5. Gaussian bumps
    for _ in range(10):
        h0 = np.zeros(N)
        center = rng.integers(0, N)
        width = rng.uniform(10, 50)
        t = np.arange(N)
        h0 += np.exp(-0.5 * ((t - center) / width)**2)
        h0 += 0.1 * rng.standard_normal(N)
        if abs(np.sum(h0)) < 1e-3:
            h0[0] += 1.0
        initializations.append(h0)

    # Optimization using scipy's minimize with BFGS
    from scipy.optimize import minimize
    
    betas = [30.0, 50.0, 100.0, 200.0]
    
    for init_idx, h0 in enumerate(initializations):
        if time.time() > deadline - 25.0:
            break
        
        h_curr = h0.copy()
        
        for beta in betas:
            if time.time() > deadline - 25.0:
                break
            
            def obj_func(x):
                return _soft_max_obj(x, beta)
            
            try:
                res = minimize(obj_func, h_curr, 
                              method='L-BFGS-B',
                              options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 30})
                h_new = res.x
            except Exception:
                h_new = h_curr
            
            val_new = _objective_hard(h_new)
            
            if val_new < best_val:
                best_val = val_new
                best_h = h_new.copy()
                h_curr = h_new.copy()
            else:
                val_curr = _objective_hard(h_curr)
                if val_new < val_curr:
                    h_curr = h_new.copy()

    # Final polish with high beta
    if time.time() < deadline - 15.0:
        for beta in [200.0, 500.0, 1000.0]:
            if time.time() > deadline - 10.0:
                break
            def obj_func(x):
                return _soft_max_obj(x, beta)
            try:
                res = minimize(obj_func, best_h,
                              method='L-BFGS-B',
                              options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50})
                val = _objective_hard(res.x)
                if val < best_val:
                    best_val = val
                    best_h = res.x.copy()
            except Exception:
                pass

    # Ensure sum is not zero
    if abs(np.sum(best_h)) < 1e-10:
        best_h = np.ones(N)
        best_val = _objective_hard(best_h)
        
    return best_h
# EVOLVE-BLOCK-END
