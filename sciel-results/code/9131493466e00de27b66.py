"""Rewritten program for third autocorrelation inequality optimization."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _autoconv_max(h, beta=50.0):
    """Soft-max approximation of max of autoconvolution entries."""
    conv = np.convolve(h, h, mode='full')
    m = np.max(conv)
    if m > 0:
        # Log-sum-exp for numerical stability
        shifted = beta * (conv - m)
        # Clip to avoid overflow
        shifted = np.clip(shifted, -500, 500)
        return m + np.log(np.sum(np.exp(shifted))) / beta
    else:
        return np.max(conv)

def _soft_max_obj(x, beta=50.0):
    """Objective to minimize: 2*N*max(autoconv) / (sum(h))^2 using soft-max."""
    h = x
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    conv_max = _autoconv_max(h, beta)
    return 2.0 * N * conv_max / (s * s)

def _soft_max_grad(x, beta=50.0):
    """Gradient of the soft-max objective."""
    h = x
    s = np.sum(h)
    if abs(s) < 1e-10:
        return np.zeros_like(h)
    
    conv = np.convolve(h, h, mode='full')
    m = np.max(conv)
    if m > 0:
        shifted = beta * (conv - m)
        shifted = np.clip(shifted, -500, 500)
        exp_shifted = np.exp(shifted)
        weights = exp_shifted / np.sum(exp_shifted)
        soft_max = m + np.log(np.sum(exp_shifted)) / beta
        
        # Gradient of soft_max w.r.t. h
        # d(soft_max)/d(h_i) = sum_j weights_j * d(conv_j)/d(h_i)
        # conv_j = sum_k h_k * h_{j-k}
        # d(conv_j)/d(h_i) = h_{j-i} + h_{i-j} if indices valid
        # This is complex, so use numerical gradient as fallback or simplify
        
        # For now, use a simpler approach: gradient of 2*N*soft_max/s^2
        # = 2*N * (d(soft_max)/dh * s^2 - soft_max * 2*s*1) / s^4
        # = 2*N * (d(soft_max)/dh * s - 2*soft_max) / s^3
        
        # Approximate d(soft_max)/dh using the weights
        # The gradient of max(conv) w.r.t. h_i is approximately
        # sum_j weights_j * (h_{j-i} + h_{i-j}) where indices are valid
        # This is essentially a weighted convolution
        
        grad_soft_max = np.zeros_like(h)
        for j in range(len(conv)):
            if weights[j] > 1e-15:
                # d(conv_j)/d(h_i) for all i
                # conv_j = sum_k h_k h_{j-k}
                # For fixed j, d(conv_j)/d(h_i) = h_{j-i} + h_{i-j} if valid
                # This is equivalent to: for each i, add h[j-i] if 0<=j-i<N and h[i-j] if 0<=i-j<N
                # Which is the same as convolving weights with h and h reversed
                pass
        
        # Simpler: use the fact that gradient of soft-max of conv w.r.t. h
        # can be computed via the weighted correlation
        # Let's use a vectorized approach
        # weights has length 2N-1
        # For each i in 0..N-1, d(soft_max)/d(h_i) = sum_j weights_j * d(conv_j)/d(h_i)
        # d(conv_j)/d(h_i) = h_{j-i} + h_{i-j} (with 0 outside range)
        
        # We can compute this as:
        # term1: for each i, sum_j weights_j * h_{j-i} where 0<=j-i<N
        # term2: for each i, sum_j weights_j * h_{i-j} where 0<=i-j<N
        # Note that term2 is just term1 with indices flipped, so they're equal
        # Actually no, let's be careful
        
        # Let's just do it numerically for now since the analytic gradient is complex
        eps = 1e-7
        grad = np.zeros_like(h)
        for i in range(N):
            h_plus = h.copy()
            h_plus[i] += eps
            h_minus = h.copy()
            h_minus[i] -= eps
            grad[i] = (_soft_max_obj(h_plus, beta) - _soft_max_obj(h_minus, beta)) / (2*eps)
        
        # Now compute full gradient of 2*N*soft_max/s^2
        # f = 2*N*soft_max/s^2
        # df/dh_i = 2*N * (grad_soft_max_i * s^2 - soft_max * 2*s) / s^4
        #          = 2*N * (grad_soft_max_i * s - 2*soft_max) / s^3
        return 2.0 * N * (grad * s - 2.0 * soft_max) / (s * s * s)
    else:
        return np.zeros_like(h)

def _objective_hard(x):
    """Hard objective for final evaluation."""
    h = x
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    conv = np.convolve(h, h, mode='full')
    return abs(2.0 * N * np.max(conv) / (s * s))

def _anneal_temperature(iteration, total_iterations, t_start=1.0, t_end=0.01):
    """Exponential annealing schedule."""
    return t_start * (t_end / t_start) ** (iteration / total_iterations)

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    best_h = np.ones(N)
    best_val = _objective_hard(best_h)
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initializations
    initializations = []
    
    # 1. Random normal
    for _ in range(5):
        h0 = rng.standard_normal(N)
        if abs(np.sum(h0)) < 1e-6:
            h0[0] += 1.0
        initializations.append(h0)
    
    # 2. Comb patterns (alternating signs with varying period)
    for period in [2, 3, 4, 5]:
        h0 = np.zeros(N)
        for i in range(N):
            h0[i] = 1.0 if (i % period) < period // 2 else -1.0
        # Add small noise
        h0 += 0.1 * rng.standard_normal(N)
        initializations.append(h0)
    
    # 3. Higher frequency waves
    for freq in [2, 5, 10, 20]:
        t = np.linspace(0, 2*np.pi*freq, N)
        h0 = np.sin(t) + 0.5 * np.cos(2*t)
        h0 += 0.1 * rng.standard_normal(N)
        initializations.append(h0)
    
    # 4. Polynomial bases
    for deg in [1, 2, 3]:
        x = np.linspace(-1, 1, N)
        h0 = np.zeros(N)
        for d in range(deg + 1):
            h0 += rng.standard_normal() * x**d
        h0 += 0.1 * rng.standard_normal(N)
        initializations.append(h0)
    
    # 5. Block patterns
    for block_size in [10, 20, 50]:
        h0 = np.zeros(N)
        n_blocks = N // block_size
        for b in range(n_blocks):
            h0[b*block_size:(b+1)*block_size] = rng.standard_normal()
        initializations.append(h0)
    
    # Optimize each initialization with multi-temperature annealing
    total_iterations = 100
    betas = [20.0, 50.0, 100.0]
    
    for init_idx, h0 in enumerate(initializations):
        if time.time() > deadline - 30.0:
            break
        
        # Try different beta values
        for beta in betas:
            if time.time() > deadline - 30.0:
                break
            
            h_curr = h0.copy()
            
            # Multi-temperature annealing
            for temp_iter in range(total_iterations):
                if time.time() > deadline - 30.0:
                    break
                
                temp = _anneal_temperature(temp_iter, total_iterations)
                
                # Use L-BFGS-B with the soft-max objective
                try:
                    res = minimize(_soft_max_obj, h_curr, 
                                  method='L-BFGS-B',
                                  args=(beta,),
                                  options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-6})
                    h_new = res.x
                except Exception:
                    h_new = h_curr
                
                # Simulated annealing acceptance
                val_curr = _objective_hard(h_curr)
                val_new = _objective_hard(h_new)
                
                if val_new < val_curr:
                    h_curr = h_new
                else:
                    # Accept with probability exp(-(val_new - val_curr) / temp)
                    delta_val = val_new - val_curr
                    if delta_val < 100 * temp:
                        prob = np.exp(-delta_val / temp)
                        if rng.random() < prob:
                            h_curr = h_new
                
                # Update best
                val_curr = _objective_hard(h_curr)
                if val_curr < best_val:
                    best_val = val_curr
                    best_h = h_curr.copy()
            
            # Final polish on the best from this beta
            if time.time() < deadline - 20.0:
                try:
                    res = minimize(_soft_max_obj, h_curr,
                                  method='L-BFGS-B',
                                  args=(beta,),
                                  options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
                    val = _objective_hard(res.x)
                    if val < best_val:
                        best_val = val
                        best_h = res.x.copy()
                except Exception:
                    pass
    
    # Final high-precision polish on the best found
    if time.time() < deadline - 10.0:
        for beta in [100.0, 200.0]:
            try:
                res = minimize(_soft_max_obj, best_h,
                              method='L-BFGS-B',
                              args=(beta,),
                              options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
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
    
    # Safety margin check
    if time.time() > deadline - 5.0:
        return best_h
    
    return best_h
# EVOLVE-BLOCK-END
