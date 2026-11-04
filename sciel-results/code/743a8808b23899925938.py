import numpy as np
import time

N = 50

def _autoconv(h):
    """Compute autoconvolution of h (full mode)."""
    return np.convolve(h, h, mode='full')

def _ratio(h):
    """Compute the ratio ||h||_2^2 / (||h||_1 * ||h*h||_inf)."""
    l1 = np.sum(h)
    if l1 <= 0:
        return 0.0
    l2_sq = np.sum(h**2)
    f = _autoconv(h)
    f_inf = np.max(f)
    if f_inf <= 0:
        return 0.0
    return l2_sq / (l1 * f_inf)

def _softmax_ratio(h, temperature=1.0):
    """Compute a smoothed version of the ratio using log-sum-exp for the peak."""
    l1 = np.sum(h)
    if l1 <= 0:
        return 0.0
    l2_sq = np.sum(h**2)
    f = _autoconv(h)
    # Use log-sum-exp for soft max
    max_f = np.max(f)
    if max_f <= 0:
        return 0.0
    # Soft max approximation: exp(logsumexp(f/temperature) - logsumexp(f/temperature)) 
    # Actually, we want a smooth approximation to max(f)
    # logsumexp(f/temperature) * temperature approximates max(f)
    log_f_scaled = f / temperature
    max_log_f = np.max(log_f_scaled)
    logsumexp_val = max_log_f + np.log(np.sum(np.exp(log_f_scaled - max_log_f)))
    f_inf_smooth = logsumexp_val * temperature
    if f_inf_smooth <= 0:
        return 0.0
    return l2_sq / (l1 * f_inf_smooth)

def _local_optimize(h_init, max_iter=500, lr=0.01, timeout_check=None, start_time=None, deadline=None):
    """Local optimization using projected gradient ascent on the ratio."""
    h = h_init.copy()
    h = np.maximum(h, 0)
    
    best_h = h.copy()
    best_ratio = _ratio(h)
    
    for i in range(max_iter):
        if timeout_check is not None and start_time is not None and deadline is not None:
            if time.time() > deadline:
                break
        
        # Compute gradient of log(ratio) w.r.t. h
        # ratio = ||h||_2^2 / (||h||_1 * ||h*h||_inf)
        # Let's work with log ratio for numerical stability
        l1 = np.sum(h)
        if l1 <= 0:
            break
        l2_sq = np.sum(h**2)
        f = _autoconv(h)
        f_inf = np.max(f)
        if f_inf <= 0:
            break
        
        ratio = l2_sq / (l1 * f_inf)
        if ratio > best_ratio:
            best_ratio = ratio
            best_h = h.copy()
        
        # Gradient computation
        # d(log ratio)/d(h_i) = 2*h_i/l2_sq - 1/l1 - (1/f_inf) * d(f_inf)/d(h_i)
        # d(f_inf)/d(h_i) is tricky since f_inf = max(f)
        # We approximate by considering which element achieves the max
        
        # Find indices where f is close to max
        f_max_idx = np.where(f >= f_inf * 0.999)[0]
        if len(f_max_idx) == 0:
            f_max_idx = [np.argmax(f)]
        
        # d(f_j)/d(h_i) = h_{j-i} for valid indices (in convolution)
        # For full convolution, f[j] = sum_k h[k]*h[j-k]
        # d(f[j])/d(h[i]) = h[j-i] + h[i-j] (if in range)
        
        grad = np.zeros(N)
        for j in f_max_idx:
            for i in range(N):
                # h[j-i] if 0 <= j-i < N
                if 0 <= j - i < N:
                    grad[i] += h[j - i]
                # h[i-j] if 0 <= i-j < N  
                if 0 <= i - j < N:
                    grad[i] += h[i - j]
        
        # grad now contains d(f_inf)/d(h_i) (approximate)
        grad_ratio = 2 * h / l2_sq - 1 / l1 - grad / f_inf
        
        # Project gradient ascent step
        h_new = h + lr * grad_ratio
        h_new = np.maximum(h_new, 0)
        
        # Normalize to keep scale reasonable (ratio is scale invariant)
        h_new_norm = np.sum(h_new)
        if h_new_norm > 0:
            h_new = h_new / h_new_norm * l1
        
        h = h_new
    
    return best_h, best_ratio

def _gaussian_profile(sigma, center=0.5):
    """Generate a Gaussian profile."""
    t = np.linspace(0, 1, N)
    return np.exp(-0.5 * ((t - center) / sigma) ** 2)

def _lorentzian_profile(gamma, center=0.5):
    """Generate a Lorentzian profile."""
    t = np.linspace(0, 1, N)
    return 1.0 / (1.0 + ((t - center) / gamma) ** 2)

def _powerlaw_profile(alpha, center=0.5, width=0.5):
    """Generate a power-law profile."""
    t = np.linspace(0, 1, N)
    # Shift and scale
    x = (t - center) / width
    return np.maximum(1.0 - np.abs(x), 0) ** alpha

def _two_peak_profile(sigma1, sigma2, c1, c2):
    """Generate a two-peak profile."""
    t = np.linspace(0, 1, N)
    g1 = np.exp(-0.5 * ((t - c1) / sigma1) ** 2)
    g2 = np.exp(-0.5 * ((t - c2) / sigma2) ** 2)
    return g1 + g2

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    best_h = np.ones(N)
    best_ratio = _ratio(best_h)
    
    # Generate diverse initializations
    initializations = []
    
    # Power laws
    for alpha in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
        initializations.append(_powerlaw_profile(alpha))
    
    # Gaussians with different widths and centers
    for sigma in [0.1, 0.2, 0.3, 0.4]:
        for center in [0.3, 0.5, 0.7]:
            initializations.append(_gaussian_profile(sigma, center))
    
    # Lorentzians
    for gamma in [0.1, 0.2, 0.3, 0.5]:
        for center in [0.3, 0.5, 0.7]:
            initializations.append(_lorentzian_profile(gamma, center))
    
    # Two peaks
    for s1 in [0.1, 0.2]:
        for s2 in [0.1, 0.2]:
            for c1 in [0.3]:
                for c2 in [0.7]:
                    initializations.append(_two_peak_profile(s1, s2, c1, c2))
    
    # Random initializations
    for _ in range(10):
        h_rand = np.random.rand(N)
        initializations.append(h_rand)
    
    # Evaluate all initializations
    for h_init in initializations:
        if time.time() > deadline:
            break
        h_init = np.maximum(h_init, 0)
        r = _ratio(h_init)
        if r > best_ratio:
            best_ratio = r
            best_h = h_init.copy()
    
    # Local optimization from best initialization and top few
    # Sort by ratio and take top 5
    scored_inits = []
    for h_init in initializations:
        h_init = np.maximum(h_init, 0)
        r = _ratio(h_init)
        scored_inits.append((r, h_init))
    
    scored_inits.sort(key=lambda x: -x[0])
    
    top_k = min(5, len(scored_inits))
    for i in range(top_k):
        if time.time() > deadline:
            break
        _, h_init = scored_inits[i]
        h_opt, r_opt = _local_optimize(
            h_init, 
            max_iter=200, 
            lr=0.005,
            timeout_check=lambda: True,
            start_time=start_time,
            deadline=deadline
        )
        if r_opt > best_ratio:
            best_ratio = r_opt
            best_h = h_opt
    
    # Final polish with smaller learning rate
    if time.time() < deadline - 20:
        h_opt, r_opt = _local_optimize(
            best_h,
            max_iter=300,
            lr=0.001,
            timeout_check=lambda: True,
            start_time=start_time,
            deadline=deadline
        )
        if r_opt > best_ratio:
            best_ratio = r_opt
            best_h = h_opt
    
    # Ensure non-negative
    best_h = np.maximum(best_h, 0)
    
    return list(best_h)
