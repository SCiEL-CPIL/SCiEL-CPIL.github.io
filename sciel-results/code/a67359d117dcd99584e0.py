"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog

def construct():
    """
    Construct a signed step function with 400 equal intervals on [-1/4, 1/4]
    to minimize the L-infinity norm of the autoconvolution.
    
    The autoconvolution of a step function h with 400 equal intervals is:
    (h * h)[k] = sum_{i=0}^{k} h[i] * h[k-i] for k = 0, ..., 2N-2
    
    We want to choose h[i] in {-1, 1} to minimize max_k |(h*h)[k]|.
    
    Strategy:
    1. Use continuous relaxation: x in [-1, 1]^N
    2. Minimize L-infinity norm of autoconvolution via projected gradient descent
    3. Discretize to {-1, 1} via hard thresholding
    4. Local search polish with simulated annealing
    """
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    # Initialize with alternating pattern (known to give good autoconvolution bounds)
    x = np.ones(N)
    x[1::2] = -1.0
    
    def autoconvolution(h):
        """Compute autoconvolution of h with itself."""
        return np.convolve(h, h, mode='full')
    
    def compute_score(h):
        """Compute L-infinity norm of autoconvolution."""
        conv = autoconvolution(h)
        return np.max(np.abs(conv))
    
    # Projected gradient descent on continuous relaxation
    # Objective: min_t t s.t. |conv_k| <= t for all k
    # Equivalent to: min max_k |conv_k|
    
    # Use smooth approximation: soft-max via log-sum-exp
    def soft_max_smooth(conv, beta=10.0):
        """Smooth approximation of max(|conv_k|) using log-sum-exp."""
        abs_conv = np.abs(conv)
        # Scale to avoid overflow
        max_val = np.max(abs_conv)
        if max_val < 1e-10:
            return 0.0
        scaled = abs_conv / max_val
        # Log-sum-exp trick
        m = np.max(scaled)
        return max_val * (m + np.log(np.sum(np.exp(scaled - m)) / len(scaled)))
    
    def gradient_of_soft_max(conv, beta=10.0):
        """Gradient of soft max with respect to h."""
        abs_conv = np.abs(conv)
        max_val = np.max(abs_conv)
        if max_val < 1e-10:
            return np.zeros_like(conv)
        scaled = abs_conv / max_val
        m = np.max(scaled)
        weights = np.exp(scaled - m)
        weights /= np.sum(weights)
        
        # Gradient of |conv_k| w.r.t. h
        # conv_k = sum_i h[i] h[k-i]
        # d|conv_k|/dh[i] = sign(conv_k) * (h[k-i] + h[i] if i != k-i else 2*h[i])
        grad = np.zeros(N)
        for k in range(2*N - 1):
            if abs_conv[k] < 1e-10:
                continue
            sign_k = np.sign(conv[k])
            weight = weights[k] * sign_k
            for i in range(min(k+1, N)):
                j = k - i
                if j >= N:
                    continue
                if i == j:
                    grad[i] += 2.0 * weight * h[i]
                else:
                    grad[i] += weight * h[j]
                    grad[j] += weight * h[i]
        return grad
    
    # Simpler approach: use subgradient method directly
    def subgradient_method(h, max_iters=500, step_size=0.01):
        """Subgradient method for minimizing max |conv_k|."""
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            conv = autoconvolution(h)
            abs_conv = np.abs(conv)
            max_val = np.max(abs_conv)
            max_indices = np.where(abs_conv == max_val)[0]
            
            # Subgradient: average of gradients at active constraints
            grad = np.zeros(N)
            for k in max_indices:
                sign_k = np.sign(conv[k])
                for i in range(min(k+1, N)):
                    j = k - i
                    if j >= N:
                        continue
                    if i == j:
                        grad[i] += sign_k * 2.0 * h[i]
                    else:
                        grad[i] += sign_k * h[j]
                    grad[j] += sign_k * h[i]
            grad /= len(max_indices)
            
            # Projected gradient step
            lr = step_size / (1.0 + it * 0.01)
            h_new = h - lr * grad
            h_new = np.clip(h_new, -1.0, 1.0)
            
            # Accept if better
            if compute_score(h_new) < compute_score(h):
                h = h_new
                step_size *= 1.01
            else:
                step_size *= 0.95
        
        return h
    
    # Run subgradient method
    x_continuous = subgradient_method(x.copy(), max_iters=200, step_size=0.05)
    
    # Discretize to {-1, 1}
    x_binary = np.where(x_continuous >= 0, 1.0, -1.0)
    
    # Local search polish with simulated annealing
    def local_search(h, max_iters=10000, t_start=1.0, t_end=0.01):
        """Simulated annealing local search."""
        current = h.copy()
        current_score = compute_score(current)
        best = current.copy()
        best_score = current_score
        
        n = len(h)
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Temperature schedule
            t = t_start * (t_end / t_start) ** (it / max_iters)
            
            # Random flip
            idx = np.random.randint(0, n)
            new_h = current.copy()
            new_h[idx] = -new_h[idx]
            new_score = compute_score(new_h)
            
            delta = new_score - current_score
            if delta < 0 or np.random.rand() < np.exp(-delta / max(t, 1e-10)):
                current = new_h
                current_score = new_score
                if current_score < best_score:
                    best = current.copy()
                    best_score = current_score
        
        return best, best_score
    
    # Run local search
    x_binary, best_score = local_search(x_binary, max_iters=50000, t_start=0.5, t_end=0.001)
    
    # Try multiple restarts if time permits
    for restart in range(5):
        if time.time() > deadline - 15:
            break
        
        # Different initial patterns
        if restart == 0:
            init = np.ones(N)
        elif restart == 1:
            init = np.ones(N)
            init[::2] = -1.0
        elif restart == 2:
            init = np.random.choice([-1.0, 1.0], size=N)
        elif restart == 3:
            # Block pattern
            init = np.ones(N)
            init[100:200] = -1.0
            init[300:400] = -1.0
        else:
            # Sinusoidal pattern
            t = np.linspace(0, 4*np.pi, N)
            init = np.sign(np.sin(t))
            init[init == 0] = 1.0
        
        # Quick subgradient
        x_c = subgradient_method(init.copy(), max_iters=50, step_size=0.03)
        x_b = np.where(x_c >= 0, 1.0, -1.0)
        
        # Local search
        x_b, score = local_search(x_b, max_iters=10000, t_start=0.3, t_end=0.005)
        
        if score < best_score:
            best_score = score
            x_binary = x_b
    
    return x_binary
# EVOLVE-BLOCK-END
