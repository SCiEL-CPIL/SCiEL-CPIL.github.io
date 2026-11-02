"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin
    
    # Initialize with random ±1 signal
    np.random.seed(42)
    x = np.random.choice([-1.0, 1.0], size=N)
    
    def autoconv(x):
        # Full autoconvolution
        return np.correlate(x, x, mode='full')
    
    def c3prime(x):
        ac = autoconv(x)
        max_abs = np.max(np.abs(ac))
        integral_sq = np.sum(x) ** 2
        if integral_sq < 1e-10:
            return np.inf
        return max_abs / integral_sq
    
    best_x = x.copy()
    best_val = c3prime(x)
    
    # Smooth surrogate: log-sum-exp of squared autoconv values
    def surrogate_loss(x, beta=10.0):
        ac = autoconv(x)
        ac_sq = ac ** 2
        # Log-sum-exp for numerical stability
        max_val = np.max(ac_sq)
        if max_val < 1e-10:
            return 1e-10
        loss = np.log(np.sum(np.exp(ac_sq / beta - max_val / beta))) * beta + max_val
        # Penalize small integral
        integral = np.sum(x)
        penalty = 1.0 / (1.0 + integral ** 2) * 1000
        return loss + penalty
    
    def surrogate_grad(x, beta=10.0):
        ac = autoconv(x)
        ac_sq = ac ** 2
        max_val = np.max(ac_sq)
        if max_val < 1e-10:
            return np.zeros_like(x)
        # Gradient of log-sum-exp
        exp_vals = np.exp(ac_sq / beta - max_val / beta)
        sum_exp = np.sum(exp_vals)
        # d/dx of ac[k] where ac[k] = sum_i x[i]*x[i+k]
        # This is complex; use finite differences or analytical
        # For now, use a simple approach: perturb each element
        grad = np.zeros_like(x)
        for i in range(N):
            if time.time() > deadline:
                break
            x[i] += 1e-4
            l_plus = surrogate_loss(x, beta)
            x[i] -= 2e-4
            l_minus = surrogate_loss(x, beta)
            x[i] += 1e-4
            grad[i] = (l_plus - l_minus) / (2e-4)
        return grad
    
    # Gradient descent with projection
    lr = 0.1
    beta = 5.0
    
    for step in range(2000):
        if time.time() > deadline:
            break
        
        # Compute gradient (use a cheaper method)
        # Use analytical gradient for autoconv
        ac = autoconv(x)
        ac_sq = ac ** 2
        max_val = np.max(ac_sq)
        if max_val < 1e-10:
            break
        
        exp_vals = np.exp(ac_sq / beta - max_val / beta)
        weights = exp_vals / np.sum(exp_vals)
        
        # Gradient of sum(weights * ac_sq) w.r.t. x
        # ac[k] = sum_i x[i] * x[i+k] (with appropriate indexing)
        # For full correlation: ac_full[k] = sum_i x[i] * x[i+k-N+1] for k in [0, 2N-2]
        # Let's use the fact that d(ac_full[k])/dx[j] = x[j+k-N+1] + x[j-(k-N+1)] if in range
        
        grad = np.zeros_like(x)
        for k in range(len(ac)):
            if abs(weights[k]) < 1e-10:
                continue
            lag = k - (N - 1)
            # ac[k] = sum_i x[i] * x[i+lag] for valid i
            # d(ac[k])/dx[j] = x[j+lag] (if j+lag in [0,N)) + x[j-lag] (if j-lag in [0,N))
            for j in range(N):
                if j + lag >= 0 and j + lag < N:
                    grad[j] += weights[k] * 2 * ac[k] * x[j + lag]
                if j - lag >= 0 and j - lag < N:
                    grad[j] += weights[k] * 2 * ac[k] * x[j - lag]
        
        # Update
        x = x - lr * grad
        
        # Project to ±1
        x = np.sign(x)
        # Handle zeros
        x[x == 0] = 1.0
        
        val = c3prime(x)
        if val < best_val:
            best_val = val
            best_x = x.copy()
        
        # Adaptive learning rate
        if step % 100 == 0:
            lr *= 0.95
    
    return best_x
# EVOLVE-BLOCK-END
