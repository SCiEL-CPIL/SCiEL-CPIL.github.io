"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

DEADLINE = 120.0
START_TIME = time.time()

def _time_left():
    return DEADLINE - (time.time() - START_TIME)

def _c3_upper_bound_exact(h):
    """
    Compute the exact C3 upper bound for a given step function h.
    Metric: max_k |conv(h, h)[k]| / (sum_i |h_i|)^2
    Uses FFT for efficient convolution.
    """
    if np.all(h == 0):
        return float('inf')
    
    # Linear convolution via FFT
    # Length of h is N. Linear conv length is 2N-1.
    # rfft requires even length for symmetry, but we can just use full FFT or pad to power of 2.
    # n_fft = 2*N is sufficient for linear conv of two length N arrays (since 2N-1 <= 2N).
    n_fft = 2 * N
    H = np.fft.rfft(h, n=n_fft)
    C = np.fft.irfft(H * H, n=n_fft)
    
    # C has length n_fft. The first 2N-1 elements are the valid linear convolution.
    # The last element (index 2N-1) should be 0 (or very close).
    # We take the max absolute value over the valid range.
    # Actually, for linear conv of length N and N, the result has length 2N-1.
    # Indices 0 to 2N-2.
    valid_C = C[:2*N-1]
    
    max_abs_conv = np.max(np.abs(valid_C))
    sum_abs_h = np.sum(np.abs(h))
    
    if sum_abs_h < 1e-12:
        return float('inf')
        
    return max_abs_conv / (sum_abs_h ** 2)

def _objective_soft(x):
    """
    Smooth objective function for optimization.
    x is the unconstrained variable, h = tanh(x).
    """
    h = np.tanh(x)
    
    # Compute convolution
    n_fft = 2 * N
    H = np.fft.rfft(h, n=n_fft)
    C = np.fft.irfft(H * H, n=n_fft)
    valid_C = C[:2*N-1]
    
    # Smooth absolute value
    eps = 1e-10
    abs_C = np.sqrt(valid_C**2 + eps)
    
    # Soft max
    # We want to minimize max(abs_C) / (sum(|h|))^2
    # Let S = sum(|h|). Minimize log(max(abs_C)) - 2 log(S).
    # Using soft max for differentiability.
    T = 5.0
    max_abs_C = np.max(abs_C)
    # Stable log-sum-exp
    shifted = (abs_C - max_abs_C) / T
    log_sum_exp = np.log(np.sum(np.exp(shifted))) + max_abs_C / T
    soft_max = T * log_sum_exp
    
    abs_h = np.sqrt(h**2 + eps)
    sum_abs_h = np.sum(abs_h)
    
    # Objective value (soft approximation of the upper bound)
    val = soft_max / (sum_abs_h**2 + 1e-8)
    return val

def _gradient_soft(x):
    """
    Numerical gradient for the soft objective.
    """
    eps = 1e-5
    grad = np.zeros_like(x)
    f0 = _objective_soft(x)
    for i in range(N):
        x_plus = x.copy()
        x_plus[i] += eps
        grad[i] = (_objective_soft(x_plus) - f0) / eps
    return grad

def _local_search_spin(h):
    """
    Local search on the discrete spin space {+1, -1}.
    Tries to flip individual spins to improve the exact objective.
    """
    h = h.copy()
    # Normalize to +/- 1
    h = np.sign(h)
    if np.any(h == 0):
        h[h == 0] = 1.0
        
    current_val = _c3_upper_bound_exact(h)
    improved = True
    while improved and _time_left() > 5.0:
        improved = False
        for i in range(N):
            if _time_left() < 5.0:
                break
            old_sign = h[i]
            h[i] = -old_sign
            new_val = _c3_upper_bound_exact(h)
            if new_val < current_val - 1e-12:
                current_val = new_val
                improved = True
            else:
                h[i] = old_sign
    return h, current_val

def construct():
    """
    Construct a signed step function with 400 equal intervals on [-1/4, 1/4]
    to minimize the upper bound on C3.
    """
    start_time = time.time()
    
    best_h = None
    best_val = float('inf')
    
    # Strategy 1: Gradient-based optimization from multiple initializations
    # We use L-BFGS-B which has its own line search, so we don't need to provide gradient
    # unless we want to help it. Given N=400, numerical gradient is expensive (400 evals/step).
    # L-BFGS-B uses finite differences by default if jac is not provided.
    # Let's try providing a more efficient gradient or just letting it run.
    # Since we have a 120s limit, let's be careful.
    
    # Initializations:
    # 1. All +1 (Seed)
    # 2. Alternating +1/-1
    # 3. Random
    # 4. Blocky random (low frequency)
    
    inits = []
    
    # 1. All +1
    x0_1 = np.full(N, 5.0)
    inits.append(x0_1)
    
    # 2. Alternating
    x0_2 = np.zeros(N)
    x0_2[::2] = 2.0
    x0_2[1::2] = -2.0
    inits.append(x0_2)
    
    # 3. Random
    np.random.seed(42)
    x0_3 = np.random.randn(N) * 2.0
    inits.append(x0_3)
    
    # 4. Blocky
    block_size = 10
    num_blocks = N // block_size
    blocks = np.random.choice([1, -1], size=num_blocks)
    x0_4 = np.repeat(blocks * 2.0, block_size)
    inits.append(x0_4)
    
    # 5. Another random
    np.random.seed(123)
    x0_5 = np.random.randn(N) * 2.0
    inits.append(x0_5)
    
    for i, x0 in enumerate(inits):
        if _time_left() < 10.0:
            break
            
        try:
            # Use L-BFGS-B
            # maxiter is a soft limit, but we check time in the callback?
            # scipy minimize doesn't have a built-in time limit callback that stops it cleanly.
            # We can limit maxiter based on remaining time.
            # Rough estimate: 1 iter ~ 1-5 ms for N=400 FFT + finite diff (if used).
            # If we don't provide gradient, it uses finite diff.
            # Let's provide a gradient? No, let's just limit iterations.
            # 200 iterations should be safe.
            
            res = minimize(_objective_soft, x0, method='L-BFGS-B', 
                           options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-5})
            
            h_opt = np.tanh(res.x)
            val = _c3_upper_bound_exact(h_opt)
            
            if val < best_val:
                best_val = val
                best_h = h_opt.copy()
                
        except Exception as e:
            pass
    
    # Strategy 2: Discrete local search (Spin glass)
    # Start from the best continuous solution found, or from good discrete inits.
    
    # Discrete init 1: Sign of best continuous
    if best_h is not None:
        h_disc = np.sign(best_h)
        if np.any(h_disc == 0):
            h_disc[h_disc == 0] = 1.0
        h_loc, val_loc = _local_search_spin(h_disc)
        if val_loc < best_val:
            best_val = val_loc
            best_h = h_loc
    
    # Discrete init 2: All +1
    h_disc = np.ones(N)
    h_loc, val_loc = _local_search_spin(h_disc)
    if val_loc < best_val:
        best_val = val_loc
        best_h = h_loc
        
    # Discrete init 3: Alternating
    h_disc = np.array([1 if i % 2 == 0 else -1 for i in range(N)], dtype=np.float64)
    h_loc, val_loc = _local_search_spin(h_disc)
    if val_loc < best_val:
        best_val = val_loc
        best_h = h_loc
        
    # Discrete init 4: Random
    np.random.seed(42)
    h_disc = np.random.choice([1, -1], size=N).astype(np.float64)
    h_loc, val_loc = _local_search_spin(h_disc)
    if val_loc < best_val:
        best_val = val_loc
        best_h = h_loc
        
    # Discrete init 5: Blocky
    np.random.seed(123)
    block_size = 10
    num_blocks = N // block_size
    blocks = np.random.choice([1, -1], size=num_blocks)
    h_disc = np.repeat(blocks, block_size).astype(np.float64)
    h_loc, val_loc = _local_search_spin(h_disc)
    if val_loc < best_val:
        best_val = val_loc
        best_h = h_loc
        
    # If we still have time, do a second pass of local search on the best solution
    # or try a few more random discrete starts
    if _time_left() > 10.0:
        for seed in [7, 8, 9]:
            if _time_left() < 5.0:
                break
            np.random.seed(seed)
            h_disc = np.random.choice([1, -1], size=N).astype(np.float64)
            h_loc, val_loc = _local_search_spin(h_disc)
            if val_loc < best_val:
                best_val = val_loc
                best_h = h_loc
                
    # Final polish: If best_h is continuous, try snapping to discrete and re-optimizing?
    # The task asks for a "signed step function". Usually this implies discrete values like +/-1 or +/-c.
    # However, the seed program returns continuous values from tanh.
    # The evaluator measures the raw value.
    # If the metric is invariant to scaling (which max_conv / sum_abs^2 is, since both scale with h^2),
    # then the magnitude doesn't matter, only the relative signs and ratios.
    # If we are allowed continuous values, the local search on spins might be suboptimal if the optimal
    # solution is not strictly +/-1.
    # But for "signed step function", +/-1 is the standard interpretation.
    # Let's check if a continuous solution is significantly better.
    
    # If best_h is from gradient optimization, let's see if its discrete version is better.
    # Actually, _local_search_spin already checks the discrete version of the best continuous.
    
    # Let's return the best_h found.
    if best_h is None:
        # Fallback to seed
        return np.ones(N)
        
    return best_h

# EVOLVE-BLOCK-END
