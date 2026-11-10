# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog

def prime_number_theorem_partial_function():
    """
    Construct a partial function f to maximize the objective using linear programming.
    Variables: f(k) for k=1..K
    Constraints:
      1. sum_k f(k)/k = 0  (after clipping and f(1) adjustment, but we enforce sum f(k)/k = 0 directly)
      2. M(x) = sum_k f(k) * floor(x/k) <= 1.0001 for all x in [1, 10K]
      3. -10 <= f(k) <= 10 (clipping bounds)
    Objective: maximize a = -sum_k f(k) * log(k) / k
    """
    start_time = time.time()
    deadline = 600.0
    
    # Choose K. Smaller K means smaller LP, faster solve.
    # K=5000 is a good balance.
    K = 5000
    
    # 1. Compute Mobius function up to K for initialization
    mu = np.ones(K + 1, dtype=np.float64)
    is_prime = np.ones(K + 1, dtype=bool)
    mu[0] = 0
    for i in range(2, K + 1):
        if is_prime[i]:
            for j in range(i, K + 1, i):
                is_prime[j] = False
                mu[j] *= -1
            i2 = i * i
            if i2 <= K:
                for j in range(i2, K + 1, i2):
                    mu[j] = 0
    
    max_x = 10 * K
    
    # Objective: maximize -sum_k f(k) * log(k) / k
    # In linprog, we minimize c^T x, so c = sum_k f(k) * log(k) / k
    # log(1) = 0, so the first component is 0
    log_k = np.zeros(K, dtype=np.float64)
    for k in range(2, K + 1):
        log_k[k-1] = np.log(k)
    c = log_k / np.arange(1, K + 1, dtype=np.float64)
    
    # Inequality constraints: M(x) = sum_k f(k) * floor(x/k) <= 1.0001
    # This is A_ub @ f <= b_ub
    # A_ub has shape (max_x, K), A_ub[i, k] = floor((i+1) / (k+1))
    # But max_x = 10*K = 50000, K = 5000, so A_ub is 50000x5000 = 250 million entries
    # That's too large for dense. We need a smarter approach.
    
    # Alternative: Use the fact that M(x) = sum_{n=1}^x A(n) where A(n) = sum_{d|n} f(d)
    # We can express the constraints more compactly.
    # Actually, let's just use K=1000 to keep the LP manageable.
    
    K = 1000
    max_x = 10 * K
    
    log_k = np.zeros(K, dtype=np.float64)
    for k in range(2, K + 1):
        log_k[k-1] = np.log(k)
    c = log_k / np.arange(1, K + 1, dtype=np.float64)
    
    # Build A_ub: A_ub[i, k] = floor((i+1) / (k+1)) for i=0..max_x-1, k=0..K-1
    # This is a 10000 x 1000 matrix = 10 million entries, manageable.
    x_indices = np.arange(1, max_x + 1, dtype=np.float64)
    k_indices = np.arange(1, K + 1, dtype=np.float64)
    A_ub = np.floor(np.outer(x_indices, 1.0 / k_indices))
    
    b_ub = np.full(max_x, 1.0001)
    
    # Equality constraint: sum_k f(k)/k = 0
    A_eq = np.zeros((1, K), dtype=np.float64)
    A_eq[0] = 1.0 / np.arange(1, K + 1, dtype=np.float64)
    b_eq = np.zeros(1)
    
    # Bounds: -10 <= f(k) <= 10
    bounds = [(-10, 10)] * K
    
    # Solve LP
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        if result.success:
            f_vals = result.x
        else:
            # Fallback to scaled Mobius
            f_vals = mu[1:K+1].copy()
            S = np.sum(f_vals[1:] / np.arange(2, K+1, dtype=np.float64))
            f_vals[0] = -S
    except Exception:
        # Fallback to scaled Mobius
        f_vals = mu[1:K+1].copy()
        S = np.sum(f_vals[1:] / np.arange(2, K+1, dtype=np.float64))
        f_vals[0] = -S
    
    # Clip to [-10, 10]
    f_vals = np.clip(f_vals, -10, 10)
    
    # Recompute S after clipping and adjust f(1)
    S = np.sum(f_vals / np.arange(1, K + 1, dtype=np.float64))
    f_vals[0] = f_vals[0] - S
    
    # Re-clip after adjustment
    f_vals = np.clip(f_vals, -10, 10)
    
    # Verify feasibility
    A_check = np.zeros(max_x + 1, dtype=np.float64)
    for d in range(1, K + 1):
        if f_vals[d-1] != 0:
            A_check[d:max_x+1:d] += f_vals[d-1]
    M_check = np.cumsum(A_check)
    
    if np.max(M_check) > 1.0001:
        # Scale down to satisfy constraint
        scale = 1.0001 / np.max(M_check) * 0.999
        f_vals *= scale
        # Re-adjust f(1)
        S = np.sum(f_vals / np.arange(1, K + 1, dtype=np.float64))
        f_vals[0] = f_vals[0] - S
        f_vals = np.clip(f_vals, -10, 10)
    
    domain = np.arange(1, K + 1, dtype=np.int64)
    values = f_vals.astype(np.float64)
    
    return domain, values

# EVOLVE-BLOCK-END
