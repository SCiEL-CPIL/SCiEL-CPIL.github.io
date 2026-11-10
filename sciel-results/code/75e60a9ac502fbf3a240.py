# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog
from scipy.sparse import csr_matrix

def _build_constraint_matrix(K, x_max):
    """
    Build sparse constraint matrix A where A[x-1, k-1] = floor(x/k) for k <= x, else 0.
    x ranges from 1 to x_max, k ranges from 1 to K.
    """
    rows = []
    cols = []
    data = []
    
    for x in range(1, x_max + 1):
        max_k = min(x, K)
        for k in range(1, max_k + 1):
            val = x // k
            if val > 0:
                rows.append(x - 1)
                cols.append(k - 1)
                data.append(val)
    
    A = csr_matrix((data, (rows, cols)), shape=(x_max, K), dtype=np.float64)
    return A

def _solve_lp(K, time_budget):
    """
    Solve the LP:
      maximize -sum_{k=1}^{K} v_k * ln(k)/k
      subject to:
        -10 <= v_k <= 10  for all k
        v_1 = 0  (handled by bounds)
        M(x) = sum_{k=1}^{K} v_k * floor(x/k) <= 1  for x = 1..10K
    """
    start = time.time()
    x_max = 10 * K
    
    # Objective: maximize -sum v_k * ln(k)/k, so c = -ln(k)/k
    k_range = np.arange(1, K + 1, dtype=np.float64)
    c = -np.log(k_range) / k_range
    
    # Bounds: v_1 = 0, v_2..v_K in [-10, 10]
    bounds = [(0, 0)] + [(-10, 10)] * (K - 1)
    
    # Build constraint matrix
    if time.time() - start > time_budget * 0.3:
        return None, time.time() - start
    
    A = _build_constraint_matrix(K, x_max)
    b = np.ones(x_max, dtype=np.float64)
    
    # Solve
    try:
        res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs',
                      options={'time_limit': max(1, time_budget - (time.time() - start))})
        if res.success:
            return res.x, time.time() - start
    except Exception:
        pass
    
    return None, time.time() - start

def _compute_M_max(v, K, x_max):
    """
    Compute max of M(x) = sum_k v[k] * floor(x/k) for x in [1, x_max].
    Uses efficient computation.
    """
    x_vals = np.arange(1, x_max + 1, dtype=np.int64)
    M = np.zeros(x_max + 1, dtype=np.float64)
    
    for k in range(1, K + 1):
        if abs(v[k - 1]) > 1e-12:
            M[1:] += v[k - 1] * (x_vals // k)
    
    return np.max(M[1:])

def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    """
    start_time = time.time()
    total_budget = 600.0
    safety_margin = 30.0
    deadline = start_time + total_budget - safety_margin
    
    best_domain = None
    best_values = None
    best_obj = -np.inf
    
    # Try K values in order of increasing size
    candidate_Ks = [100, 200, 300, 500, 700, 1000]
    
    for K in candidate_Ks:
        if time.time() > deadline:
            break
        
        time_left = deadline - time.time()
        if time_left < 15:
            break
        
        # Solve LP
        v, elapsed = _solve_lp(K, time_left)
        
        if v is None:
            continue
        
        # Clip values
        v = np.clip(v, -10, 10)
        
        # Verify feasibility: M(x) <= 1.0001 for x in [1, 10K]
        x_max = 10 * K
        if time.time() > deadline - 10:
            # Not enough time to verify, skip
            continue
        
        max_M = _compute_M_max(v, K, x_max)
        
        if max_M > 1.0001:
            continue
        
        # Calculate objective a = -sum f(k) log(k) / k for k >= 2
        # (f(1) term vanishes since log(1) = 0)
        mask = np.abs(v) > 1e-9
        domain = np.where(mask)[0] + 1
        values = v[mask]
        
        if len(domain) == 0:
            continue
        
        a = -np.sum(values * np.log(domain) / domain)
        
        if a > best_obj:
            best_obj = a
            best_domain = domain.copy()
            best_values = values.copy()
    
    # If no solution found, return baseline
    if best_domain is None:
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
    else:
        domain = best_domain
        values = best_values
        
    return domain, values

# EVOLVE-BLOCK-END
