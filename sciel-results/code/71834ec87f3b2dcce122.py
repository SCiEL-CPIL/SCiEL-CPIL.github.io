# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog
from scipy.sparse import lil_matrix

def _solve_lp(K, time_budget):
    """
    Solve the LP:
      maximize -sum_{k=2}^{K} v_k * ln(k)/k
      subject to:
        -10 <= v_k <= 10  for all k
        M(x) = sum_{k=2}^{K} v_k * floor(x/k) <= 1  for x = 1..K
    Returns (v_array, elapsed) where v_array[1..K] are the values.
    """
    start = time.time()
    n_vars = K  # v_1..v_K, but we fix v_1 = 0 in the LP and handle it separately
    # Actually, let's just solve for v_2..v_K to save a variable.
    # Let's use v_1..v_K all as variables for simplicity in indexing, but fix v_1=0.
    
    # Objective: c = -ln(k)/k for k=1..K
    k_range = np.arange(1, K+1, dtype=np.float64)
    c = -np.log(k_range) / k_range
    
    # Bounds
    # v_1 = 0, v_2..v_K in [-10, 10]
    bounds = [(0, 0)]
    for _ in range(K - 1):
        bounds.append((-10, 10))
    
    # Constraints: M(x) <= 1 for x = 1..K
    # A_ub * v <= b_ub
    # A_ub[i, j] = floor((i+1) / (j+1))
    # Using sparse matrix construction
    A = lil_matrix((K, K), dtype=np.float64)
    for x in range(1, K + 1):
        if time.time() - start > time_budget - 10:
            return None, time.time() - start
        row = x - 1
        for k in range(1, K + 1):
            if k > x:
                break
            A[row, k - 1] = int(x // k)
    A = A.tocsr()
    b = np.ones(K, dtype=np.float64)
    
    # Solve
    try:
        res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')
        if res.success:
            return res.x, time.time() - start
    except Exception:
        pass
    
    return None, time.time() - start

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
    best_score = -np.inf
    best_obj = -np.inf
    
    # We know from knowledge that K=1000 gives 1.075.
    # Let's try to solve for K=1000 first, then maybe larger if time permits?
    # But linprog with K=1000 might be slow if not careful.
    # Let's start with a smaller K to get a quick result, then try larger.
    
    candidate_Ks = [50, 100, 200, 500, 1000]
    
    for K in candidate_Ks:
        if time.time() > deadline:
            break
        
        # Allocate remaining time minus some overhead
        time_left = deadline - time.time()
        if time_left < 10:
            break
        
        # Solve LP
        v, elapsed = _solve_lp(K, time_left)
        
        if v is not None:
            # Evaluate the objective and feasibility
            # v contains v_1..v_K. v_1 should be 0 from bounds.
            # Clip values to [-10, 10] (though LP bounds enforce this, floating point might be slightly off)
            v = np.clip(v, -10, 10)
            
            # Construct domain and values (excluding k=1 if v_1 is 0, or including it?
            # The task says domain holds distinct positive integers.
            # If v_1 is 0, we can exclude it or include it.
            # The evaluator replaces f(1) with f(1) - S.
            # If 1 is not in domain, f(1) is treated as 0.
            # It's cleaner to include 1 if it's non-zero, or just let the evaluator handle it.
            # Let's include all k where v_k != 0.
            
            mask = np.abs(v) > 1e-9
            domain = np.where(mask)[0] + 1
            values = v[mask]
            
            if len(domain) == 0:
                continue
            
            # Check feasibility: M(x) <= 1.0001 for x in [1, 10K]
            # M(x) = sum_k v_k * floor(x/k)
            # We only computed constraints up to x=K in the LP.
            # We must verify up to 10K.
            feasible = True
            max_M = 0.0
            # Efficient computation of M(x) for x=1..10K
            # M(x) = sum_{k=1}^K v_k floor(x/k)
            # For large K, this is O(K * 10K) which is too slow.
            # We can use the fact that floor(x/k) is piecewise constant.
            # Or just check a few points? No, must be robust.
            # Let's use a difference array or similar?
            # Actually, M(x) is non-decreasing if v_k >= 0? No, v_k can be negative.
            # Let's just compute it for x in [1, 10K] but optimized.
            # Wait, 10K * K = 10 * 10^6 = 10^7 operations. That's fast in numpy.
            
            X_max = 10 * K
            # We can vectorize the computation of M(x)
            # But doing it for all x is memory heavy if X_max is large.
            # Let's just loop over x? 10K iterations, each O(K) -> 10^7 ops.
            # In Python this is slow.
            # Use numpy:
            # M(x) = sum_{k=1}^K v_k * floor(x/k)
            # We can compute this by iterating k and adding v_k * floor(x/k) to M.
            # M = np.zeros(X_max + 1)
            # for k in range(1, K+1):
            #   if v[k] != 0:
            #     M[:X_max+1] += v[k] * (np.arange(1, X_max+1) // k)
            
            # This is O(K * X_max) = O(10 K^2). For K=1000, 10^7.
            # Let's try it.
            
            M = np.zeros(X_max + 1, dtype=np.float64)
            ks = np.arange(1, K + 1, dtype=np.int64)
            vs = v
            # Vectorized outer operation?
            # M[x] = sum_k v[k] * floor((x+1)/k)
            # Let's do it in chunks or just use a loop with numpy slicing.
            # Actually, a simple loop over k is fine if K is small.
            for idx, k in enumerate(ks):
                if vs[idx] != 0:
                    # floor((1..X_max) / k)
                    # We can use integer division
                    # np.arange(1, X_max+1) // k
                    # This creates an array of size X_max.
                    # Doing this K times is O(K * X_max).
                    pass
            
            # A faster way:
            # M[x] = sum_{k|x?} no.
            # Let's just use the loop with numpy vectors.
            x_vals = np.arange(1, X_max + 1, dtype=np.int64)
            for idx, k in enumerate(ks):
                if abs(vs[idx]) > 1e-12:
                    M[1:] += vs[idx] * (x_vals // k)
            
            if np.max(M[1:]) > 1.0001:
                feasible = False
            
            if feasible:
                # Calculate objective a = -sum f(k) log(k) / k
                # Note: The evaluator clips values to [-10, 10]. We did that.
                # The evaluator calculates S = sum f(k)/k and subtracts it from f(1).
                # Then a = -sum f'(k) log(k) / k.
                # Since f'(k) = f(k) for k != 1, and f'(1) = f(1) - S.
                # a = -f'(1)*log(1)/1 - sum_{k>1} f(k) log(k)/k
                # log(1) = 0, so the f'(1) term vanishes!
                # So a = -sum_{k>1} f(k) log(k)/k.
                # This is exactly what the LP objective maximizes.
                
                a = -np.sum(values * np.log(domain) / domain)
                
                # Update best
                if a > best_obj:
                    best_obj = a
                    best_domain = domain
                    best_values = values
    
    # If no solution found, return baseline
    if best_domain is None:
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
    else:
        domain = best_domain
        values = best_values
        
    return domain, values

# EVOLVE-BLOCK-END
