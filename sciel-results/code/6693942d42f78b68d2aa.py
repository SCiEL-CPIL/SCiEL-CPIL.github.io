import numpy as np
import time
import cvxpy as cp
from scipy.sparse import csr_matrix

def prime_number_theorem_partial_function():
    start_time = time.time()
    deadline = 580.0  # Safety margin before 600s limit
    
    best_domain = None
    best_values = None
    best_score = -np.inf
    
    # We want to maximize a = -sum_{k=2}^K f(k) log(k)/k
    # Subject to:
    # 1. sum_{k=2}^K f(k)/k = S (f(1) = -S, so sum f(k)/k = 0 holds)
    #    Actually, the evaluator computes S = sum f(k)/k over clipped values.
    #    Then f(1) is replaced by f(1) - S.
    #    Wait, the task says: "f(1) is replaced by f(1) - S ... so that afterwards sum_k f(k)/k = 0".
    #    This implies we can choose f(1) arbitrarily initially? No, we return (domain, values).
    #    The evaluator takes our values, clips them, computes S, adjusts f(1).
    #    So the constraint on our returned values is:
    #    Let g(k) = f(k) for k >= 2.
    #    The evaluator sets f(1) = -sum_{k>=2} g(k)/k.
    #    The metric is a = -sum_{k>=2} g(k) log(k)/k. (f(1) term is 0 because log(1)=0).
    #    
    #    Constraints on M(x):
    #    M(x) = sum_{k=1}^K f(k) floor(x/k) <= 1.0001
    #    M(x) = f(1) * x + sum_{k=2}^K g(k) floor(x/k)
    #    Substitute f(1) = -sum_{j=2}^K g(j)/j:
    #    M(x) = -x * sum_{j=2}^K g(j)/j + sum_{k=2}^K g(k) floor(x/k)
    #    M(x) = sum_{k=2}^K g(k) [ floor(x/k) - x/k ] <= 1.0001
    #    
    #    Let h(x, k) = floor(x/k) - x/k. Note that h(x, k) is in [-1, 0).
    #    Actually, floor(x/k) - x/k = -frac(x/k).
    #    So M(x) = - sum_{k=2}^K g(k) frac(x/k).
    #    Constraint: sum_{k=2}^K g(k) frac(x/k) >= -1.0001.
    #    Since frac(x/k) >= 0, if all g(k) are positive, LHS is positive, constraint is easy.
    #    If g(k) are negative, LHS is negative.
    #    
    #    We want to maximize -sum g(k) log(k)/k.
    #    Let w(k) = log(k)/k. Maximize -sum g(k) w(k).
    #    This suggests we want g(k) to be as negative as possible for large w(k).
    #    w(k) is max around e. w(2) = 0.346, w(3) = 0.366, w(4) = 0.346.
    #    So we want g(3), g(2), g(4) to be negative (close to -10).
    #    
    #    Let's check the constraint for x=1.
    #    frac(1/k) = 1/k for k > 1.
    #    Constraint: sum g(k) (1/k) >= -1.0001.
    #    If g(k) = -10 for all k, sum -10/k is very negative. Violation.
    #    So we are constrained by the sum of g(k)/k.
    #    
    #    Let's set up the LP/CVXPY problem.
    #    Variables: g (vector of length K-1, for k=2..K)
    #    Maximize: -sum g * (log(k)/k)
    #    Subject to:
    #    For each x in 1..10K: sum_{k=2}^K g(k) * (floor(x/k) - x/k) <= 1.0001
    #    -10 <= g(k) <= 10
    
    # We will try increasing K.
    # For each K, we build the sparse matrix A where A[x, k-2] = floor(x/k) - x/k
    # Note: x ranges 1 to 10K. k ranges 2 to K.
    
    K_values = [100, 200, 500, 1000, 2000, 5000]
    
    for K in K_values:
        if time.time() > deadline:
            break
            
        n_vars = K - 1
        if n_vars <= 0:
            continue
            
        ks = np.arange(2, K + 1)
        w = np.log(ks) / ks
        
        # Objective: maximize -sum(g * w)
        # In cvxpy, minimize sum(g * w)
        
        # Build sparse matrix A of size (10K, K-1)
        # A[i, j] corresponds to x = i+1, k = j+2
        # A[i, j] = floor((i+1)/(j+2)) - (i+1)/(j+2)
        
        max_x = 10 * K
        x_vals = np.arange(1, max_x + 1)
        
        # Efficient construction of A
        # We can iterate over k and fill columns, or use broadcasting if memory allows.
        # For K=5000, 10K=50000. 50000 * 5000 = 250M entries. Too big for dense.
        # Sparse is better.
        
        rows = []
        cols = []
        data = []
        
        # Iterate over k (columns)
        for j, k in enumerate(ks):
            # x from 1 to 10K
            # floor(x/k) - x/k
            # For x < k, floor(x/k)=0, val = -x/k
            # For x >= k, floor(x/k) = m, val = m - x/k = -frac(x/k)
            
            # We can compute this vectorized for each k
            x = x_vals
            vals = np.floor(x / k) - x / k
            # vals are in [-1, 0]
            # Only non-zero if x % k != 0
            # Actually, we can just store all, SCS handles zeros fine, but sparsity helps.
            # Let's find non-zeros
            mask = vals != 0
            if np.any(mask):
                idx = np.where(mask)[0]
                rows.extend(idx)
                cols.extend([j] * len(idx))
                data.extend(vals[idx])
                
        if len(data) == 0:
            continue
            
        A = csr_matrix((data, (rows, cols)), shape=(max_x, n_vars))
        
        # Constraint: A @ g <= 1.0001
        # Bounds: -10 <= g <= 10
        
        g = cp.Variable(n_vars)
        
        objective = cp.Minimize(cp.sum(cp.multiply(g, w)))
        
        constraints = [
            A @ g <= 1.0001,
            g >= -10,
            g <= 10
        ]
        
        prob = cp.Problem(objective, constraints)
        
        try:
            # SCS is good for large sparse LPs
            prob.solve(solver=cp.SCS, verbose=False, max_iters=100000)
        except Exception:
            continue
            
        if prob.status in ['optimal', 'optimal_inaccurate']:
            g_val = g.value
            if g_val is None:
                continue
                
            # Check feasibility strictly
            # M(x) = sum g(k) (floor(x/k) - x/k)
            # We want M(x) <= 1.0001
            # Our constraint was A @ g <= 1.0001 where A = floor - x/k.
            # So this is satisfied.
            
            # Calculate score
            # a = -sum g(k) log(k)/k
            # Our objective was min sum g(k) w(k). So a = -objective_value.
            a = -prob.value
            
            # Verify f(1) constraint
            # S = sum g(k)/k
            # f(1) = -S
            # We need f(1) to be within [-10, 10] after clipping?
            # The task says: "values are clipped to [-10, 10]. Then f(1) is replaced..."
            # It doesn't explicitly say f(1) must be in [-10,10] *before* adjustment,
            # but the adjustment makes sum f(k)/k = 0.
            # However, if S is huge, f(1) becomes huge.
            # Does the evaluator clip f(1) *after* adjustment?
            # "values are clipped to [-10, 10]. Then f(1) is replaced by f(1) - S..."
            # This implies:
            # 1. Clip all returned values to [-10, 10].
            # 2. Calculate S from clipped values.
            # 3. Set f(1) = f(1)_clipped - S.
            # 4. Does it clip f(1) again? "so that afterwards sum_k f(k)/k = 0".
            # If f(1) is not clipped again, then f(1) can be outside [-10, 10].
            # But M(x) uses f(1).
            # Let's check the constraint M(x) <= 1.0001.
            # If f(1) is very large negative, M(x) for large x will be very negative, which is fine for <= 1.
            # If f(1) is very large positive, M(x) might violate.
            # Our LP constraint accounted for f(1) = -sum g/k.
            # So the LP ensures M(x) <= 1.0001 assuming f(1) is set to -S.
            # The only risk is if the evaluator clips f(1) *after* the adjustment?
            # "values are clipped to [-10, 10]. Then f(1) is replaced..."
            # Usually "clipped" applies to the initial array.
            # If f(1) is not in domain, f(1)=0.
            # Let's assume the standard interpretation:
            # 1. We return domain, values.
            # 2. Evaluator clips values to [-10, 10].
            # 3. Evaluator computes S = sum f_clipped(k)/k.
            # 4. Evaluator sets f_final(1) = f_clipped(1) - S.
            # 5. Evaluator checks M(x) using f_final.
            #    
            # If we include 1 in domain, we should return f(1) such that it helps?
            # No, f(1) is overwritten.
            # So we should probably NOT include 1 in the domain, or set f(1)=0.
            # If 1 is not in domain, f(1)=0 initially.
            # Then f_final(1) = 0 - S = -S.
            # This matches our LP assumption f(1) = -S.
            #    
            # So, we should return domain 2..K, and values g.
            # And ensure S = sum g/k is not so large that it causes issues?
            # The LP constraints on M(x) already handle the effect of f(1)=-S.
            # So if LP is feasible, M(x) <= 1.0001 is satisfied.
            
            # Score a
            if a > best_score:
                best_score = a
                best_domain = ks.astype(np.int64)
                # Clip g to [-10, 10] just in case, though LP bounds enforce it
                best_values = np.clip(g_val, -10, 10)
                
                # Optional: Add 1 to domain with value 0?
                # If we add 1, f(1)_clipped = 0. S is same. f_final(1) = -S.
                # Same result.
                # Let's stick to k>=2 to be safe and minimal.
                
                # Check time
                if time.time() > deadline - 10:
                    break

    if best_domain is not None:
        return best_domain, best_values
    else:
        # Fallback
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values

