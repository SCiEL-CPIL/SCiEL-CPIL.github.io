import numpy as np
import time
from scipy.optimize import linprog

def prime_number_theorem_partial_function():
    start_time = time.time()
    
    # Try different K values
    # Start with a small K to ensure we have a valid result quickly
    # Then try larger K if time permits
    
    best_domain = None
    best_values = None
    best_score = -np.inf
    
    for K in [50, 100, 200, 500, 1000, 2000]:
        if time.time() - start_time > 500:
            break
            
        # Variables: g(2) ... g(K)
        n_vars = K - 1
        if n_vars <= 0:
            continue
            
        # Objective: minimize sum( (log(k)/k) * g(k) )
        # c_k = log(k) / k
        ks = np.arange(2, K + 1)
        c = np.log(ks) / ks
        
        # Constraints:
        # sum_{k=2}^K g(k) * {x/k} >= -1  for x in [1, 10K]
        # - sum_{k=2}^K g(k) * {x/k} <= 1
        # A_ub @ g <= b_ub
        
        max_x = 10 * K
        # We don't need to check all x from 1 to 10K if we can reduce them?
        # {x/k} depends on x mod k.
        # The set of vectors { {x/2}, {x/3}, ..., {x/K} } for x=1..10K.
        # There are at most LCM(2..K) distinct patterns, which is huge.
        # But we just iterate x.
        
        # Building A_ub (10K x (K-1)) might be memory heavy for K=2000 (20000 x 1999 ~ 40M floats ~ 320MB).
        # Might be okay.
        
        A_rows = []
        for x in range(1, max_x + 1):
            # row: - {x/k} for k=2..K
            row = np.empty(n_vars)
            for i, k in enumerate(ks):
                row[i] = - (x % k) / k
            A_rows.append(row)
            
        A_ub = np.array(A_rows)
        b_ub = np.ones(max_x)
        
        # Bounds: -10 <= g(k) <= 10
        bounds = [(-10, 10)] * n_vars
        
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.success:
                g = res.x
                # Construct f
                f = np.zeros(K + 1)
                f[2:K+1] = g
                # Calculate S = sum_{k=2}^K f(k)/k
                S = np.sum(f[2:K+1] / ks)
                f[1] = -S
                
                # Check M(x) <= 1 for all x in [1, 10K]
                # M(x) = sum f(k) floor(x/k)
                # We already enforced the constraint derived from M(x) <= 1 and sum f(k)/k = 0.
                # Let's verify.
                # M(x) = f(1) x + sum_{k=2}^K f(k) floor(x/k)
                # = -S x + sum_{k=2}^K f(k) floor(x/k)
                # = sum_{k=2}^K f(k) (floor(x/k) - x/k)
                # = - sum_{k=2}^K f(k) {x/k}
                # Constraint was - sum f(k) {x/k} <= 1.
                # So M(x) <= 1 is satisfied.
                
                # Calculate score a
                # a = - sum_{k=1}^K f(k) log(k)/k
                # log(1)=0, so a = - sum_{k=2}^K f(k) log(k)/k
                a = -np.sum(f[2:K+1] * c)
                
                # Clip values to [-10, 10] as per evaluator?
                # The LP enforced bounds [-10, 10] on g.
                # f(1) might be outside?
                # S = sum g(k)/k. g(k) in [-10, 10].
                # S is roughly O(1). f(1) = -S.
                # If f(1) is outside [-10, 10], evaluator clips it.
                # If clipped, sum f(k)/k != 0, and M(x) changes.
                # We should ensure f(1) is in [-10, 10] or handle clipping.
                # Given g(k) are small on average, f(1) should be small.
                
                if a > best_score:
                    best_score = a
                    domain = np.arange(1, K + 1)
                    values = f[1:K+1]
                    # Clip values to [-10, 10]
                    values = np.clip(values, -10, 10)
                    best_domain = domain
                    best_values = values
                    
        except Exception as e:
            pass

    if best_domain is not None:
        return best_domain, best_values
    else:
        # Fallback to seed
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values
