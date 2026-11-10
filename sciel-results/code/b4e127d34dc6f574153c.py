# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import linprog
import scipy.sparse as sp

def prime_number_theorem_partial_function():
    """
    Maximizes a = -sum f(k) log(k)/k subject to M(x) <= 1.0001.
    Uses LP formulation:
    Maximize sum_{k=2}^K w_k * log(k)/k
    Subject to sum_{k=2}^K w_k * {x/k} <= 1.0001 for x in [1, 10K]
    and 0 <= w_k <= 10.
    Then set v_k = -w_k for k>1, v_1=0.
    """
    start_time = time.time()
    deadline = start_time + 580  # 20s safety margin
    
    best_domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
    best_values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
    best_score = 1.0  # Seed score

    def evaluate(domain, values):
        """
        Compute the actual score as the evaluator would.
        Returns (score, feasible).
        """
        if len(domain) == 0:
            return 0.0, False
        
        # Clip values
        values_clipped = np.clip(values, -10, 10)
        
        # Compute S
        S = np.sum(values_clipped / domain.astype(float))
        
        # Adjust f(1)
        # Find index of 1 in domain
        idx_1 = np.where(domain == 1)[0]
        f_adj = values_clipped.copy()
        if len(idx_1) > 0:
            f_adj[idx_1[0]] = f_adj[idx_1[0]] - S
        else:
            # If 1 not in domain, f(1) is effectively 0 - S = -S
            # But the sum is over k in domain.
            # The evaluator says: "f(1) is replaced by f(1) - S ... (f(1)=0 before this if 1 is not in the domain)"
            # So if 1 is not in domain, we add 1 to domain with value -S?
            # No, the domain is fixed by our return.
            # "values holds the finite reals f(k)".
            # "f(1) is replaced by ...". This implies 1 is in the domain or we treat it as such.
            # The contract says: "domain holds distinct positive integers".
            # If 1 is not in domain, does the evaluator add it?
            # "f(1) = 0 before this if 1 is not in the domain".
            # This implies the set of k for the sum includes 1 implicitly?
            # Let's assume we should always include 1 in the domain to be safe, 
            # or the evaluator handles it by adding it.
            # To be safe, let's always include 1 in our domain.
            pass
            
        K_max = domain.max()
        max_x_check = 10 * K_max
        
        # Check M(x)
        # M(x) = sum_k f_adj(k) * floor(x/k)
        # We can compute this for all x in [1, 10K]
        # Vectorized approach:
        # x_vals = np.arange(1, max_x_check + 1)
        # M_vals = np.zeros(max_x_check)
        # for i, k in enumerate(domain):
        #     M_vals += f_adj[i] * np.floor(x_vals / k)
        
        # Better:
        x_vals = np.arange(1, max_x_check + 1)
        M_max = 0.0
        for i in range(len(domain)):
            k = domain[i]
            if f_adj[i] != 0:
                M_vals = f_adj[i] * np.floor(x_vals / k)
                if np.max(M_vals) > 1.0001:
                    # Optimization: if any single term makes it too big? No, they sum.
                    # But we need the sum.
                    pass
        
        # Full calculation
        M_sum = np.zeros(max_x_check)
        for i in range(len(domain)):
            k = domain[i]
            if f_adj[i] != 0:
                M_sum += f_adj[i] * np.floor(x_vals / k)
        
        if np.max(M_sum) > 1.0001:
            return 0.0, False
        
        # Compute objective
        # a = -sum f_adj(k) * log(k) / k
        # Note: log(1)=0, so term for k=1 is 0.
        a = -np.sum(f_adj * np.log(domain.astype(float)) / domain.astype(float))
        
        return a, True

    # Try increasing K
    # K=1 is trivial (score 0).
    # Start with K=20, go up to maybe 500 or 1000.
    K_values = [20, 50, 100, 200, 400, 800, 1500]
    
    for K in K_values:
        elapsed = time.time() - start_time
        if elapsed > deadline - 10:
            break
            
        n_vars = K - 1
        if n_vars <= 0:
            continue
        
        # Objective: maximize sum w_k * log(k)/k
        # linprog minimizes, so minimize -sum w_k * log(k)/k
        c = -np.array([math.log(k) / k for k in range(2, K + 1)])
        
        # Constraints: sum w_k * {x/k} <= 1.0001 for x in 1..10K
        max_x = 10 * K
        n_constraints = max_x
        
        # Build A_ub
        # A_ub[i, j] = { (i+1) / (j+2) }
        # x = i+1, k = j+2
        
        # Use vectorized construction
        x_col = np.arange(1, max_x + 1, dtype=np.float64)[:, None]
        k_row = np.arange(2, K + 1, dtype=np.float64)[None, :]
        
        # Fractional part
        A_ub = (x_col / k_row) - np.floor(x_col / k_row)
        
        b_ub = np.full(n_constraints, 1.0001)
        
        # Bounds: w_k in [0, 10]
        bounds = [(0, 10)] * n_vars
        
        try:
            result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        except Exception as e:
            # print(f"LP failed for K={K}: {e}")
            continue
            
        if not result.success:
            continue
            
        w = result.x
        # w_k >= 0
        
        # Construct domain and values
        domain = np.arange(1, K + 1, dtype=np.int64)
        values = np.zeros(K, dtype=np.float64)
        values[0] = 0.0 # v_1
        values[1:] = -w # v_k = -w_k
        
        score, feasible = evaluate(domain, values)
        
        if feasible and score > best_score:
            best_score = score
            best_domain = domain.copy()
            best_values = values.copy()
            # print(f"K={K}: score={score:.6f}")

    return best_domain, best_values

# EVOLVE-BLOCK-END
