# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes the objective a = -sum_k f(k)*log(k)/k subject to M(x) <= 1.0001
    for all x in [1, 10*K], where M(x) = sum_k f(k)*floor(x/k).
    
    Uses LP formulation with scipy.optimize.linprog (HiGHS backend).
    """
    start_time = time.time()
    deadline = start_time + 600  # 600s hard limit
    
    # We'll try increasing K values until we run out of time
    # Start with a small K that we know works, then try larger
    
    # First, let's compute the baseline score
    baseline_domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
    baseline_values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
    best_domain = baseline_domain.copy()
    best_values = baseline_values.copy()
    best_score = 1.0  # The seed score
    
    # Try different K values
    # The key insight: M(x) = sum_{k|x} f(k) ... no wait, M(x) = sum_k f(k)*floor(x/k)
    # For integer x, floor(x/k) = number of multiples of k up to x
    # So M(x) = sum_{j=1}^{x} (sum_{k|j} f(k)) = sum_{j=1}^{x} (f * mu)(j) ... 
    # Actually M(x) = sum_k f(k) * floor(x/k) = sum_{j=1}^{x} sum_{k|j} f(k)
    # Let g(j) = sum_{k|j} f(k), then M(x) = sum_{j=1}^{x} g(j)
    # We need M(x) <= 1 for all x in [1, 10K]
    # This means the partial sums of g must be <= 1
    
    # Alternative: M(x) for x = 1, 2, ..., 10K
    # M(x) = sum_k f(k) * floor(x/k)
    # floor(x/k) is nonzero only when k <= x, so for x in [1, 10K], k in [1, min(K, x)]
    
    # Let's set up the LP:
    # Variables: f(1), f(2), ..., f(K)
    # Maximize: -sum_k f(k) * log(k) / k
    # Subject to: M(x) = sum_k f(k) * floor(x/k) <= 1 for x = 1, 2, ..., 10K
    # Also: f(k) is clipped to [-10, 10], but we can just add bounds
    
    # Actually, the clipping happens BEFORE the f(1) adjustment. Let me re-read.
    # "values are clipped to [-10, 10]. Then f(1) is replaced by f(1) - S with S = sum_k f(k)/k"
    # So the actual f used in M(x) and the objective is after this adjustment.
    # But we can choose any values, they get clipped to [-10,10] first.
    # To simplify, let's assume our values are in [-10, 10] so clipping doesn't change them.
    
    # After clipping, let f_clipped(k) be the clipped value.
    # S = sum_k f_clipped(k) / k
    # f_adjusted(1) = f_clipped(1) - S
    # f_adjusted(k) = f_clipped(k) for k > 1
    
    # M(x) = sum_k f_adjusted(k) * floor(x/k)
    #       = f_adjusted(1) * floor(x/1) + sum_{k=2}^{K} f_clipped(k) * floor(x/k)
    #       = (f_clipped(1) - S) * x + sum_{k=2}^{K} f_clipped(k) * floor(x/k)
    #       = f_clipped(1) * x - S * x + sum_{k=2}^{K} f_clipped(k) * floor(x/k)
    
    # S = f_clipped(1)/1 + sum_{k=2}^{K} f_clipped(k)/k
    
    # This is getting complex. Let me just solve the LP directly with the original f values,
    # then apply the adjustment in the evaluation check.
    
    # Actually, let me think about this differently. The evaluator:
    # 1. Takes our (domain, values)
    # 2. Clips values to [-10, 10]
    # 3. Computes S = sum f(k)/k over clipped values
    # 4. Sets f(1) = f(1) - S (if 1 in domain, else f(1)=0-S)
    # 5. Checks M(x) = sum f(k)*floor(x/k) <= 1.0001 for x in [1, 10K]
    # 6. Computes a = -sum f(k)*log(k)/k
    
    # So we need to account for the f(1) adjustment.
    # Let's define: for k in domain, let v_k be the clipped value.
    # S = sum_{k in domain} v_k / k
    # f_adj(1) = v_1 - S if 1 in domain, else -S
    # f_adj(k) = v_k for k != 1
    
    # M(x) = f_adj(1) * x + sum_{k in domain, k>1} v_k * floor(x/k)
    #      = (v_1 - S)*x + sum_{k>1} v_k * floor(x/k)  [if 1 in domain]
    #      = -S*x + sum_{k>1} v_k * floor(x/k)  [if 1 not in domain]
    
    # To maximize a = -sum f_adj(k) * log(k) / k
    # = -f_adj(1)*log(1)/1 - sum_{k>1} v_k * log(k)/k
    # = -sum_{k>1} v_k * log(k)/k  [since log(1)=0]
    
    # So the objective only depends on v_k for k > 1!
    # And the constraint is:
    # (v_1 - S)*x + sum_{k>1} v_k * floor(x/k) <= 1.0001 for all x in [1, 10K]
    
    # S = v_1 + sum_{k>1} v_k/k
    
    # Let's include k=1 in domain for flexibility.
    
    # For a given K, set up LP:
    # Variables: v_1, v_2, ..., v_K (all in [-10, 10])
    # Objective: maximize -sum_{k=2}^{K} v_k * log(k) / k
    #   (v_1 doesn't appear in objective since log(1)=0)
    # Constraints: for x = 1, 2, ..., 10K:
    #   (v_1 - (v_1 + sum_{k=2}^K v_k/k)) * x + sum_{k=2}^K v_k * floor(x/k) <= 1.0001
    #   Simplify: -sum_{k=2}^K v_k/k * x + sum_{k=2}^K v_k * floor(x/k) <= 1.0001
    #   sum_{k=2}^K v_k * (floor(x/k) - x/k) <= 1.0001
    
    # Interesting! v_1 cancels out in the constraint!
    # So v_1 is free (within [-10,10]) and doesn't affect feasibility or objective.
    # We can just set v_1 = 0 for simplicity.
    
    # So the LP is:
    # Variables: v_2, ..., v_K in [-10, 10]
    # Maximize: -sum_{k=2}^K v_k * log(k) / k
    # Subject to: for x = 1, ..., 10K:
    #   sum_{k=2}^K v_k * (floor(x/k) - x/k) <= 1.0001
    
    # Note: floor(x/k) - x/k is in [-1, 0], so these are "negative" coefficients.
    # Let w_k = -v_k for k >= 2. Then:
    # Minimize: sum_{k=2}^K w_k * log(k) / k
    # Subject to: for x = 1, ..., 10K:
    #   -sum_{k=2}^K w_k * (floor(x/k) - x/k) <= 1.0001
    #   i.e., sum_{k=2}^K w_k * (x/k - floor(x/k)) <= 1.0001
    # Where x/k - floor(x/k) = {x/k} (fractional part) is in [0, 1).
    
    # And w_k in [-10, 10].
    
    # This is a standard LP. Let's solve it.
    
    def solve_lp(K, time_remaining):
        """Solve the LP for given K."""
        from scipy.optimize import linprog
        
        n_vars = K - 1  # variables v_2, ..., v_K
        if n_vars <= 0:
            return None
        
        # Objective: minimize sum_{k=2}^K w_k * log(k) / k where w_k = -v_k
        # c_k = log(k) / k for k = 2, ..., K
        c = np.array([math.log(k) / k for k in range(2, K + 1)])
        
        # Constraints: for x = 1, ..., 10K:
        # sum_{k=2}^K w_k * frac(x/k) <= 1.0001
        # where frac(x/k) = x/k - floor(x/k)
        
        max_x = 10 * K
        n_constraints = max_x
        
        # Build constraint matrix A_ub
        # A_ub[i, j] = frac((i+1) / (j+2)) for i in [0, max_x-1], j in [0, n_vars-1]
        # x = i+1, k = j+2
        
        # This could be large. Let's estimate size.
        # For K=50: n_vars=49, n_constraints=500, matrix is 500x49 = 24500 entries. Fine.
        # For K=100: n_vars=99, n_constraints=1000, matrix is 1000x99 = 99000. Fine.
        # For K=200: n_vars=199, n_constraints=2000, matrix is 2000x199 = 398000. Still fine.
        
        if time_remaining < 5:
            return None
        
        A_ub = np.zeros((n_constraints, n_vars))
        for i in range(n_constraints):
            x = i + 1
            for j in range(n_vars):
                k = j + 2
                A_ub[i, j] = (x / k) - math.floor(x / k)
        
        b_ub = np.full(n_constraints, 1.0001)
        
        # Bounds: w_k in [-10, 10]
        bounds = [(-10, 10)] * n_vars
        
        # Solve
        try:
            result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        except Exception:
            return None
        
        if not result.success:
            return None
        
        w = result.x
        v = -w  # v_k = -w_k for k >= 2
        
        # Construct domain and values
        domain = np.arange(1, K + 1, dtype=np.int64)
        values = np.zeros(K, dtype=np.float64)
        values[0] = 0.0  # v_1 = 0
        values[1:] = v
        
        # Now compute the actual score as the evaluator would
        # Clip values
        values_clipped = np.clip(values, -10, 10)
        
        # Compute S
        S = np.sum(values_clipped / domain.astype(float))
        
        # Adjust f(1)
        f_adj = values_clipped.copy()
        f_adj[0] = f_adj[0] - S  # assuming 1 is in domain
        
        # Check M(x) <= 1.0001 for x in [1, 10K]
        K_max = domain.max()
        max_x_check = 10 * K_max
        feasible = True
        for x in range(1, max_x_check + 1):
            M_x = np.sum(f_adj * np.floor(x / domain.astype(float)))
            if M_x > 1.0001:
                feasible = False
                break
        
        if not feasible:
            return None
        
        # Compute objective
        a = -np.sum(f_adj * np.log(domain.astype(float)) / domain.astype(float))
        
        return domain, values, a
    
    # Try increasing K values
    K_values = [20, 30, 50, 80, 120, 150]
    
    for K in K_values:
        elapsed = time.time() - start_time
        time_remaining = deadline - elapsed - 30  # 30s safety margin
        if time_remaining < 10:
            break
        
        try:
            result = solve_lp(K, time_remaining)
            if result is not None:
                domain, values, a = result
                if a > best_score:
                    best_score = a
                    best_domain = domain.copy()
                    best_values = values.copy()
                    print(f"K={K}: score={a:.6f}")
        except Exception as e:
            print(f"K={K} failed: {e}")
            continue
    
    return best_domain, best_values

# EVOLVE-BLOCK-END
