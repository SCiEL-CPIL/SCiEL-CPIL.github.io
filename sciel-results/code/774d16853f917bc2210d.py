# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog

def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    """
    start_time = time.time()
    deadline = start_time + 580.0  # 20s safety margin

    K_max = 200000

    def compute_mobius(K):
        """Compute Mobius function mu(k) for k <= K using linear sieve."""
        mu = np.ones(K + 1, dtype=np.int64)
        primes = []
        is_composite = np.zeros(K + 1, dtype=bool)
        mu[1] = 1
        for i in range(2, K + 1):
            if not is_composite[i]:
                primes.append(i)
                mu[i] = -1
            for p in primes:
                if i * p > K:
                    break
                is_composite[i * p] = True
                if i % p == 0:
                    mu[i * p] = 0
                    break
                else:
                    mu[i * p] = -mu[i]
        return mu

    def compute_M_max(f, K):
        """
        Compute max M(x) for x in [1, 10*K] where M(x) = sum_k f(k) * floor(x/k).
        Returns the maximum value of M(x).
        """
        max_x = 10 * K
        g = np.zeros(max_x + 1, dtype=np.float64)
        
        for d in range(1, K + 1):
            val = f[d]
            if val != 0:
                g[d:max_x+1:d] += val
        
        M = np.cumsum(g[1:max_x+1])
        return np.max(M) if len(M) > 0 else 0.0

    def try_mobius_approach(K):
        """
        Try the Mobius function approach for domain [1, K].
        The evaluator will adjust f(1) so that sum f(k)/k = 0.
        We need to ensure M(x) <= 1.0001 after this adjustment.
        
        After adjustment: f'(1) = f(1) - S where S = sum_k f(k)/k
        M(x) = f'(1)*floor(x/1) + sum_{k=2}^K f(k)*floor(x/k)
             = f(1)*floor(x/1) - S*floor(x/1) + sum_{k=2}^K f(k)*floor(x/k)
             = sum_{k=1}^K f(k)*floor(x/k) - S*floor(x/1)
             = M_raw(x) - S*x  (since floor(x/1) = x)
        
        So we need: M_raw(x) - S*x <= 1.0001 for all x in [1, 10K]
        """
        mu = compute_mobius(K)
        f = mu[1:K+1].astype(np.float64)
        
        # S = sum_k f(k)/k
        S = np.sum(f / np.arange(1, K+1))
        
        # M_raw(x) = sum_{k=1}^K mu(k) * floor(x/k)
        # For x <= K: M_raw(x) = 1 (by Mobius inversion identity)
        # For x > K: M_raw(x) = 1 + sum_{j=K+1}^x g(j) where g(j) = sum_{d|j, d<=K} mu(d)
        
        max_x = 10 * K
        g = np.zeros(max_x + 1, dtype=np.float64)
        for d in range(1, K + 1):
            if mu[d] != 0:
                g[d:max_x+1:d] += mu[d]
        
        # M_raw(x) for x in [1, max_x]
        # M_raw(x) = sum_{j=1}^x g(j)
        M_raw = np.cumsum(g[1:max_x+1])
        
        # M_adjusted(x) = M_raw(x) - S*x
        x_vals = np.arange(1, max_x + 1)
        M_adj = M_raw - S * x_vals
        
        max_M = np.max(M_adj)
        
        if max_M <= 1.0001:
            # Feasible! Compute the objective
            # a = -sum_k f(k) * log(k) / k
            k_vals = np.arange(1, K+1)
            a = -np.sum(f * np.log(k_vals) / k_vals)
            return domain, values, a, max_M
        else:
            return None, None, None, max_M

    def try_lp_approach(K):
        """
        Use LP to find optimal f for domain [1, K] with the constraint that
        after the evaluator's adjustment, M(x) <= 1.0001.
        
        Decision variables: f(1), f(2), ..., f(K)
        
        After adjustment: f'(1) = f(1) - S where S = sum_k f(k)/k
        M(x) = sum_{k=1}^K f'(k) * floor(x/k) where f'(k) = f(k) for k >= 2, f'(1) = f(1) - S
        
        M(x) = (f(1) - S) * x + sum_{k=2}^K f(k) * floor(x/k)
             = f(1)*x - S*x + sum_{k=2}^K f(k) * floor(x/k)
             = f(1)*x - x*sum_{k=1}^K f(k)/k + sum_{k=2}^K f(k) * floor(x/k)
             = f(1)*x - f(1) - sum_{k=2}^K f(k)*x/k + sum_{k=2}^K f(k) * floor(x/k)
             = f(1)*(x-1) + sum_{k=2}^K f(k) * (floor(x/k) - x/k)
        
        So M(x) = f(1)*(x-1) + sum_{k=2}^K f(k) * (floor(x/k) - x/k)
        
        We need M(x) <= 1.0001 for all x in [1, 10K]
        
        Also: -10 <= f(k) <= 10 for all k
        
        Objective: maximize a = -sum_k f(k) * log(k) / k
                 = -f(1)*log(1)/1 - sum_{k=2}^K f(k) * log(k)/k
                 = -sum_{k=2}^K f(k) * log(k)/k  (since log(1) = 0)
        
        So we want to minimize sum_{k=2}^K f(k) * log(k)/k
        
        This is an LP with:
        - K variables: f(1), f(2), ..., f(K)
        - Constraints: -10 <= f(k) <= 10 for all k
        - Constraints: M(x) <= 1.0001 for all x in [1, 10K]
        
        The number of M(x) constraints is 10K, which can be large.
        For K=200, that's 2000 constraints, which is manageable.
        """
        if K > 300:
            return None, None, None
        
        # Build constraint matrix for M(x) <= 1.0001
        max_x = 10 * K
        
        # For each x, M(x) = f(1)*(x-1) + sum_{k=2}^K f(k) * (floor(x/k) - x/k)
        # Coefficient of f(1) is (x-1)
        # Coefficient of f(k) for k>=2 is (floor(x/k) - x/k)
        
        # We'll build the A matrix and b vector for linprog
        # linprog minimizes c^T x subject to A_ub @ x <= b_ub, bounds
        
        # Number of variables = K
        n_vars = K
        
        # Objective: minimize sum_{k=2}^K f(k) * log(k)/k
        # c[0] = 0 (coefficient of f(1))
        # c[k-1] = log(k)/k for k >= 2
        c = np.zeros(n_vars)
        for k in range(2, K+1):
            c[k-1] = np.log(k) / k
        
        # Bounds: -10 <= f(k) <= 10
        bounds = [(-10, 10)] * n_vars
        
        # Constraints: M(x) <= 1.0001 for x in [1, 10K]
        # A_ub has shape (10K, K)
        # b_ub = 1.0001 * ones(10K)
        
        # For memory efficiency, we can be smart about this.
        # But for K=200, 10K=2000 rows, K=200 cols, that's 400K floats, fine.
        
        A_ub = np.zeros((max_x, n_vars))
        b_ub = np.full(max_x, 1.0001)
        
        for x_idx, x in enumerate(range(1, max_x + 1)):
            # Coefficient of f(1)
            A_ub[x_idx, 0] = x - 1
            # Coefficients of f(k) for k >= 2
            for k in range(2, K+1):
                A_ub[x_idx, k-1] = np.floor(x / k) - x / k
        
        # Solve LP
        try:
            result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if result.success:
                f_vals = result.x
                domain = np.arange(1, K+1, dtype=np.int64)
                values = f_vals.astype(np.float64)
                
                # Verify feasibility
                # Compute M(x) after adjustment
                S = np.sum(f_vals / np.arange(1, K+1))
                f_adj = f_vals.copy()
                f_adj[0] = f_vals[0] - S
                
                max_x = 10 * K
                g = np.zeros(max_x + 1, dtype=np.float64)
                for d in range(1, K + 1):
                    val = f_adj[d-1]
                    if val != 0:
                        g[d:max_x+1:d] += val
                M = np.cumsum(g[1:max_x+1])
                max_M = np.max(M)
                
                if max_M <= 1.0001:
                    a = -np.sum(f_vals * np.log(np.arange(1, K+1)) / np.arange(1, K+1))
                    return domain, values, a, max_M
                else:
                    return None, None, None, max_M
            else:
                return None, None, None, None
        except Exception:
            return None, None, None, None

    # Strategy: Try LP for small K, Mobius for larger K
    # LP is exact but slow for large K
    # Mobius is fast but may not be optimal
    
    best_domain = None
    best_values = None
    best_a = -np.inf
    best_max_M = None
    
    # Try LP for increasing K
    for K_lp in [50, 100, 150, 200, 250, 300]:
        if time.time() > deadline - 60:
            break
        
        domain, values, a, max_M = try_lp_approach(K_lp)
        if domain is not None and a > best_a:
            best_domain = domain
            best_values = values
            best_a = a
            best_max_M = max_M
    
    # Try Mobius for larger K
    for K_mob in [500, 1000, 5000, 10000, 50000, 100000, 200000]:
        if time.time() > deadline - 30:
            break
        
        domain, values, a, max_M = try_mobius_approach(K_mob)
        if domain is not None and a > best_a:
            best_domain = domain
            best_values = values
            best_a = a
            best_max_M = max_M
    
    # If we found a good solution, return it
    if best_domain is not None:
        return best_domain, best_values
    
    # Fallback to small baseline
    return np.array([1, 2, 3, 5, 30], dtype=np.int64), np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)

# EVOLVE-BLOCK-END
