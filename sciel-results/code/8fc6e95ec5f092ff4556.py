import numpy as np
import time
from scipy.optimize import linprog
from scipy.sparse import csr_matrix
import sys

def _compute_mu(n):
    """Compute Mobius function for 1..n."""
    mu = np.ones(n + 1, dtype=np.int64)
    mu[0] = 0
    is_prime = np.ones(n + 1, dtype=bool)
    is_prime[:2] = False
    primes = []
    
    for i in range(2, n + 1):
        if is_prime[i]:
            primes.append(i)
            mu[i] = -1
        for p in primes:
            if i * p > n:
                break
            is_prime[i * p] = False
            if i % p == 0:
                mu[i * p] = 0
                break
            else:
                mu[i * p] = -mu[i]
    mu[1] = 1
    return mu

def prime_number_theorem_partial_function():
    start_time = time.time()
    deadline = start_time + 590
    
    # Try different K values
    # Start with smaller K to ensure we get a result, then try larger if time permits
    K_candidates = [2000, 5000, 10000]
    
    best_domain = None
    best_values = None
    best_a = -np.inf
    
    for K in K_candidates:
        if time.time() > deadline - 60:
            break
            
        print(f"Trying K={K}", file=sys.stderr)
        
        mu = _compute_mu(K)
        
        # Variables are f_2, ..., f_K
        # Index mapping: var_idx k-2 corresponds to f_k
        n_vars = K - 1
        if n_vars <= 0:
            continue
            
        # Objective: Minimize sum_{k=2}^K (log(k)/k) * f_k
        # We want to maximize -sum, so minimize sum.
        c_obj = np.zeros(n_vars)
        ks = np.arange(2, K + 1)
        c_obj[:] = np.log(ks) / ks
        
        # Constraints:
        # sum_{k=2}^K f_k * (floor(x/k) - x/k) <= 1.0001 for x in [1, 10K]
        # Also -10 <= f_k <= 10
        
        # Construct A matrix
        # A has shape (10K + n_vars*2, n_vars)
        # We use sparse matrix to save memory
        
        x_max = 10 * K
        n_constraints_M = x_max
        n_bounds = 2 * n_vars
        total_constraints = n_constraints_M + n_bounds
        
        A_rows = []
        A_cols = []
        A_vals = []
        b = np.zeros(total_constraints)
        
        # M constraints
        # For each x, for each k, coeff is floor(x/k) - x/k
        # This is O(K * 10K) which is 10 * K^2. For K=10000, 10^9 ops. Too slow in Python loop.
        # We need to vectorize or use a smarter way.
        
        # Actually, we can compute the matrix A_M more efficiently?
        # A_M[x-1, k-2] = floor(x/k) - x/k
        
        # Let's try to build it row by row for x, using vectorized numpy for k.
        A_M = np.zeros((x_max, n_vars), dtype=np.float64)
        
        # This loop is O(x_max * n_vars) if done naively.
        # x_max = 10K, n_vars = K. 10 K^2.
        # For K=2000, 4e7. Feasible.
        # For K=5000, 2.5e8. Might be slow in Python, but numpy vectorized per row is fast.
        
        for x in range(1, x_max + 1):
            # coeffs for k=2..K
            # floor(x/k) - x/k
            # x is fixed. k varies.
            # We can compute this for all k at once.
            k_vals = ks
            coeffs = np.floor(x / k_vals) - x / k_vals
            A_M[x-1, :] = coeffs
            b[x-1] = 1.0001
            
        # Bound constraints: -10 <= f_k <= 10
        # f_k <= 10  =>  f_k <= 10
        # -f_k <= 10 => -f_k <= 10
        
        # A_bound is identity for first n_vars rows, -identity for next n_vars rows
        # But we are appending to A_M.
        
        # Construct full A, b
        # A = [A_M; I; -I]
        # b = [1.0001...; 10...; 10...]
        
        I = np.eye(n_vars)
        A_full = np.vstack([A_M, I, -I])
        b_full = np.concatenate([b, np.full(n_vars, 10.0), np.full(n_vars, 10.0)])
        
        # Solve LP
        try:
            res = linprog(c_obj, A_ub=A_full, b_ub=b_full, method='highs')
            
            if res.status == 0:
                f_vars = res.x
                # Construct full f vector
                f_full = np.zeros(K + 1)
                f_full[1] = 0.0
                f_full[2:K+1] = f_vars
                
                # Calculate objective a
                # a = -sum f(k) log(k) / k
                # Note: f(1) term is 0.
                log_k = np.log(ks)
                a_val = -np.sum(f_vars * log_k / ks)
                
                # Verify feasibility quickly
                # M(x) = sum f(k) (floor(x/k) - x/k)
                # We trust the LP, but let's check max M(x)
                # M_vec = A_M @ f_vars
                # M_max = np.max(M_vec)
                # if M_max <= 1.0001:
                
                if a_val > best_a:
                    best_a = a_val
                    best_domain = ks
                    best_values = f_vars
                    print(f"K={K}, a={a_val}", file=sys.stderr)
            else:
                print(f"LP failed for K={K}: {res.message}", file=sys.stderr)
                
        except Exception as e:
            print(f"LP error for K={K}: {e}", file=sys.stderr)

    if best_domain is None:
        # Fallback
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([0.0, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values
    
    # Include 1 in domain? The problem says domain holds distinct positive integers.
    # f(1) is adjusted by evaluator.
    # If 1 is not in domain, f(1) is treated as 0 before adjustment.
    # Adjustment: f(1) = f(1) - S. If f(1) was 0, it becomes -S.
    # My LP assumed f(1) cancels out.
    # If I return domain without 1, evaluator sets f(1)=0, then adjusts to -S.
    # This matches my LP assumption that f(1) is effectively determined by others.
    # However, the metric a uses the *adjusted* values.
    # a = - sum f_adj(k) log(k)/k.
    # f_adj(1) = -S. log(1)=0. So f_adj(1) doesn't contribute to a.
    # So returning domain without 1 is fine, as long as the values for k>1 are correct.
    
    return best_domain.astype(np.int64), best_values.astype(np.float64)
