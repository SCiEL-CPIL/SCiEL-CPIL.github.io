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
    deadline = start_time + 600  # 600s safety margin
    
    # Try LP-based approach for moderate K
    # Variables: f(1), f(2), ..., f(K)
    # Objective: maximize -sum f(k)*log(k)/k  =>  minimize sum f(k)*log(k)/k
    # Constraints: M(x) = sum_k f(k)*floor(x/k) <= 1 for x in [1, 10*K]
    #              f(k) >= -10, f(k) <= 10 (clipping bounds, but we can let LP handle it)
    
    # For efficiency, note that M(x) for x >= K is: sum_{k=1}^K f(k)*floor(x/k)
    # The most constraining x values are likely small ones.
    
    # Strategy: For a given K, build the constraint matrix A where A[x,k] = floor(x/k)
    # This is a K x (10K) matrix, which is large. But we can be smarter.
    
    # Key insight: M(x) = sum_{n=1}^x (sum_{k|n} f(k)) for the full sum, but here
    # M(x) = sum_{k=1}^K f(k) * floor(x/k) = sum_{k=1}^K f(k) * (number of multiples of k in [1,x])
    # = sum_{n=1}^x (sum_{k|n, k<=K} f(k))
    
    # Let g(n) = sum_{k|n, k<=K} f(k). Then M(x) = sum_{n=1}^x g(n).
    # Constraint: sum_{n=1}^x g(n) <= 1 for all x in [1, 10K].
    # This is equivalent to: the partial sums of g are bounded by 1.
    
    # This means g(n) can be positive or negative, but cumulative sum <= 1.
    
    # For the LP, the number of constraints is 10K. For K=500, that's 5000 constraints
    # and 500 variables. This is feasible for linprog.
    
    candidate_Ks = [2000, 1000, 500, 200, 100]
    
    best_domain = None
    best_values = None
    best_obj = -np.inf
    
    for K in candidate_Ks:
        if time.time() > deadline - 120:
            break
            
        # Build constraint matrix A: shape (10K, K)
        # A[x-1, k-1] = floor(x/k) for x in [1, 10K], k in [1, K]
        x_vals = np.arange(1, 10*K + 1)
        k_vals = np.arange(1, K + 1)
        
        # Vectorized: A[i,j] = floor((i+1)/(j+1))
        # This is a (10K) x K matrix
        A = np.floor(x_vals[:, None] / k_vals[None, :]).astype(np.float64)
        
        # Objective: minimize c^T f where c[k] = log(k+1)/(k+1)
        # (since we want to maximize -sum f(k)*log(k)/k)
        c = np.log(k_vals) / k_vals
        c = c.astype(np.float64)
        
        # Bounds: f(k) in [-10, 10]
        bounds = [(-10, 10)] * K
        
        # Inequality: A @ f <= 1
        b = np.ones(10*K)
        
        # Solve LP
        try:
            res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')
            if time.time() > deadline - 60:
                break
            if res.success:
                f = res.x
                # Verify feasibility
                M = A @ f
                if np.max(M) <= 1.0001:
                    # Compute objective
                    obj = -np.sum(f * c)
                    if obj > best_obj:
                        best_obj = obj
                        best_domain = k_vals.astype(np.int64)
                        best_values = f.astype(np.float64)
        except Exception:
            continue
    
    if best_domain is not None:
        return best_domain, best_values
    
    # Fallback: use Möbius function
    K_max = 20000
    mu = np.zeros(K_max + 1, dtype=np.int8)
    mu[1] = 1
    is_prime = np.ones(K_max + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    primes = []
    for i in range(2, K_max + 1):
        if is_prime[i]:
            primes.append(i)
            mu[i] = -1
        for p in primes:
            if i * p > K_max:
                break
            is_prime[i * p] = False
            if i % p == 0:
                mu[i * p] = 0
                break
            else:
                mu[i * p] = -mu[i]
    
    domain = np.arange(1, 1001, dtype=np.int64)
    values = mu[1:1001].astype(np.float64)
    
    return domain, values

# EVOLVE-BLOCK-END
