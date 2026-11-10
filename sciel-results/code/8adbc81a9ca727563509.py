# EVOLVE-BLOCK-START
import numpy as np
import time


def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.

    The function f is scored by a fixed evaluator (see evaluator.py) that:
      * clips values to [-10, 10],
      * subtracts a constant from f(1) so that sum_k f(k)/k == 0 (using clipped values),
      * requires the step function M(x) = sum_k f(k) * floor(x/k) to stay <= 1 for every
        integer x in [1, 10*max_key] (otherwise the construction is infeasible), and
      * returns the objective a = -sum_k f(k) * log(k) / k (to be maximized; optimum 1.0).
    The optimum is approached by the Mobius function mu(k), for which the classic identity
    sum_{k<=x} mu(k) floor(x/k) = 1 holds and -sum mu(k) log(k)/k -> 1.

    Representation:
        domain : np.ndarray, shape (m,), dtype int   -> positive integers in the domain of f
        values : np.ndarray, shape (m,), dtype float -> the corresponding real values f(k)

    Returns:
        (domain, values): a tuple of two numpy arrays of equal length describing f
    """
    start_time = time.time()
    deadline = start_time + 580  # Leave 50s margin for safety

    # Sieve for Mobius function up to K_max
    K_max = 50000
    mu = np.zeros(K_max + 1, dtype=np.int8)
    mu[1] = 1
    is_prime = np.ones(K_max + 1, dtype=bool)
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

    # Precompute
    k_vals = np.arange(1, K_max + 1, dtype=np.float64)
    mu_float = mu[1:].astype(np.float64)
    mu_over_k = mu_float / k_vals

    def check_M_constraint(values, K, x_max):
        """
        Check if M(x) = sum_k f(k) * floor(x/k) <= 1.0001 for all x in [1, x_max].
        Uses interval-based floor division for efficiency.
        """
        M_prime = np.zeros(x_max + 1, dtype=np.float64)  # 1-indexed
        for k in range(1, K + 1):
            val = values[k - 1]
            if val == 0.0:
                continue
            # floor(x/k) is constant on intervals [j*k, (j+1)*k - 1]
            # For x in [1, x_max], j ranges from 0 to floor(x_max/k)
            # But we only need x >= 1, so j starts at 0 (x in [0, k-1] -> floor=0)
            # Actually for x in [1, x_max]:
            #   j = 0: x in [1, k-1] -> floor(x/k) = 0 (if k > 1) or x in [1,0] empty (if k=1)
            #   j = 1: x in [k, 2k-1] -> floor(x/k) = 1
            #   j = 2: x in [2k, 3k-1] -> floor(x/k) = 2
            #   ...
            #   j = max_j: x in [j*k, min((j+1)*k-1, x_max)]
            max_j = x_max // k
            for j in range(1, max_j + 1):
                lo = j * k
                hi = min((j + 1) * k - 1, x_max)
                M_prime[lo:hi + 1] += val * j
        return np.max(M_prime[1:])  # x from 1 to x_max

    def check_M_constraint_fast(values, K, x_max):
        """
        Faster check using numpy vectorization for small k values.
        For large k, use the interval approach.
        """
        M_prime = np.zeros(x_max + 1, dtype=np.float64)  # 1-indexed
        x_range = np.arange(1, x_max + 1, dtype=np.float64)
        
        # For k up to some threshold, use direct computation
        threshold = 200
        for k in range(1, min(threshold, K) + 1):
            val = values[k - 1]
            if val != 0.0:
                M_prime[1:] += val * (x_range // k)
        
        # For larger k, use interval approach
        for k in range(max(1, threshold), K + 1):
            val = values[k - 1]
            if val == 0.0:
                continue
            max_j = x_max // k
            for j in range(1, max_j + 1):
                lo = j * k
                hi = min((j + 1) * k - 1, x_max)
                M_prime[lo:hi + 1] += val * j
        
        return np.max(M_prime[1:])

    best_domain = None
    best_values = None
    best_score = -np.inf

    # Try various K values
    # Start with larger K for potentially better scores, but be mindful of compute time
    # M(x) check cost is roughly O(K * x_max / k_avg) with the interval method
    # For K=2000, x_max=20000: sum_{k=1}^{2000} (20000/k) ≈ 20000 * ln(2000) ≈ 150000 intervals
    # That's very fast!
    
    # Let's try K values: 4000, 3500, 3000, 2500, 2000, 1500, 1000
    K_candidates = [4000, 3500, 3000, 2500, 2000, 1500, 1000, 800, 600, 400, 200]
    
    for K in K_candidates:
        if time.time() > deadline:
            break
        if K > K_max:
            continue
            
        # Compute S1 = sum_{k=1}^K mu(k)/k (before clipping)
        S1 = np.sum(mu_over_k[:K])
        
        # Construct initial f
        # f(1) = 1 - S1
        # f(k) = mu(k) for k > 1
        values = mu_float[:K].copy()
        values[0] = 1.0 - S1
        
        # Clip to [-10, 10] as the evaluator does
        values_clipped = np.clip(values, -10.0, 10.0)
        
        # Now compute S using clipped values
        S = np.sum(values_clipped / k_vals[:K])
        
        # Adjust f(1) so that sum_k f(k)/k = 0
        # We need: f(1)/1 + sum_{k=2}^K f(k)/k = 0
        # f(1) = -sum_{k=2}^K f(k)/k
        if K >= 1:
            sum_rest = np.sum(values_clipped[1:] / k_vals[1:K])
            values_clipped[0] = -sum_rest
        
        # Re-clip f(1) (it should be in [-10,10] for reasonable K)
        values_clipped = np.clip(values_clipped, -10.0, 10.0)
        
        # Check M(x) constraint
        x_max = 10 * K
        M_max = check_M_constraint_fast(values_clipped, K, x_max)
        
        if M_max <= 1.0001:
            # Calculate score: a = -sum_k f(k) * log(k) / k
            log_k = np.log(k_vals[:K])
            score = -np.sum(values_clipped * log_k / k_vals[:K])
            if score > best_score:
                best_score = score
                best_domain = k_vals[:K].astype(np.int64)
                best_values = values_clipped.copy()

    if best_domain is None:
        # Fallback: use a small known-good solution
        # Try K=100
        for K in [100, 50, 20, 10]:
            if K > K_max:
                continue
            S1 = np.sum(mu_over_k[:K])
            values = mu_float[:K].copy()
            values[0] = 1.0 - S1
            values_clipped = np.clip(values, -10.0, 10.0)
            S = np.sum(values_clipped / k_vals[:K])
            if K >= 1:
                sum_rest = np.sum(values_clipped[1:] / k_vals[1:K])
                values_clipped[0] = -sum_rest
            values_clipped = np.clip(values_clipped, -10.0, 10.0)
            
            x_max = 10 * K
            M_max = check_M_constraint_fast(values_clipped, K, x_max)
            
            if M_max <= 1.0001:
                best_domain = k_vals[:K].astype(np.int64)
                best_values = values_clipped
                break

    if best_domain is None:
        # Final fallback: seed solution
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
    else:
        domain = best_domain
        values = best_values

    return domain, values

# EVOLVE-BLOCK-END
