# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_mobius(n):
    """Compute the Mobius function mu(k) for k = 1..n using a linear sieve."""
    mu = np.ones(n + 1, dtype=np.int64)
    primes = []
    is_prime = np.ones(n + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    
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
    
    return mu


def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.

    The function f is scored by a fixed evaluator (see evaluator.py) that:
      * subtracts a constant from f(1) so that sum_k f(k)/k == 0,
      * requires the step function M(x) = sum_k f(k) * floor(x/k) to stay <= 1 for every
        real x in [1, 10*max_key] (otherwise the construction is infeasible), and
      * returns the objective a = -sum_k f(k) * log(k) / k (to be maximized; optimum 1.0).
    The optimum is approached by the Mobius function mu(k), for which the classic identity
    sum_{k<=x} mu(k) floor(x/k) = 1 holds and -sum mu(k) log(k)/k -> 1.

    Representation:
        domain : np.ndarray, shape (m,), dtype int   -> positive integers in the domain of f
        values : np.ndarray, shape (m,), dtype float -> the corresponding real values f(k)

    Returns:
        (domain, values): a tuple of two numpy arrays of equal length describing f.
    """
    start_time = time.time()
    deadline = start_time + 580  # Leave 20s safety margin
    
    # Try progressively smaller K if needed
    for K in [200000, 100000, 50000, 20000, 10000, 5000, 2000, 1000, 500, 200, 100, 50, 20, 10, 5, 3, 2]:
        if time.time() > deadline:
            break
        
        # Compute Mobius function up to K
        mu = _compute_mobius(K)
        
        # Domain: square-free numbers (where mu(k) != 0)
        mask = mu != 0
        domain = np.where(mask)[0]
        domain = domain[domain >= 1]  # Ensure positive integers
        values = mu[domain].astype(np.float64)
        
        if len(domain) == 0:
            continue
        
        # Verify constraint: M(x) <= 1.0001 for all x in [1, 10*K]
        # M(x) = sum_k f(k) * floor(x/k)
        # Use vectorized computation in chunks to avoid memory issues
        
        max_x = 10 * K
        m = len(domain)
        
        # Check if we have time for verification
        if time.time() > deadline - 30:
            # Not enough time to verify, return a safe small case
            if K <= 10:
                return domain, values
            continue
        
        # Compute M(x) for all x in [1, max_x]
        # Process in chunks to manage memory
        chunk_size = min(100000, max_x)
        feasible = True
        
        for x_start in range(1, max_x + 1, chunk_size):
            x_end = min(x_start + chunk_size - 1, max_x)
            x_chunk = np.arange(x_start, x_end + 1)
            
            # Compute M(x) for this chunk
            # M(x) = sum_k f(k) * floor(x/k)
            # Vectorize over k
            M_chunk = np.zeros(len(x_chunk), dtype=np.float64)
            
            # Process in sub-chunks over k to manage memory
            k_chunk_size = 1000
            for k_start in range(0, m, k_chunk_size):
                k_end = min(k_start + k_chunk_size, m)
                k_chunk = domain[k_start:k_end]
                f_chunk = values[k_start:k_end]
                
                # floor(x/k) for x in x_chunk, k in k_chunk
                # Shape: (len(x_chunk), len(k_chunk))
                floor_vals = np.floor(np.outer(x_chunk, 1.0) / k_chunk)
                M_chunk += np.dot(floor_vals, f_chunk)
                
                if time.time() > deadline - 10:
                    feasible = False
                    break
            
            if not feasible:
                break
            
            # Check constraint
            if np.max(M_chunk) > 1.0001:
                feasible = False
                break
        
        if feasible:
            return domain, values
    
    # Fallback: simple known-good case
    domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
    values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
    return domain, values


# EVOLVE-BLOCK-END
