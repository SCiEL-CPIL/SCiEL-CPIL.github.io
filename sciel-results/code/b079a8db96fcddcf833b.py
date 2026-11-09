# EVOLVE-BLOCK-START
import numpy as np
import time

def compute_mobius(n):
    """Compute Möbius function mu(k) for k = 0..n using linear sieve."""
    mu = np.zeros(n + 1, dtype=np.int64)
    mu[1] = 1
    is_prime = np.ones(n + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
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
    return mu

def check_feasibility_v2(mu, K, max_x):
    """
    Check if M(x) <= 1.0001 for all x in [1, max_x].
    Uses the structure: M(x) = sum_{k=1}^{min(x,K)} mu(k) * floor(x/k)
    Group by floor(x/k) values for efficiency.
    """
    # Precompute prefix sums of mu
    mu_prefix = np.zeros(K + 1, dtype=np.float64)
    mu_prefix[1:] = np.cumsum(mu[1:K+1].astype(np.float64))
    
    M_max = 0.0
    # Check M(x) for x = 1, 2, ..., max_x
    # For large x, floor(x/k) changes slowly for small k
    # We check all x but use efficient computation
    
    # For x <= K: M(x) = sum_{k=1}^{x} mu(k) * floor(x/k)
    # For x > K: M(x) = sum_{k=1}^{K} mu(k) * floor(x/k)
    
    # Check x = 1
    M_1 = mu[1]  # = 1
    if M_1 > M_max:
        M_max = M_1
    
    # For x from 2 to max_x, compute M(x) efficiently
    # M(x) = sum_{k=1}^{min(x,K)} mu(k) * floor(x/k)
    # Group terms where floor(x/k) is constant
    
    for x in range(2, max_x + 1):
        k_max = min(x, K)
        M_x = 0.0
        k = 1
        while k <= k_max:
            q = x // k
            if q == 0:
                break
            k_next = x // q + 1
            k_next = min(k_next, k_max + 1)
            # Sum mu(k) for k in [k, k_next-1]
            s = mu_prefix[k_next - 1] - mu_prefix[k - 1]
            M_x += s * q
            k = k_next
        if M_x > M_max:
            M_max = M_x
            if M_max > 1.0001:
                return False, M_max
    return True, M_max

def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    """
    start_time = time.time()
    deadline = start_time + 570  # Safety margin before 630s limit
    
    max_K = 200000
    mu = compute_mobius(max_K)
    
    best_domain = None
    best_values = None
    best_score = -np.inf
    
    # Try K values from large to small
    K_list = [200000, 150000, 100000, 75000, 50000, 30000, 20000, 10000, 5000]
    
    for K in K_list:
        if time.time() > deadline - 30:
            break
        
        max_x = 10 * K
        
        # Check feasibility
        feasible, M_max = check_feasibility_v2(mu, K, max_x)
        
        if feasible:
            # Extract domain and values where mu(k) != 0
            mask = mu[1:K + 1] != 0
            domain = np.nonzero(mask)[0] + 1
            values = mu[1:K + 1][mask].astype(np.float64)
            
            # Compute the score a = -sum f(k) * log(k) / k
            log_k = np.log(domain.astype(np.float64))
            a = -np.sum(values * log_k / domain.astype(np.float64))
            
            if a > best_score:
                best_score = a
                best_domain = domain
                best_values = values
                
            # Found feasible solution with large K, stop
            if K >= 50000:
                break
    
    # Fallback
    if best_domain is None:
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values
    
    return best_domain, best_values

# EVOLVE-BLOCK-END
