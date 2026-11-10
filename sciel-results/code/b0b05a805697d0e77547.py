# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_mobius(K):
    """Compute the Möbius function μ(n) for n = 1..K using a sieve."""
    mu = np.ones(K + 1, dtype=np.int64)
    is_prime = np.ones(K + 1, dtype=bool)
    is_prime[0] = False
    if K >= 1:
        is_prime[1] = False
    primes = []
    for i in range(2, K + 1):
        if is_prime[i]:
            primes.append(i)
            mu[i] = -1
        for p in primes:
            if i * p > K:
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
    
    Uses the Möbius function scaled to satisfy the M(x) <= 1.0001 constraint.
    """
    start_time = time.time()
    deadline = 600.0  # seconds
    
    best_domain = None
    best_values = None
    best_score = -np.inf
    
    # Try increasing K values
    K_candidates = [1000, 5000, 10000, 20000, 50000, 100000, 200000]
    
    for K in K_candidates:
        elapsed = time.time() - start_time
        if elapsed > deadline - 30:
            break
            
        # Compute Mobius
        mu = _compute_mobius(K)
        
        # Compute S_K(n) = sum_{d|n, d<=K} mu(d) for n in 1..10K
        # M_K(x) = sum_{j=1}^x S_K(j)
        max_x = 10 * K
        S = np.zeros(max_x + 1, dtype=np.float64)
        
        # Sieve-like update for S
        # d goes from 1 to K
        # mu is 1-indexed in logic, 0-indexed in array (mu[0] is unused/0, mu[1] is mu(1))
        # Our mu array from _compute_mobius has mu[0]=1 (default), mu[1]=1, mu[2]=-1 etc.
        # Wait, _compute_mobius returns mu[0..K].
        # mu[0] is 1 in the code above?
        # mu = np.ones(K+1). is_prime[0]=False.
        # Loop starts at 2.
        # So mu[0]=1, mu[1]=1.
        # We want mu(1)=1, mu(2)=-1.
        # So mu[i] in array corresponds to mu(i) for i>=1.
        
        for d in range(1, K + 1):
            val = mu[d]
            if val != 0:
                # Add val to S[d-1], S[d-1+d], S[d-1+2d], ...
                # Indices: d-1, 2d-1, 3d-1, ...
                S[d-1::d] += val
        
        # M_vals[x-1] = M_K(x)
        M_vals = np.cumsum(S)
        
        # Find max M(x) for x in 1..10K
        M_max = np.max(M_vals)
        
        # Compute T = sum_{k=1}^K mu(k) log(k) / k
        ks = np.arange(1, K + 1, dtype=np.float64)
        # mu is int, convert to float
        mu_float = mu[1:K+1].astype(np.float64)
        T = np.sum(mu_float * np.log(ks) / ks)
        
        # Calculate optimal scaling c
        if M_max > 0:
            c = 1.0001 / M_max
        else:
            # If M_max <= 0, we can scale up arbitrarily?
            # But f must be finite. And M(1)=1 always for mu(1)=1.
            # So M_max >= 1.
            c = 1.0
            
        # Compute score a = -c * T
        a = -c * T
        
        # Update best
        if a > best_score:
            best_score = a
            best_domain = np.arange(1, K + 1, dtype=np.int64)
            best_values = c * mu_float
            
    if best_domain is None:
        # Fallback
        best_domain = np.array([1], dtype=np.int64)
        best_values = np.array([0.0], dtype=np.float64)
        
    return best_domain, best_values


# EVOLVE-BLOCK-END
