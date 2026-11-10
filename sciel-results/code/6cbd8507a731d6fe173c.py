# EVOLVE-BLOCK-START
import numpy as np
import time

def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    """
    start_time = time.time()
    deadline = 600.0

    def compute_mobius(K):
        mu = np.ones(K + 1, dtype=np.int8)
        is_prime = np.ones(K + 1, dtype=bool)
        primes = []
        is_prime[0] = False
        is_prime[1] = False
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

    # Optimized feasibility check using O(N log N) algorithm
    # B_K(x) = sum_{k=1}^K mu(k) floor(x/k)
    # We can compute B_K(x) for all x in 1..N efficiently.
    # Let N = 10K.
    # B_K(x) = sum_{n=1}^x (sum_{d|n, d<=K} mu(d))
    # Let g(n) = sum_{d|n, d<=K} mu(d).
    # Then B_K(x) = sum_{n=1}^x g(n).
    # We can compute g(n) for n=1..N by iterating d=1..K and adding mu(d) to multiples of d.
    # This is O(N log N).
    
    def is_feasible(K, mu):
        N = 10 * K
        # g[n] will hold sum_{d|n, d<=K} mu(d)
        # We only need n up to N.
        # Note: d only goes up to K.
        g = np.zeros(N + 1, dtype=np.int64)
        
        # For each d in 1..K, add mu[d] to g[m*d] for m=1..floor(N/d)
        for d in range(1, K + 1):
            md = mu[d]
            if md != 0:
                # Add md to indices d, 2d, 3d, ... <= N
                # g[d::d] += md
                # This is efficient in numpy
                g[d:N+1:d] += md
        
        # B_K(x) = sum_{n=1}^x g(n)
        # So B_K(x) is the cumulative sum of g.
        # B_K(0) = 0
        B = np.cumsum(g)
        
        # We need S = sum_{k=1}^K mu(k)/k
        k_arr = np.arange(1, K + 1)
        S = np.sum(mu[1:K+1] / k_arr)
        
        # M(x) = B_K(x) - S*x
        # Check max_{x in 1..N} M(x) <= 1.0001
        x = np.arange(1, N + 1)
        M_vals = B[1:N+1] - S * x
        
        return np.max(M_vals) <= 1.0001

    best_score = -np.inf
    best_domain = None
    best_values = None

    # Try different K
    # We can go larger now due to O(N log N) feasibility check.
    # K=200000 -> N=2,000,000.
    # Loop K times, each doing N/K operations? No, sum_{d=1}^K N/d = N log K.
    # 2000000 * log(200000) ~ 2.4e6 * 12 = 3e7 ops. Very fast in numpy.
    
    for K in [1000, 5000, 20000, 50000, 100000, 200000]:
        if time.time() > deadline - 10:
            break
        
        mu = compute_mobius(K)
        
        # Check feasibility
        if is_feasible(K, mu):
            # Calculate score
            k_arr = np.arange(1, K + 1)
            S = np.sum(mu[1:K+1] / k_arr)
            
            # Construct f
            f = mu[1:K+1].astype(np.float64)
            f[0] = 1.0 - S # f(1)
            
            # Clip
            f = np.clip(f, -10, 10)
            
            # a = -sum f(k) log(k) / k
            # log(1) = 0, so f(1) term is 0.
            a = -np.sum(f * np.log(k_arr) / k_arr)
            
            if a > best_score:
                best_score = a
                best_domain = k_arr
                best_values = f

    if best_domain is None:
        # Fallback
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values

    return best_domain, best_values

# EVOLVE-BLOCK-END
