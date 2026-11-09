# EVOLVE-BLOCK-START
import numpy as np


def prime_number_theorem_partial_function():
    """
    Construct a partial function f that maximizes the metric a = -sum_k f(k)*log(k)/k
    subject to the feasibility constraint M(x) = sum_k f(k)*floor(x/k) <= 1.0001 for all x in [1, 10K].

    The M(o)bius function mu(k) has the property sum_{k<=x} mu(k) floor(x/k) = 1 for all x.
    Truncating at K=200000 and scaling by c slightly below 1.0001/max(M(x)) gives a feasible
    function with metric close to 1.0.
    """
    K = 200000
    x_max = 10 * K

    # Compute M(o)bius function up to K using linear sieve
    mu = np.zeros(K + 1, dtype=np.int64)
    mu[0] = 0
    mu[1] = 1
    primes = []
    is_composite = np.zeros(K + 1, dtype=bool)

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

    # Compute M(x) = sum_k mu(k) * floor(x/k) for x in [1, x_max]
    # Using the identity: sum_{k=1}^K mu(k) floor(x/k)
    # We can compute this efficiently using the fact that floor(x/k) changes value O(sqrt(x)) times
    # But for simplicity and given K=200000, x_max=2000000, let's use a direct but vectorized approach
    # Actually, let's use the standard approach: for each k, add mu[k] to all multiples in the floor sense
    # M(x) = sum_{k=1}^{min(x,K)} mu[k] * floor(x/k)
    # We can compute M for all x using the fact that this is a Dirichlet convolution with floor
    
    # Efficient computation: M(x) = sum_{k=1}^K mu[k] * floor(x/k)
    # For each k, floor(x/k) is constant on intervals. But we need all x values.
    # Alternative: use the relation that M is the cumulative sum of mu * 1 (convolution)
    # Actually, let's just compute it directly using numpy for each x is too slow.
    # Let's use the fact that M(x) = sum_{d|x} (sum_{k|d, k<=K} mu[k]) is not right.
    
    # The standard way: M(x) = sum_{k=1}^K mu[k] * floor(x/k)
    # We can compute this by iterating k and adding mu[k] to positions where floor(x/k) contributes.
    # Actually, floor(x/k) = number of multiples of k up to x.
    # So M(x) = sum_{k=1}^K mu[k] * (number of multiples of k up to x)
    # = sum_{m=1}^x (sum_{k|m, k<=K} mu[k])
    # = sum_{m=1}^x (sum_{k|m} mu[k]) for x <= K, and partial for x > K.
    # sum_{k|m} mu[k] = 1 if m=1, else 0 (this is the identity: sum_{d|n} mu(d) = 1 if n=1, else 0)
    # But we only sum k up to K, not all divisors.
    
    # For x <= K: M(x) = sum_{m=1}^x sum_{k|m, k<=K} mu[k] = sum_{m=1}^x 1 (if m=1) + 0 (if m>1) = 1
    # Wait, that's only if K >= x. For x <= K, sum_{k|m, k<=K} mu[k] = sum_{k|m} mu[k] = 1 if m=1 else 0.
    # So M(x) = 1 for all x <= K.
    
    # For x > K: M(x) = sum_{m=1}^x sum_{k|m, k<=K} mu[k]
    # = sum_{m=1}^K sum_{k|m, k<=K} mu[k] + sum_{m=K+1}^x sum_{k|m, k<=K} mu[k]
    # = 1 + sum_{m=K+1}^x sum_{k|m, k<=K} mu[k]
    # = 1 - sum_{m=K+1}^x sum_{k|m, k>K} mu[k]
    # This is getting complex. Let's just compute M(x) directly using numpy.
    
    # Direct computation: M(x) = sum_{k=1}^K mu[k] * floor(x/k)
    # We can vectorize over k for each x, but that's O(x_max * K) which is too slow.
    # Instead, we can compute M for all x at once using the following:
    # M = zeros(x_max + 1)
    # For each k from 1 to K:
    #   M[k::k] += mu[k]  -- this adds mu[k] to all multiples of k
    # But that gives sum_{k|m, k<=K} mu[k] for each m, not floor(x/k).
    
    # Actually, M(x) = sum_{k=1}^K mu[k] * floor(x/k)
    # Let's use the identity: floor(x/k) = sum_{j=1}^{floor(x/k)} 1
    # So M(x) = sum_{k=1}^K mu[k] * sum_{j=1}^{floor(x/k)} 1 = sum_{k=1}^K mu[k] * sum_{j: kj <= x} 1
    # = sum_{m=1}^x sum_{k: k|m, k<=K} mu[k]
    
    # So let's compute g(m) = sum_{k|m, k<=K} mu[k] for m=1..x_max
    # Then M(x) = sum_{m=1}^x g(m)
    
    # To compute g(m) for all m:
    # g = zeros(x_max + 1)
    # For each k from 1 to K:
    #   g[k::k] += mu[k]
    
    g = np.zeros(x_max + 1, dtype=np.float64)
    mu_float = mu.astype(np.float64)
    for k in range(1, K + 1):
        if mu[k] != 0:
            g[k::k] += mu[k]
    
    M = np.cumsum(g)  # M[x] = sum_{m=1}^x g(m) for x >= 1, M[0] = 0
    
    # Find the maximum of M(x) for x in [1, x_max]
    M_max = np.max(M[1:])  # M[1] to M[x_max]
    
    # The scaling factor c must satisfy c * M_max <= 1.0001
    # So c <= 1.0001 / M_max
    # We use c = 1.0001 / M_max to maximize the metric
    c = 1.0001 / M_max
    
    # Construct domain and values
    # Domain: all k from 1 to K where mu[k] != 0
    nonzero_mask = mu[1:] != 0
    domain = np.nonzero(nonzero_mask)[0] + 1
    values = mu[1:][nonzero_mask] * c
    
    return domain, values


# EVOLVE-BLOCK-END
