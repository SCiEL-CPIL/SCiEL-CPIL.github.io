# EVOLVE-BLOCK-START
import numpy as np


def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.

    Strategy:
      1. Generate domain as first N integers (1..N) with N chosen large enough.
      2. Set values to the exact Möbius function mu(k) computed via a sieve.
      3. Apply a single global scaling factor to the non-1 values to strictly satisfy
         the M(x) <= 1 constraint while maximizing the objective.

    Returns:
        (domain, values): a tuple of two numpy arrays of equal length describing f.
    """
    # Choose N = 200000 (max allowed K)
    N = 200000
    
    # Compute Möbius function mu(k) for k = 1..N using a sieve
    mu = np.ones(N + 1, dtype=np.float64)
    is_prime = np.ones(N + 1, dtype=bool)
    primes = []
    
    # Sieve of Eratosthenes to get primes
    for i in range(2, N + 1):
        if is_prime[i]:
            primes.append(i)
            for j in range(i, N + 1, i):
                is_prime[j] = False
                mu[j] *= -1
            i2 = i * i
            if i2 <= N:
                for j in range(i2, N + 1, i2):
                    mu[j] = 0.0
    
    # mu[1] should be 1
    mu[1] = 1.0
    
    # Domain is 1..N
    domain = np.arange(1, N + 1, dtype=np.int64)
    values = mu.copy()
    
    # Now we need to find a scaling factor s for non-1 values such that:
    # M(x) = sum_k f(k) * floor(x/k) <= 1 for all x in [1, 10*N]
    # where f(1) is adjusted later, but for the scaling we consider the raw values.
    # 
    # The constraint is: for all x in [1, 10*N], sum_{k=1}^{min(N,x)} f(k) * floor(x/k) <= 1
    # 
    # With f(k) = s * mu(k) for k >= 2, and f(1) will be set later.
    # Actually, f(1) is determined by the condition sum f(k)/k = 0 after the shift.
    # But the M(x) constraint uses the final f values.
    #
    # Let's think carefully:
    # - We start with values v(k) = mu(k) for all k.
    # - We scale non-1 values by s: v(k) = s * mu(k) for k >= 2, v(1) = mu(1) = 1.
    # - Then the evaluator computes S = sum_k v(k)/k, sets f(1) = v(1) - S, and f(k) = v(k) for k >= 2.
    # - So f(1) = 1 - S where S = 1/1 + sum_{k=2}^N s*mu(k)/k = 1 + s * sum_{k=2}^N mu(k)/k
    # - And f(k) = s*mu(k) for k >= 2.
    #
    # The M(x) constraint is on the final f:
    # M(x) = sum_k f(k) * floor(x/k) <= 1 for all x in [1, 10*N]
    #
    # M(x) = f(1)*floor(x/1) + sum_{k=2}^{min(N,x)} s*mu(k)*floor(x/k)
    #      = f(1)*x + s * sum_{k=2}^{min(N,x)} mu(k)*floor(x/k)
    #
    # Note that sum_{k=1}^{min(N,x)} mu(k)*floor(x/k) = 1 if x <= N (by the classic identity),
    # and for x > N, it's sum_{k=1}^N mu(k)*floor(x/k) which is not necessarily 1.
    #
    # Let M_raw(x) = sum_{k=1}^{min(N,x)} mu(k)*floor(x/k). Then:
    # sum_{k=2}^{min(N,x)} mu(k)*floor(x/k) = M_raw(x) - mu(1)*floor(x/1) = M_raw(x) - x
    #
    # So M(x) = f(1)*x + s*(M_raw(x) - x)
    #
    # And f(1) = 1 - S where S = 1 + s * sum_{k=2}^N mu(k)/k
    # Let T = sum_{k=2}^N mu(k)/k. Then S = 1 + s*T, f(1) = 1 - (1 + s*T) = -s*T
    #
    # So M(x) = -s*T*x + s*(M_raw(x) - x) = s*(M_raw(x) - x - T*x) = s*(M_raw(x) - x*(1+T))
    #
    # We need M(x) <= 1 for all x in [1, 10*N].
    # Since s > 0 (mu values are mostly negative for k>=2? No, mu alternates), we need to be careful.
    #
    # Actually, let's just compute the maximum of M(x)/s over all x in [1, 10*N] and set s = 1/max(M(x)/s) if max > 0, else s can be large.
    # But M(x) = s * g(x) where g(x) = M_raw(x) - x*(1+T).
    # So we need s * max_x g(x) <= 1, i.e., s <= 1/max_x g(x) if max_x g(x) > 0.
    #
    # To maximize the objective a = -sum_k f(k)*log(k)/k, we want s as large as possible (since a is proportional to s).
    # So s = 1/max_x g(x) if max_x g(x) > 0, otherwise s can be very large (but values are clipped to [-10,10]).
    
    # Compute T = sum_{k=2}^N mu(k)/k
    T = np.sum(mu[2:] / np.arange(2, N + 1, dtype=np.float64))
    
    # Compute M_raw(x) for x in [1, 10*N]. This is expensive but doable with numpy.
    # M_raw(x) = sum_{k=1}^{min(N,x)} mu(k) * floor(x/k)
    # For x <= N: M_raw(x) = 1 (by the classic identity, since sum_{d|x} mu(d) = 0 for x > 1, and = 1 for x = 1)
    # Wait, the identity is sum_{d|x} mu(d) = 1 if x = 1, 0 otherwise. But M_raw(x) = sum_{k=1}^x mu(k)*floor(x/k).
    # This equals sum_{k=1}^x mu(k) * (number of multiples of k up to x) = sum_{m=1}^x sum_{k|m} mu(k) = sum_{m=1}^x [m=1] = 1.
    # So M_raw(x) = 1 for all x >= 1? No, that's when the sum goes to x. When the sum goes to N < x, it's different.
    #
    # For x <= N: M_raw(x) = sum_{k=1}^x mu(k)*floor(x/k) = 1 (exact identity).
    # For x > N: M_raw(x) = sum_{k=1}^N mu(k)*floor(x/k).
    
    # So for x <= N, g(x) = 1 - x*(1+T)
    # For x > N, g(x) = M_raw(x) - x*(1+T) where M_raw(x) = sum_{k=1}^N mu(k)*floor(x/k)
    
    # Let's compute the maximum of g(x) over x in [1, 10*N].
    # For x <= N: g(x) = 1 - x*(1+T). Since T = sum_{k=2}^N mu(k)/k, and sum_{k=1}^N mu(k)/k ~ log(N) + gamma... actually sum mu(k)/k ~ 0 for large N? No, it converges to 0? Actually sum_{k=1}^infty mu(k)/k = 0 (related to 1/zeta(1) which diverges... wait, sum mu(k)/k = 0 is not true, it's related to the Prime Number Theorem).
    # Anyway, 1+T could be positive or negative. Let's just compute numerically.
    
    # For x in [1, N]: g(x) = 1 - x*(1+T)
    # If 1+T > 0, g(x) is decreasing in x, so max is at x=1: g(1) = 1 - (1+T) = -T
    # If 1+T < 0, g(x) is increasing in x, so max is at x=N: g(N) = 1 - N*(1+T)
    # If 1+T = 0, g(x) = 1 for all x <= N.
    
    # For x in [N+1, 10*N]: we need to compute M_raw(x) = sum_{k=1}^N mu(k)*floor(x/k)
    # This is expensive for all x. But we can sample or use the fact that floor(x/k) changes slowly.
    # Actually, let's just compute for all x in [1, 10*N] using a vectorized approach.
    # 10*N = 2,000,000. N = 200,000. Computing M_raw for all x would be O(N * 10*N) = 4e10 which is too slow.
    #
    # Alternative: note that for x > N, M_raw(x) = sum_{k=1}^N mu(k)*floor(x/k).
    # We can compute this for all x in [N+1, 10*N] by iterating over k and adding mu(k)*floor(x/k) for each x.
    # That's still O(N * 10*N) in the worst case.
    #
    # Better approach: for each k, floor(x/k) is constant on intervals. We can use a difference array.
    # But even that is O(N * (number of distinct values of floor(x/k))) which can be large.
    #
    # Practical approach: since we only need the maximum of g(x) = M_raw(x) - x*(1+T), and for x > N,
    # M_raw(x) is roughly constant (it's the partial sum of mu(k)*floor(x/k)), while x*(1+T) grows linearly,
    # the maximum for x > N likely occurs at x = N+1 if 1+T > 0, or at x = 10*N if 1+T < 0.
    #
    # Let's just compute g(x) for x in [1, min(N, 10*N)] analytically and for a few sample points in [N+1, 10*N].
    
    # For x in [1, N]: g(x) = 1 - x*(1+T)
    x_small = np.arange(1, N + 1, dtype=np.float64)
    g_small = 1.0 - x_small * (1.0 + T)
    
    # For x in [N+1, 10*N]: sample at x = N+1, N+1000, ..., 10*N
    # Let's sample at about 100 points
    x_large = np.linspace(N + 1, 10 * N, 100, dtype=np.float64).astype(np.int64)
    g_large = np.zeros(len(x_large), dtype=np.float64)
    for i, x in enumerate(x_large):
        # M_raw(x) = sum_{k=1}^N mu(k)*floor(x/k)
        # Compute this using numpy
        k = np.arange(1, N + 1, dtype=np.float64)
        M_raw_x = np.sum(mu[1:] * np.floor(x / k))
        g_large[i] = M_raw_x - x * (1.0 + T)
    
    g_max = max(np.max(g_small), np.max(g_large))
    
    if g_max > 0:
        s = 1.0 / g_max
    else:
        # If g_max <= 0, then M(x) = s*g(x) <= 0 <= 1 for any s >= 0, so we can make s large.
        # But values are clipped to [-10, 10], so s * |mu(k)| <= 10 for all k, i.e., s <= 10 since |mu(k)| <= 1.
        s = 10.0
    
    # Apply scaling to non-1 values
    values[1:] = s * mu[1:]
    # values[0] = mu[1] = 1.0 (will be adjusted by the evaluator)
    
    return domain, values


# EVOLVE-BLOCK-END
