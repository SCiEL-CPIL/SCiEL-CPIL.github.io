# EVOLVE-BLOCK-START
import numpy as np
import time


def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    """
    start_time = time.time()
    deadline = 600.0  # seconds total
    safety_margin = 30.0  # return by 570s

    def elapsed_ok():
        return (time.time() - start_time) < (deadline - safety_margin)

    # We will search over subsets of positive integers up to some max K.
    # The objective is a = -sum_k f(k) * log(k) / k, with constraints:
    #   1. sum_k f(k)/k = 0 (after f(1) adjustment)
    #   2. M(x) = sum_k f(k) * floor(x/k) <= 1.0001 for all x in [1, 10K]
    #
    # The Mobius function mu(k) is the "ideal" solution:
    #   - sum_{d|n} mu(d) = 1 if n=1 else 0, which gives sum mu(k) floor(x/k) = 1 for x>=1
    #   - -sum mu(k) log(k)/k -> 1 as K -> infinity
    #
    # Strategy: use the exact Mobius function mu(k) for k = 1..K.
    # For finite K, M(x) = sum_{k=1}^{K} mu(k) floor(x/k) is not exactly 1 for all x,
    # but it should be close. We need to verify and possibly adjust.
    #
    # Actually, the identity sum_{k=1}^{infty} mu(k) floor(x/k) = 0 for x > 1 (in the limit),
    # and = 1 for x = 1. For finite K, there will be deviations.
    #
    # Let me reconsider. The standard identity is:
    #   sum_{d|n} mu(d) = 1 if n=1, 0 otherwise
    #   => sum_{k=1}^{n} mu(k) floor(n/k) = 1 for all n >= 1
    #
    # So if we use mu(k) for all k up to K, then M(x) = 1 for x <= K, but for x > K,
    # M(x) = sum_{k=1}^{K} mu(k) floor(x/k) which is NOT necessarily 1.
    #
    # Wait, the identity holds for ALL positive integers x, not just x <= K.
    # sum_{k=1}^{infty} mu(k) floor(x/k) = 1 for x >= 1
    # For finite K, M(x) = sum_{k=1}^{K} mu(k) floor(x/k) = 1 - sum_{k=K+1}^{infty} mu(k) floor(x/k)
    #
    # The tail sum is small for large K. For x <= 10K, the tail involves k > K, and
    # floor(x/k) <= floor(10K/K) = 10, so the tail is bounded.
    #
    # Let's just compute mu(k) for k=1..K and verify M(x) <= 1.0001.

    def compute_mobius(K):
        """Compute the Mobius function mu(n) for n = 1..K using a sieve."""
        mu = np.ones(K + 1, dtype=np.int64)
        mu[0] = 0
        primes = []
        is_prime = np.ones(K + 1, dtype=bool)
        is_prime[0] = is_prime[1] = False

        for i in range(2, K + 1):
            if is_prime[i]:
                primes.append(i)
                for p in primes:
                    if i * p > K:
                        break
                    is_prime[i * p] = False
                    if i % p == 0:
                        mu[i * p] = 0  # i*p has p^2 as factor
                        break
                    else:
                        mu[i * p] = -mu[i]

        return mu[1:]  # mu[1], mu[2], ..., mu[K]

    def verify_M(f_values, K):
        """
        Verify that M(x) = sum_k f(k) * floor(x/k) <= 1.0001 for all x in [1, 10K].
        Returns True if feasible, False otherwise.
        """
        max_x = 10 * K
        # M(x) = sum_{k=1}^{K} f(k) * floor(x/k)
        # For each k, f(k) * floor(x/k) changes when x crosses multiples of k.
        # We can compute M(x) for all x = 1..max_x using a difference array.
        #
        # For a given k, f(k) * floor(x/k) is:
        #   f(k) * 0 for x = 0
        #   f(k) * 1 for x in [1, k-1] ... wait, floor(x/k) for x in [1, k-1] is 0 if x < k
        #   Actually floor(x/k): for x in [1, k-1], floor(x/k) = 0 (if x < k)
        #   For x in [k, 2k-1], floor(x/k) = 1
        #   For x in [2k, 3k-1], floor(x/k) = 2
        #   etc.
        #
        # So f(k) contributes f(k) * j to M(x) for x in [j*k, (j+1)*k - 1] where j >= 1.
        #
        # Use a difference array: for each k, add f(k) to the range [k, 2k-1],
        # 2*f(k) to [2k, 3k-1], etc.
        #
        # This is O(K * max_x / K) = O(max_x) per k, total O(K * max_x) which is too slow.
        #
        # Alternative: direct computation. M(x) = sum_{k=1}^{min(K,x)} f(k) * (x // k)
        # For each x, this is O(min(K,x)). Total O(sum_{x=1}^{10K} min(K,x)) = O(K * 10K) = O(10 K^2).
        # For K = 10000, that's 10^9 operations, too slow in Python.
        #
        # Let's use numpy for vectorization where possible, but this is inherently O(K^2).
        #
        # Actually, let's just check a subset of x values for speed, and if K is small enough,
        # check all.
        if max_x <= 200000:
            # Compute M(x) for all x using vectorized approach
            # M(x) = sum_{k=1}^{K} f(k) * floor(x/k)
            # For x from 1 to max_x:
            xs = np.arange(1, max_x + 1, dtype=np.int64)
            # For each k, compute floor(xs/k) and multiply by f(k), sum over k
            # This is O(K * max_x) which for K=20000, max_x=200000 is 4*10^9, too slow.
            #
            # Let's use a smarter approach: group by floor(x/k) values.
            # For fixed x, floor(x/k) takes O(sqrt(x)) distinct values.
            # But we need M(x) for all x, not just one x.
            #
            # Alternative: for each k, the function g_k(x) = f(k) * floor(x/k) is a step function.
            # We can accumulate using a difference array on the step points.
            #
            # For k, the steps are at x = k, 2k, 3k, ... up to max_x.
            # g_k(x) increases by f(k) at each multiple of k.
            # So we can do: for each k, for j = 1, 2, ..., floor(max_x/k):
            #   diff[j*k] += f(k)
            # Then prefix sum gives M(x) for x >= 1 (with M(0) = 0).
            diff = np.zeros(max_x + 1, dtype=np.float64)
            for k_idx in range(K):
                k = k_idx + 1
                fk = f_values[k_idx]
                if fk == 0:
                    continue
                # Add fk at positions k, 2k, 3k, ... up to max_x
                multiples = np.arange(k, max_x + 1, k)
                diff[multiples] += fk
            M_vals = np.cumsum(diff)
            # M_vals[x] = M(x) for x = 1..max_x
            if np.any(M_vals > 1.0001):
                return False
            return True
        else:
            # For large K, just check sampled x values
            # Check x = 1, 2, ..., min(10000, 10K) and some larger values
            check_xs = set()
            for x in range(1, min(10000, max_x + 1)):
                check_xs.add(x)
            for x in range(10000, max_x + 1, max(1, (max_x - 10000) // 1000)):
                check_xs.add(x)
            check_xs = sorted(check_xs)
            for x in check_xs:
                # M(x) = sum_{k=1}^{min(K,x)} f(k) * (x // k)
                lim = min(K, x)
                M_x = np.sum(f_values[:lim] * (x // np.arange(1, lim + 1)))
                if M_x > 1.0001:
                    return False
            return True

    def compute_objective(f_values, domain):
        """Compute a = -sum_k f(k) * log(k) / k."""
        logs = np.log(domain.astype(np.float64))
        return -np.sum(f_values * logs / domain.astype(np.float64))

    # Start with a small K and increase
    best_score = -np.inf
    best_result = None

    # Try K = 1, 2, 5, 10, 50, 100, 500, 1000, 5000, 10000
    K_candidates = [1, 2, 5, 10, 50, 100, 500, 1000, 5000, 10000, 20000, 50000, 100000, 200000]

    for K in K_candidates:
        if not elapsed_ok():
            break

        try:
            mu = compute_mobius(K)
            domain = np.arange(1, K + 1, dtype=np.int64)
            f_values = mu.astype(np.float64)

            # Verify feasibility
            if not verify_M(f_values, K):
                # Try with a smaller effective K or skip
                continue

            score = compute_objective(f_values, domain)
            if score > best_score:
                best_score = score
                best_result = (domain.copy(), f_values.copy())
        except Exception:
            continue

    if best_result is not None:
        return best_result

    # Fallback: the seed program
    domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
    values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
    return domain, values


# EVOLVE-BLOCK-END
