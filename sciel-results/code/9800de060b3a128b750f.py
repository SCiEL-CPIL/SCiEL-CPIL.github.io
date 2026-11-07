import math
import time
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit

    # Strategy: Use sieve to find k*sigma(k) collisions
    # Then verify and return the best M found

    # Try multiple N values, starting large
    # Use efficient sieve: sigma via divisor sum
    best_result = None
    best_count = 0

    # Try N values that fit in time
    # For N=50M, sigma sieve takes ~2-3s, unique takes ~1-2s
    # We can try N=50M first, then fallback to smaller if needed

    for N in [50_000_000, 40_000_000, 30_000_000, 20_000_000, 15_000_000, 10_000_000]:
        if time.time() > deadline - 20:
            break

        try:
            # Efficient sigma sieve using smallest prime factor
            # Or use the direct divisor sum approach
            # For large N, use a segmented approach or optimized sieve

            # Method: Compute sigma using the formula
            # sigma(n) = product over prime factors p^a || n of (p^(a+1)-1)/(p-1)
            # But for sieve, the divisor sum is simpler

            # Use numpy for fast divisor sum
            # sigma[d::d] += d is O(N log N) which is ~10^8 for N=50M
            # This might be too slow. Let's use a different approach.

            # Alternative: Use the fact that we only need to find collisions
            # We can compute k*sigma(k) for k in range and use a hash map

            # For very large N, use a probabilistic approach:
            # Sample k values and look for collisions

            # Actually, let's use a smarter sieve:
            # Compute spf (smallest prime factor) first, then sigma

            if N <= 30_000_000:
                # Direct divisor sum sieve - fast enough for N <= 30M
                sigma = np.zeros(N + 1, dtype=np.int64)
                for d in range(1, N + 1):
                    sigma[d::d] += d

                if time.time() > deadline - 15:
                    continue

                k_arr = np.arange(1, N + 1, dtype=np.int64)
                M_arr = k_arr * sigma[1:]

                # Find most frequent M
                unique_M, counts = np.unique(M_arr, return_counts=True)
                max_count = int(np.max(counts))

                if max_count > best_count:
                    best_count = max_count
                    best_M_idx = int(np.argmax(counts))
                    best_M = int(unique_M[best_M_idx])

                    # Get all k that map to best_M
                    indices = np.where(M_arr == best_M)[0]
                    k_sols_np = k_arr[indices]
                    k_sols = [int(k) for k in k_sols_np]

                    # Verify
                    limit = math.isqrt(best_M)
                    valid_sols = []
                    for k in k_sols:
                        if k <= limit:
                            s = _sigma_fast(k)
                            if k * s == best_M:
                                valid_sols.append(k)

                    if len(valid_sols) > 0:
                        best_result = {'M': best_M, 'k': valid_sols}
                        best_count = len(valid_sols)

        except Exception:
            continue

    # If we found a good result, return it
    if best_result and best_count >= 3:
        return best_result

    # Fallback: Use known constructions or smaller search
    # Try a targeted search for known collision structures

    # Known: M = 333312 has 3 solutions: 336, 372, 434
    # Let's try to find 4-way collisions with a more targeted approach

    # Use the construction strategy: M = H * P
    # where H is highly composite, P is prime
    # Find k | H such that sigma(k) = P * (H/k)

    # For simplicity, try small highly composite H and check
    # H = 2^a * 3^b * 5^c * 7^d ...
    # We want multiple k | H with k*sigma(k) = M

    # Alternative: brute force small k and look for collisions
    # For k up to 5M, compute k*sigma(k) and find collisions

    if time.time() < deadline - 5:
        N_small = 5_000_000
        try:
            sigma = np.zeros(N_small + 1, dtype=np.int64)
            for d in range(1, N_small + 1):
                sigma[d::d] += d

            k_arr = np.arange(1, N_small + 1, dtype=np.int64)
            M_arr = k_arr * sigma[1:]

            unique_M, counts = np.unique(M_arr, return_counts=True)
            max_count = int(np.max(counts))

            if max_count >= 3:
                best_M_idx = int(np.argmax(counts))
                best_M = int(unique_M[best_M_idx])
                indices = np.where(M_arr == best_M)[0]
                k_sols = [int(k) for k in k_arr[indices]]

                limit = math.isqrt(best_M)
                valid_sols = []
                for k in k_sols:
                    if k <= limit:
                        s = _sigma_fast(k)
                        if k * s == best_M:
                            valid_sols.append(k)

                if len(valid_sols) >= 3:
                    return {'M': best_M, 'k': valid_sols}

        except Exception:
            pass

    # Final fallback
    return {'M': 333312, 'k': [336, 372, 434]}


def _sigma_fast(n):
    """Compute sigma(n) by trial division, optimized."""
    if n == 1:
        return 1
    result = 1
    temp = n
    # Check 2
    if temp % 2 == 0:
        exp = 0
        while temp % 2 == 0:
            temp //= 2
            exp += 1
        result *= (2 ** (exp + 1) - 1)
    # Check odd primes
    p = 3
    while p * p <= temp:
        if temp % p == 0:
            exp = 0
            while temp % p == 0:
                temp //= p
                exp += 1
            result *= (p ** (exp + 1) - 1) // (p - 1)
        p += 2
    if temp > 1:
        result *= (temp * temp - 1) // (temp - 1)
    return result
