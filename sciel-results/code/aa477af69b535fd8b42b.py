import time
import math
from fractions import Fraction
import numpy as np

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin for 90s limit

    best_a = 11
    best_b = 6
    best_error = Fraction(1, 3)

    def compute_sigma_numpy(L):
        """Compute sigma(n) for all n from 1 to L using NumPy vectorized sieve."""
        sigma = np.zeros(L + 1, dtype=np.int64)
        for i in range(1, L + 1):
            sigma[i::i] += i
        return sigma

    def search_pairs(sigma, L, best_a, best_b, best_error, deadline, start_time):
        """Given sigma array, find best pair with ratio close to 3/2."""
        # Build groups: map sigma value -> list of n
        # Use a dict for grouping
        groups = {}
        for n in range(1, L + 1):
            s = int(sigma[n])
            if s not in groups:
                groups[s] = [n]
            else:
                groups[s].append(n)

        for s, nums in groups.items():
            if len(nums) < 2:
                continue
            if time.time() - start_time > deadline:
                break

            nums_sorted = sorted(nums)
            n_elems = len(nums_sorted)

            # For each b in nums_sorted, find a in nums_sorted closest to 1.5*b
            # Use two-pointer or binary search
            # For efficiency, iterate and use bisect
            import bisect
            for b in nums_sorted:
                target = 1.5 * b
                # Find insertion point
                pos = bisect.bisect_left(nums_sorted, int(target))
                # Check pos and pos-1
                for idx in [pos, pos - 1]:
                    if 0 <= idx < n_elems:
                        a = nums_sorted[idx]
                        if a == b:
                            continue
                        # Exact error computation
                        num = 2 * a - 3 * b
                        den = 2 * b
                        if num < 0:
                            num = -num
                        # err = num / den
                        # Compare with best_error: num/den < best_num/best_den
                        # => num * best_den < best_num * den
                        if num * best_error.denominator < best_error.numerator * den:
                            best_error = Fraction(num, den)
                            best_a = a
                            best_b = b

        return best_a, best_b, best_error

    # Try progressively larger limits with NumPy
    limits = [10000, 50000, 100000, 200000, 500000, 1000000, 2000000, 5000000]

    for L in limits:
        if time.time() - start_time > deadline - 10:
            break

        try:
            sigma = compute_sigma_numpy(L)
        except MemoryError:
            break

        best_a, best_b, best_error = search_pairs(
            sigma, L, best_a, best_b, best_error, deadline, start_time
        )

        if time.time() - start_time > deadline - 5:
            break

    return [int(best_a), int(best_b)]
