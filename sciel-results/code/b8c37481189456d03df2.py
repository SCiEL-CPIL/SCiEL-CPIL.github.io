import math
import time
import heapq
import numpy as np

def construct():
    N_MIN = 10000
    N_MAX = 1000000
    DEADLINE = 85

    start_time = time.time()
    max_val = N_MAX + 1

    # Sieve for Largest Prime Factor
    lpf = [1] * (max_val + 1)

    for i in range(2, max_val + 1):
        if lpf[i] == 1:
            for j in range(i, max_val + 1, i):
                lpf[j] = i

    elapsed = time.time() - start_time
    if elapsed > DEADLINE:
        return N_MIN

    # Vectorized scan
    lpf_arr = np.array(lpf, dtype=np.int64)
    a = lpf_arr[N_MIN:N_MAX + 1]
    b = lpf_arr[N_MIN + 1:N_MAX + 2]
    vals = np.maximum(a, b)
    best_idx = int(np.argmin(vals))
    best_n = N_MIN + best_idx
    best_val = int(vals[best_idx])

    # Check if we can improve with smooth number search
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - 5:
        return best_n

    # Generate y-smooth numbers for small y and check for consecutive pairs
    # We'll try y = 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47
    # For each y, generate all y-smooth numbers in [N_MIN, N_MAX+1]
    # Then check if any two are consecutive

    # First, let's get the current best value
    # If best_val is already very small, we might not need to search further

    # Generate smooth numbers using a priority queue
    def generate_smooth_numbers(y_bound, lo, hi, time_limit=5):
        """Generate all y-smooth numbers in [lo, hi] using a heap."""
        # Get all primes up to y_bound
        primes = []
        for p in range(2, y_bound + 1):
            is_prime = True
            for q in range(2, int(math.isqrt(p)) + 1):
                if p % q == 0:
                    is_prime = False
                    break
            if is_prime:
                primes.append(p)

        if not primes:
            return []

        # Use a heap to generate smooth numbers in order
        # Start with 1
        heap = [(1, [0] * len(primes))]
        seen = set()
        result = []
        start = time.time()

        while heap:
            val, exponents = heapq.heappop(heap)
            if val in seen:
                continue
            seen.add(val)

            if val > hi:
                break
            if val >= lo:
                result.append(val)

            # Generate next numbers by incrementing each prime's exponent
            for i in range(len(primes)):
                new_exponents = list(exponents)
                new_exponents[i] += 1
                new_val = val * primes[i]
                if new_val <= hi:
                    heapq.heappush(heap, (new_val, new_exponents))

            if time.time() - start > time_limit:
                break

        return result

    # Try different smooth bounds
    smooth_bounds = [5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]

    for y in smooth_bounds:
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 3:
            break

        smooth_nums = generate_smooth_numbers(y, N_MIN, N_MAX + 1, time_limit=3)
        if not smooth_nums:
            continue

        # Check for consecutive pairs
        smooth_set = set(smooth_nums)
        for n in smooth_nums:
            if n + 1 in smooth_set:
                # Found a consecutive pair
                candidate_val = max(lpf_arr[n], lpf_arr[n + 1])
                if candidate_val < best_val:
                    best_val = int(candidate_val)
                    best_n = n
                    break

        # Also check if n-1 is smooth (i.e., n is the second of a pair)
        # Already covered above since we check n and n+1

    elapsed = time.time() - start_time
    if elapsed > DEADLINE - 1:
        return best_n

    return best_n
