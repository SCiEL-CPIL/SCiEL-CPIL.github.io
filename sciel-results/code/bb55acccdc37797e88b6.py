"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from math import gcd
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 85.0

    def time_exceeded():
        return time.time() - start_time > deadline - 2.0

    # Precompute prime factors for all possible sums (2..20000)
    MAX_SUM = 20000
    pf = [set() for _ in range(MAX_SUM + 1)]
    for p in range(2, MAX_SUM + 1):
        if not pf[p]:  # p is prime
            for m in range(p, MAX_SUM + 1, p):
                pf[m].add(p)

    def count_distinct_primes(arr):
        """Count distinct prime factors across all pairwise sums."""
        primes = set()
        n = len(arr)
        for i in range(n):
            for j in range(i + 1, n):
                primes |= pf[arr[i] + arr[j]]
        return len(primes)

    def get_primes_for(arr):
        """Return the set of primes for a given array."""
        primes = set()
        n = len(arr)
        for i in range(n):
            for j in range(i + 1, n):
                primes |= pf[arr[i] + arr[j]]
        return primes

    def is_smooth(n, prime_set):
        """Check if n has only prime factors in prime_set."""
        if n <= 1:
            return True
        for p in prime_set:
            while n % p == 0:
                n //= p
        return n == 1

    def try_greedy_for_prime_set(prime_set, max_val=10000):
        """Greedy: build set of 12 numbers where all pairwise sums are smooth."""
        candidates = []
        for x in range(1, max_val + 1):
            if time_exceeded():
                return None
            valid = True
            for y in candidates:
                if not is_smooth(x + y, prime_set):
                    valid = False
                    break
            if valid:
                candidates.append(x)
            if len(candidates) >= 12:
                return candidates[:12]
        return None

    # Strategy 1: Powers of 2 (structural)
    powers_of_2 = [2**i for i in range(12)]
    assert all(x <= 10000 for x in powers_of_2)
    assert len(set(powers_of_2)) == 12

    best_result = list(powers_of_2)
    best_count = count_distinct_primes(best_result)

    # Strategy 2: Try greedy for small prime sets
    prime_set_candidates = [
        {2},
        {2, 3},
        {3, 5},
        {2, 3, 5},
        {5, 7},
        {2, 5},
        {3, 7},
        {2, 3, 5, 7},
    ]

    for ps in prime_set_candidates:
        if time_exceeded():
            break
        result = try_greedy_for_prime_set(ps)
        if result is not None and len(result) == 12:
            if len(set(result)) == 12 and all(1 <= x <= 10000 for x in result):
                cnt = count_distinct_primes(result)
                if cnt < best_count:
                    best_count = cnt
                    best_result = list(result)
                if best_count <= len(ps):
                    break

    # Strategy 3: Structured sets
    # All odd numbers in a range, sums are even
    odds = list(range(1, 24, 2))[:12]
    cnt = count_distinct_primes(odds)
    if cnt < best_count:
        best_count = cnt
        best_result = list(odds)

    # Multiples of 6
    mult6 = [6*(i+1) for i in range(12)]
    cnt = count_distinct_primes(mult6)
    if cnt < best_count:
        best_count = cnt
        best_result = list(mult6)

    # Close numbers
    close = list(range(100, 112))
    cnt = count_distinct_primes(close)
    if cnt < best_count:
        best_count = cnt
        best_result = list(close)

    # Small numbers
    small = list(range(1, 13))
    cnt = count_distinct_primes(small)
    if cnt < best_count:
        best_count = cnt
        best_result = list(small)

    # Strategy 4: Local search with simulated annealing
    # Use the precomputed prime factor cache for fast evaluation
    def local_search(arr, max_iters=50000):
        """Local search: try replacing elements to reduce prime count."""
        arr = list(arr)
        current_count = count_distinct_primes(arr)
        current_primes = get_primes_for(arr)
        best_arr = list(arr)
        best_cnt = current_count
        rng = random.Random(42)

        for it in range(max_iters):
            if time_exceeded():
                break
            # Pick a random element to replace
            idx = rng.randint(0, 11)
            old_val = arr[idx]
            new_val = rng.randint(1, 10000)
            if new_val == old_val:
                continue
            # Check distinctness
            if new_val in arr and arr.index(new_val) != idx:
                continue

            # Evaluate change
            arr[idx] = new_val
            new_primes = get_primes_for(arr)
            new_count = len(new_primes)

            if new_count < current_count:
                current_count = new_count
                current_primes = new_primes
                if new_count < best_cnt:
                    best_cnt = new_count
                    best_arr = list(arr)
            else:
                # Accept with probability based on simulated annealing
                # Simple: only accept if equal (to explore plateaus)
                if new_count == current_count and rng.random() < 0.3:
                    current_primes = new_primes
                else:
                    arr[idx] = old_val
                    continue

        return best_arr, best_cnt

    # Multiple diverse initializations for local search
    initializations = []

    # 1. Powers of 2
    initializations.append(list(powers_of_2))
    # 2. Small numbers
    initializations.append(list(range(1, 13)))
    # 3. Close numbers around different centers
    for center in [50, 100, 200, 500, 1000]:
        init = [center + i for i in range(12)]
        if all(1 <= x <= 10000 for x in init):
            initializations.append(init)
    # 4. Multiples of various numbers
    for m in [2, 3, 4, 5, 6, 7, 10]:
        init = [m * (i + 1) for i in range(12)]
        if all(1 <= x <= 10000 for x in init):
            initializations.append(init)
    # 5. Random initializations
    for _ in range(5):
        init = random.sample(range(1, 10001), 12)
        initializations.append(init)

    for init in initializations:
        if time_exceeded():
            break
        result_arr, result_cnt = local_search(init, max_iters=20000)
        if result_cnt < best_count:
            best_count = result_cnt
            best_result = result_arr
        # Also try a second pass from the best found
        if result_cnt == best_count and not time_exceeded():
            result_arr2, result_cnt2 = local_search(result_arr, max_iters=10000)
            if result_cnt2 < best_count:
                best_count = result_cnt2
                best_result = result_arr2

    # Strategy 5: Final polish - try targeted replacements on best solution
    if not time_exceeded():
        best_arr = list(best_result)
        current_primes = get_primes_for(best_arr)
        current_count = len(current_primes)
        rng = random.Random(123)

        for _ in range(30000):
            if time_exceeded():
                break
            idx = rng.randint(0, 11)
            old_val = best_arr[idx]
            # Try values near the current value and some random ones
            if rng.random() < 0.5:
                new_val = old_val + rng.randint(-20, 20)
            else:
                new_val = rng.randint(1, 10000)

            if new_val < 1 or new_val > 10000 or new_val == old_val:
                continue
            if new_val in best_arr and best_arr.index(new_val) != idx:
                continue

            best_arr[idx] = new_val
            new_primes = get_primes_for(best_arr)
            new_count = len(new_primes)

            if new_count < current_count:
                current_count = new_count
                current_primes = new_primes
                if current_count < best_count:
                    best_count = current_count
                    best_result = list(best_arr)
            else:
                best_arr[idx] = old_val

    return best_result
# EVOLVE-BLOCK-END
