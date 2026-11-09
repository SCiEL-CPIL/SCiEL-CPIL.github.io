import math
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin for 90s limit

    # Sieve primes up to 256 (max sum = 128+128)
    max_sum = 256
    is_prime_arr = [True] * (max_sum + 1)
    is_prime_arr[0] = False
    is_prime_arr[1] = False
    for i in range(2, int(math.isqrt(max_sum)) + 1):
        if is_prime_arr[i]:
            for j in range(i * i, max_sum + 1, i):
                is_prime_arr[j] = False

    def is_prime(n):
        return 0 <= n <= max_sum and is_prime_arr[n]

    def is_pairwise_coprime(nums):
        if len(nums) < 2:
            return True
        nums = list(nums)
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                if math.gcd(nums[i], nums[j]) != 1:
                    return False
        return True

    def evaluate(A, B):
        sums = set()
        for a in A:
            for b in B:
                sums.add(a + b)
        if len(sums) == 0:
            return -1
        if not is_pairwise_coprime(sums):
            return -1
        return len(sums)

    # Precompute for each a in [1,128] the set of b in [1,128] where a+b is prime
    valid_b_for_a = []
    for a in range(129):
        vbs = set()
        for b in range(1, 129):
            if is_prime_arr[a + b]:
                vbs.add(b)
        valid_b_for_a.append(vbs)

    # Search for A, B where all 9 sums are prime (hence pairwise coprime)
    # Iterate over all 3-element subsets of [1..128] for A
    # For each A, find valid b's (intersection of valid_b_for_a for each a in A)
    # If >= 3 valid b's, try to find 3 that give 9 distinct sums

    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    best_score = evaluate(best_A, best_B)

    # Generate all 3-element subsets
    all_A = list(combinations(range(1, 129), 3))

    for idx, A in enumerate(all_A):
        if time.time() > deadline - 10:
            break

        a1, a2, a3 = A

        # Intersection of valid b's
        valid_bs = valid_b_for_a[a1] & valid_b_for_a[a2] & valid_b_for_a[a3]

        if len(valid_bs) < 3:
            continue

        # Sort valid b's and try combinations to get 9 distinct sums
        valid_bs_sorted = sorted(valid_bs)
        n = len(valid_bs_sorted)

        # Try all triples if n is small, otherwise use a smarter strategy
        if n <= 30:
            triples_to_try = list(combinations(range(n), 3))
        else:
            # Use a greedy approach: try to pick b's that spread out the sums
            # First, try the first 50, last 50, and some in the middle
            indices = list(range(min(50, n))) + list(range(max(0, n-50), n))
            indices = sorted(set(indices))
            if n > 100:
                mid = n // 2
                indices.extend(range(max(0, mid-25), min(n, mid+25)))
                indices = sorted(set(indices))
            triples_to_try = list(combinations(indices, 3))

        for i_idx, j_idx, k_idx in triples_to_try:
            if time.time() > deadline - 5:
                break
            B = [valid_bs_sorted[i_idx], valid_bs_sorted[j_idx], valid_bs_sorted[k_idx]]
            # Check if all 9 sums are distinct
            sums = set()
            for a in A:
                for b in B:
                    sums.add(a + b)
            if len(sums) == 9:
                # All prime, all distinct => optimal
                return [list(A), B]

    # Fallback: randomized search for pairwise-coprime sums (not necessarily all prime)
    import random
    random.seed(42)

    prime_candidates = [i for i in range(1, 129) if is_prime_arr[i]]

    for trial in range(200000):
        if time.time() > deadline:
            break

        # Mix strategies
        if trial % 4 == 0:
            A = list(random.sample(range(1, 129), 3))
            B = list(random.sample(range(1, 129), 3))
        elif trial % 4 == 1:
            A = list(random.sample(prime_candidates, 3))
            B = list(random.sample(range(1, 129), 3))
        elif trial % 4 == 2:
            A = list(random.sample(range(1, 129), 3))
            B = list(random.sample(prime_candidates, 3))
        else:
            # Bias toward small values to keep sums in prime-friendly range
            A = list(random.sample(range(1, 65), 3))
            B = list(random.sample(range(1, 65), 3))

        score = evaluate(A, B)
        if score > best_score:
            best_score = score
            best_A = A
            best_B = B
            if score == 9:
                return [A, B]

    return [best_A, best_B]
