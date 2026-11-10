"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from itertools import combinations

def are_pairwise_coprime(nums):
    unique = list(set(nums))
    for i in range(len(unique)):
        for j in range(i + 1, len(unique)):
            if gcd(unique[i], unique[j]) != 1:
                return False
    return True

def is_prime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safe margin before 90s kill

    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    best_count = 5

    # Precompute primes up to 256
    primes_set = set(x for x in range(2, 257) if is_prime(x))

    # Precompute "pairwise-coprime-friendly" numbers: primes and prime powers
    # A number is "safe" if it's a prime power (p^k) - any two distinct prime powers
    # of different primes are coprime. So if all distinct sums are prime powers of
    # distinct primes, they're pairwise coprime.
    # But more generally, we can check pairwise coprime directly.

    # Phase 1: Search for all-prime sumsets (strongest guarantee)
    # For a given A, b is valid if a_i + b is prime for all i in A
    for a1 in range(1, 101):
        if time.time() > deadline:
            break
        for a2 in range(a1 + 1, 101):
            if time.time() > deadline:
                break
            for a3 in range(a2 + 1, 101):
                if time.time() > deadline:
                    break
                A = [a1, a2, a3]
                valid_b = []
                for b in range(1, 129):
                    if (a1 + b) in primes_set and (a2 + b) in primes_set and (a3 + b) in primes_set:
                        valid_b.append(b)

                if len(valid_b) < 3:
                    continue

                for b1, b2, b3 in combinations(valid_b, 3):
                    B = [b1, b2, b3]
                    sums = [A[i] + B[j] for i in range(3) for j in range(3)]
                    distinct_sums = set(sums)
                    count = len(distinct_sums)
                    if count > best_count:
                        best_count = count
                        best_A = A[:]
                        best_B = B[:]
                        if best_count == 9:
                            return [best_A, best_B]

    # Phase 2: Broader pairwise-coprime search using structured approach
    # Use numbers that are products of small primes to create diverse sum values
    # Try to find A, B where all 9 sums are distinct and pairwise coprime

    # Strategy: Use A = [a1, a2, a3] and B = [b1, b2, b3]
    # We want all 9 sums a_i + b_j to be pairwise coprime
    # This is very restrictive. Let's try a more targeted approach.

    # Approach: Pick 9 target numbers that are pairwise coprime, then try to
    # factor them into a_i + b_j form.
    # Simpler: enumerate A from a smaller set, then for each A, find B values
    # that make sums pairwise coprime.

    # Precompute a set of "good" numbers: primes and prime powers up to 256
    good_numbers = set()
    for p in range(2, 257):
        if is_prime(p):
            good_numbers.add(p)
            pk = p
            while pk * p <= 256:
                pk *= p
                good_numbers.add(pk)

    # For phase 2, try A values from a curated set and search for B
    # that makes all sums pairwise coprime
    # Use a time-bounded random + structured hybrid

    import random
    rng = random.Random(12345)

    # Structured: try A with small values
    structured_A_candidates = []
    for a1 in range(1, 30):
        for a2 in range(a1 + 1, 30):
            for a3 in range(a2 + 1, 30):
                structured_A_candidates.append([a1, a2, a3])

    # For each A candidate, find B values where all 9 sums are pairwise coprime
    # This is expensive, so limit the search
    for A in structured_A_candidates:
        if time.time() > deadline:
            break

        # For each b in [1,128], check if it can be part of a valid B
        # b is "compatible" with A if for all a in A, a+b is pairwise coprime
        # with all other sums... this is complex. Instead, just try B candidates.

        # Precompute for each b: the set of sums {a+b for a in A}
        b_sums = {}
        for b in range(1, 129):
            b_sums[b] = [A[0] + b, A[1] + b, A[2] + b]

        # Try triples of b values
        # To speed up, first filter: for two b values b1, b2, the 6 sums
        # b_sums[b1] + b_sums[b2] must be pairwise coprime
        # Then add a third b3.

        # This is still O(128^3) which is ~2M, too slow.
        # Instead, use a greedy approach: pick b1, then find b2 compatible, then b3.

        # Actually, let's just do a limited random search combined with the prime approach
        # for phase 2.

    # Phase 2: Random search with pairwise coprime check
    attempts = 0
    while time.time() < deadline and attempts < 500000:
        # Mix of random and structured
        if attempts % 10 == 0:
            # Structured: small A, random B
            A = sorted(rng.sample(range(1, 40), 3))
            B = sorted(rng.sample(range(1, 129), 3))
        else:
            A = sorted(rng.sample(range(1, 129), 3))
            B = sorted(rng.sample(range(1, 129), 3))

        sums = [A[i] + B[j] for i in range(3) for j in range(3)]
        distinct_sums = set(sums)
        count = len(distinct_sums)

        if count > best_count and count >= 7:
            if are_pairwise_coprime(distinct_sums):
                best_count = count
                best_A = A[:]
                best_B = B[:]
                if best_count == 9:
                    return [best_A, best_B]
        attempts += 1

    # Phase 3: Targeted search for 9 distinct pairwise coprime sums
    # Try to find A, B such that all 9 sums are distinct primes (or prime powers)
    # Use a different A range that wasn't fully explored

    if time.time() < deadline - 5:
        # Try A with larger values
        for a1 in range(50, 101):
            if time.time() > deadline - 5:
                break
            for a2 in range(a1 + 1, 101):
                if time.time() > deadline - 5:
                    break
                for a3 in range(a2 + 1, 101):
                    if time.time() > deadline - 5:
                        break
                    A = [a1, a2, a3]
                    valid_b = []
                    for b in range(1, 129):
                        if (a1 + b) in primes_set and (a2 + b) in primes_set and (a3 + b) in primes_set:
                            valid_b.append(b)

                    if len(valid_b) < 3:
                        continue

                    for b1, b2, b3 in combinations(valid_b, 3):
                        B = [b1, b2, b3]
                        sums = [A[i] + B[j] for i in range(3) for j in range(3)]
                        distinct_sums = set(sums)
                        count = len(distinct_sums)
                        if count > best_count:
                            best_count = count
                            best_A = A[:]
                            best_B = B[:]
                            if best_count == 9:
                                return [best_A, best_B]

    return [best_A, best_B]
# EVOLVE-BLOCK-END
