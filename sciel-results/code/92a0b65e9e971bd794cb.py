"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from itertools import combinations, product

def are_pairwise_coprime(nums):
    unique = list(set(nums))
    for i in range(len(unique)):
        for j in range(i + 1, len(unique)):
            if gcd(unique[i], unique[j]) != 1:
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill

    # Default fallback
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    best_count = 5

    # --- Sieve primes up to 256 ---
    def sieve(n):
        is_p = [True] * (n + 1)
        is_p[0] = is_p[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_p[i]:
                for j in range(i * i, n + 1, i):
                    is_p[j] = False
        return is_p

    is_prime = sieve(256)

    # --- Phase 1: Exhaustive search for 9 distinct prime sums ---
    # Strategy: iterate over all A triples, for each find B triples where
    # all 9 sums are prime (hence pairwise coprime) and distinct.
    
    found_9 = False
    
    # Precompute for speed: for each a, list of b such that a+b is prime
    prime_b_for_a = {}
    for a in range(1, 129):
        prime_b_for_a[a] = [b for b in range(1, 129) if is_prime[a + b]]
    
    # Small primes for modular filtering
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    for a1 in range(1, 129):
        if time.time() > deadline - 5:
            break
        for a2 in range(a1 + 1, 129):
            if time.time() > deadline - 5:
                break
            # Quick modular check: a2 - a1 should not be 0 mod p for p > 2
            # (otherwise a1+b and a2+b have same residue mod p for all b)
            diff12 = a2 - a1
            skip = False
            for p in small_primes:
                if p > 2 and diff12 % p == 0:
                    skip = True
                    break
            if skip:
                continue
                
            for a3 in range(a2 + 1, 129):
                if time.time() > deadline - 5:
                    break
                
                diff13 = a3 - a1
                diff23 = a3 - a2
                skip = False
                for p in small_primes:
                    if p > 2:
                        if diff13 % p == 0 or diff23 % p == 0:
                            skip = True
                            break
                if skip:
                    continue
                
                # Find b's where all three sums are prime
                valid_b = []
                for b in range(1, 129):
                    if is_prime[a1 + b] and is_prime[a2 + b] and is_prime[a3 + b]:
                        valid_b.append(b)
                
                if len(valid_b) < 3:
                    continue
                
                # Try to find 3 b's giving 9 distinct sums
                # Sums are: a1+b1, a1+b2, a1+b3, a2+b1, a2+b2, a2+b3, a3+b1, a3+b2, a3+b3
                # Distinctness condition:
                # a_i + b_j = a_k + b_l  =>  a_i - a_k = b_l - b_j
                # For fixed A, collisions happen when b differences match A differences.
                # A differences: d12 = a2-a1, d13 = a3-a1, d23 = a3-a2
                # So we need b_j - b_l not in {±d12, ±d13, ±d23} for j != l
                
                forbidden_diffs = set()
                forbidden_diffs.add(diff12)
                forbidden_diffs.add(-diff12)
                forbidden_diffs.add(diff13)
                forbidden_diffs.add(-diff13)
                forbidden_diffs.add(diff23)
                forbidden_diffs.add(-diff23)
                
                # Try all triples from valid_b
                if len(valid_b) <= 20:
                    candidates = valid_b
                else:
                    candidates = valid_b[:20]
                
                for b1, b2, b3 in combinations(candidates, 3):
                    # Check no forbidden differences
                    if abs(b2 - b1) in forbidden_diffs:
                        continue
                    if abs(b3 - b1) in forbidden_diffs:
                        continue
                    if abs(b3 - b2) in forbidden_diffs:
                        continue
                    
                    # All 9 sums are prime (by construction of valid_b) and distinct
                    A = [a1, a2, a3]
                    B = [b1, b2, b3]
                    best_A = A
                    best_B = B
                    best_count = 9
                    found_9 = True
                    break
                if found_9:
                    break
            if found_9:
                break
        if found_9:
            break
    
    if found_9:
        return [best_A, best_B]

    # --- Phase 2: Search for 9 distinct pairwise-coprime (not necessarily prime) sums ---
    # Use a smarter approach: pick A, then greedily pick b's
    
    import random
    rng = random.Random(42)
    
    # Try structured A triples
    for a1 in range(1, 50):
        if time.time() > deadline - 3:
            break
        for a2 in range(a1 + 1, 50):
            if time.time() > deadline - 3:
                break
            for a3 in range(a2 + 1, 50):
                if time.time() > deadline - 3:
                    break
                
                # Try to find 3 b's such that all 9 sums are pairwise coprime
                # For each b, the three sums are a1+b, a2+b, a3+b
                # These must be pairwise coprime with each other and with sums from other b's
                
                valid_b = []
                for b in range(1, 129):
                    s1 = a1 + b
                    s2 = a2 + b
                    s3 = a3 + b
                    # Check if s1, s2, s3 are pairwise coprime
                    if gcd(s1, s2) == 1 and gcd(s1, s3) == 1 and gcd(s2, s3) == 1:
                        valid_b.append((b, s1, s2, s3))
                
                if len(valid_b) < 3:
                    continue
                
                # Try to pick 3 b's such that all 9 sums are pairwise coprime
                # This is complex; try greedy + random
                for _ in range(100):
                    if time.time() > deadline - 3:
                        break
                    sample = rng.sample(valid_b, min(3, len(valid_b)))
                    B = [x[0] for x in sample]
                    sums = []
                    for a in [a1, a2, a3]:
                        for b in B:
                            sums.append(a + b)
                    distinct_sums = set(sums)
                    if len(distinct_sums) == 9 and are_pairwise_coprime(distinct_sums):
                        best_A = [a1, a2, a3]
                        best_B = B
                        best_count = 9
                        return [best_A, best_B]

    # --- Phase 3: Random search for best pairwise-coprime count ---
    attempts = 0
    while time.time() < deadline and attempts < 100000:
        A = sorted(rng.sample(range(1, 129), 3))
        B = sorted(rng.sample(range(1, 129), 3))
        sums = [A[i] + B[j] for i in range(3) for j in range(3)]
        distinct_sums = set(sums)
        count = len(distinct_sums)
        if count > best_count and are_pairwise_coprime(distinct_sums):
            best_count = count
            best_A = A[:]
            best_B = B[:]
            if best_count == 9:
                return [best_A, best_B]
        attempts += 1

    return [best_A, best_B]
# EVOLVE-BLOCK-END
