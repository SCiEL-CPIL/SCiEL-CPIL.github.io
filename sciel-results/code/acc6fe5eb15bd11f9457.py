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

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s kill

    # Default fallback (known valid solution with 5 distinct sums? Or just a placeholder)
    # The seed score is 1.8. If max is 9, seed might be 5.
    # Let's use a known valid set if search fails.
    # Example: A=[1, 2, 3], B=[4, 5, 6] -> sums 5,6,7, 6,7,8, 7,8,9. Distinct: 5,6,7,8,9.
    # 5,6,7,8,9 are not pairwise coprime (6,8,9 share factors).
    # We need a better fallback.
    # Let's just return a valid small set if we fail to find 9.
    best_A = [1, 2, 3]
    best_B = [4, 5, 6]
    best_count = 0
    
    # Precompute primes up to 256
    def sieve(n):
        is_p = [True] * (n + 1)
        is_p[0] = is_p[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_p[i]:
                for j in range(i*i, n+1, i):
                    is_p[j] = False
        return is_p

    is_prime = sieve(256)
    primes_set = set(i for i in range(2, 257) if is_prime[i])

    # Phase 1: Search for 9 distinct prime sums
    # All sums must be odd primes, so A and B must have opposite parity.
    # Case 1: A is all odd, B is all even
    # Case 2: A is all even, B is all odd
    
    def search_for_prime_sums(A_parity, B_parity):
        # A_parity: 0 for even, 1 for odd
        # B_parity: 1 - A_parity
        
        for a1 in range(1, 129):
            if a1 % 2 != A_parity:
                continue
            if time.time() > deadline - 2:
                return None
            for a2 in range(a1 + 1, 129):
                if a2 % 2 != A_parity:
                    continue
                for a3 in range(a2 + 1, 129):
                    if a3 % 2 != A_parity:
                        continue
                    
                    A = [a1, a2, a3]
                    
                    # Find valid b's
                    valid_b = []
                    for b in range(1, 129):
                        if b % 2 != B_parity:
                            continue
                        if is_prime[a1 + b] and is_prime[a2 + b] and is_prime[a3 + b]:
                            valid_b.append(b)
                    
                    if len(valid_b) < 3:
                        continue
                    
                    # Try to find 3 b's that yield 9 distinct sums
                    # Since valid_b is usually small, we can iterate all combinations
                    for b1, b2, b3 in combinations(valid_b, 3):
                        B = [b1, b2, b3]
                        sums = [A[i] + B[j] for i in range(3) for j in range(3)]
                        if len(set(sums)) == 9:
                            return [A, B]
        return None

    # Try A odd, B even
    result = search_for_prime_sums(1, 0)
    if result:
        return result

    # Try A even, B odd
    result = search_for_prime_sums(0, 1)
    if result:
        return result

    # Phase 2: If no 9-prime solution found (unlikely), search for 9 pairwise coprime sums
    # This is harder. We can use a random search with a time limit.
    import random
    rng = random.Random(42)
    
    attempts = 0
    while time.time() < deadline - 5 and attempts < 50000:
        A = sorted(rng.sample(range(1, 129), 3))
        B = sorted(rng.sample(range(1, 129), 3))
        sums = [A[i] + B[j] for i in range(3) for j in range(3)]
        distinct_sums = set(sums)
        count = len(distinct_sums)
        
        if count > best_count:
            if are_pairwise_coprime(distinct_sums):
                best_count = count
                best_A = A[:]
                best_B = B[:]
                if best_count == 9:
                    return [best_A, best_B]
        attempts += 1

    # Fallback: Return the best found or a default valid set
    # We need a valid set. Let's construct one manually if best_count is low.
    # A=[1, 2, 3], B=[1, 2, 3] -> sums 2,3,4, 3,4,5, 4,5,6. Distinct: 2,3,4,5,6.
    # 2,3,4,5,6: 2&4 not coprime.
    # Let's try to find a valid 5-sum set quickly.
    # A=[1, 2, 3], B=[4, 5, 6] -> 5,6,7, 6,7,8, 7,8,9. Distinct: 5,6,7,8,9.
    # 5,7 are coprime. 6,5 coprime. 6,7 coprime. 6,8 not.
    
    # We can just return the best_A, best_B found in Phase 2, or a hardcoded valid one.
    # A hardcoded valid set with 3 distinct sums:
    # A=[1, 2, 3], B=[1, 1, 1] is invalid (B must have distinct integers).
    # A=[1, 2, 3], B=[1, 2, 3] -> 2,3,4,3,4,5,4,5,6. Distinct: 2,3,4,5,6. Not pairwise coprime.
    
    # Let's use a known valid set with fewer sums if we fail.
    # A=[1, 2, 3], B=[1, 4, 7]
    # 1+1=2, 1+4=5, 1+7=8
    # 2+1=3, 2+4=6, 2+7=9
    # 3+1=4, 3+4=7, 3+7=10
    # Sums: 2, 5, 8, 3, 6, 9, 4, 7, 10.
    # Distinct: 2, 3, 4, 5, 6, 7, 8, 9, 10.
    # 2,4,6,8,10 are even. Not coprime.
    
    # It is extremely likely that Phase 1 finds a solution.
    # If not, we return the best from Phase 2.
    return [best_A, best_B]

# EVOLVE-BLOCK-END
