import time
import math
import random
from itertools import combinations, product

def construct():
    start_time = time.time()
    deadline = start_time + 85

    # Precompute prime factor masks for numbers 1..256
    primes = []
    is_prime = [True] * 257
    is_prime[0] = is_prime[1] = False
    for i in range(2, 257):
        if is_prime[i]:
            primes.append(i)
            for j in range(i * i, 257, i):
                is_prime[j] = False

    prime_to_idx = {p: i for i, p in enumerate(primes)}

    pf_mask = [0] * 257
    for n in range(1, 257):
        mask = 0
        for p in primes:
            if n % p == 0:
                mask |= (1 << prime_to_idx[p])
        pf_mask[n] = mask

    def is_coprime(a, b):
        return (pf_mask[a] & pf_mask[b]) == 0

    def is_pairwise_coprime(nums):
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                if not is_coprime(nums[i], nums[j]):
                    return False
        return True

    def check(A, B):
        sums = [a + b for a in A for b in B]
        distinct_sums = list(set(sums))
        if len(distinct_sums) != 9:
            return len(distinct_sums)
        if not is_pairwise_coprime(distinct_sums):
            return 0
        return 9

    # Key insight: all sums must be pairwise coprime.
    # If we can make all 9 sums be distinct primes, they are automatically pairwise coprime.
    # For all sums to be odd primes: one set all-odd, other all-even.
    # Strategy: A = all odd, B = all even, and a+b is prime for all 9 pairs.
    
    # Precompute primes up to 256
    primes_set = set(primes)
    
    def all_sums_prime(A, B):
        for a in A:
            for b in B:
                if a + b > 256 or (a + b) not in primes_set:
                    return False
        return True

    def find_valid_pair():
        # Try A all-odd, B all-even
        # For each triple of odd numbers A, check if there exist 3 even numbers B
        # such that all a+b are prime.
        
        # Precompute for each odd a and even b whether a+b is prime
        # odd numbers in [1,128]
        odds = list(range(1, 129, 2))
        evens = list(range(2, 129, 2))
        
        # For each pair (a, b), precompute if a+b is prime
        # a+b is in [3, 256]
        
        # For a fixed A = (a1, a2, a3), a viable b must satisfy:
        # a1+b, a2+b, a3+b are all prime
        # Then for 3 such b's, we need all 9 sums to be prime (and distinct)
        
        # Precompute viable_b for each A
        # To speed up, iterate A and for each, find viable b's
        
        # Sort odds to try smaller ones first (smaller sums more likely prime)
        for a1 in odds[:40]:  # limit to smaller odds for speed
            if time.time() > deadline:
                return None
            for a2 in odds[:40]:
                if a2 <= a1:
                    continue
                if time.time() > deadline:
                    return None
                for a3 in odds[:40]:
                    if a3 <= a2:
                        continue
                    A = (a1, a2, a3)
                    
                    # Find all even b such that a1+b, a2+b, a3+b are all prime
                    viable_b = []
                    for b in evens[:60]:  # limit b for speed
                        s1 = a1 + b
                        s2 = a2 + b
                        s3 = a3 + b
                        if s1 > 256 or s2 > 256 or s3 > 256:
                            continue
                        if s1 in primes_set and s2 in primes_set and s3 in primes_set:
                            viable_b.append(b)
                    
                    if len(viable_b) < 3:
                        continue
                    
                    # Try all triples from viable_b
                    for i in range(len(viable_b)):
                        if time.time() > deadline:
                            return None
                        for j in range(i + 1, len(viable_b)):
                            for k in range(j + 1, len(viable_b)):
                                B = (viable_b[i], viable_b[j], viable_b[k])
                                # Check all 9 sums are prime and distinct
                                sums = [a + b for a in A for b in B]
                                if len(set(sums)) != 9:
                                    continue
                                if all(s in primes_set for s in sums):
                                    return (list(A), list(B))
        
        # Also try A all-even, B all-odd (symmetric)
        for a1 in evens[:40]:
            if time.time() > deadline:
                return None
            for a2 in evens[:40]:
                if a2 <= a1:
                    continue
                if time.time() > deadline:
                    return None
                for a3 in evens[:40]:
                    if a3 <= a2:
                        continue
                    A = (a1, a2, a3)
                    
                    viable_b = []
                    for b in odds[:60]:
                        s1 = a1 + b
                        s2 = a2 + b
                        s3 = a3 + b
                        if s1 > 256 or s2 > 256 or s3 > 256:
                            continue
                        if s1 in primes_set and s2 in primes_set and s3 in primes_set:
                            viable_b.append(b)
                    
                    if len(viable_b) < 3:
                        continue
                    
                    for i in range(len(viable_b)):
                        if time.time() > deadline:
                            return None
                        for j in range(i + 1, len(viable_b)):
                            for k in range(j + 1, len(viable_b)):
                                B = (viable_b[i], viable_b[j], viable_b[k])
                                sums = [a + b for a in A for b in B]
                                if len(set(sums)) != 9:
                                    continue
                                if all(s in primes_set for s in sums):
                                    return (list(A), list(B))
        
        return None

    result = find_valid_pair()
    if result:
        A, B = result
        score = check(A, B)
        if score == 9:
            return [A, B]

    # Fallback: use the original search approach
    def fast_find_B(A, max_b, deadline):
        a1, a2, a3 = A
        viable_b = []
        for b in range(1, max_b + 1):
            s = [a1 + b, a2 + b, a3 + b]
            if len(set(s)) != 3:
                continue
            if not is_pairwise_coprime(s):
                continue
            viable_b.append(b)
        
        if len(viable_b) < 3:
            return None
        
        n = len(viable_b)
        viable_b.sort(key=lambda b: min(a1 + b, a2 + b, a3 + b))
        
        for i in range(n):
            if time.time() > deadline:
                return None
            b1 = viable_b[i]
            sums1 = [a1 + b1, a2 + b1, a3 + b1]
            mask1 = pf_mask[sums1[0]] | pf_mask[sums1[1]] | pf_mask[sums1[2]]
            
            for j in range(i + 1, n):
                b2 = viable_b[j]
                sums2 = [a1 + b2, a2 + b2, a3 + b2]
                mask2 = pf_mask[sums2[0]] | pf_mask[sums2[1]] | pf_mask[sums2[2]]
                
                if mask1 & mask2:
                    continue
                
                for k in range(j + 1, n):
                    b3 = viable_b[k]
                    sums3 = [a1 + b3, a2 + b3, a3 + b3]
                    mask3 = pf_mask[sums3[0]] | pf_mask[sums3[1]] | pf_mask[sums3[2]]
                    
                    if mask1 & mask3 or mask2 & mask3:
                        continue
                    
                    return [b1, b2, b3]
        
        return None

    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    best_score = check(best_A, best_B)

    if best_score == 9:
        return [best_A, best_B]

    a_candidates = list(range(1, 41))
    a_combos = list(combinations(a_candidates, 3))
    a_combos.sort(key=lambda x: (min(x), sum(x)))

    for limit_b in [30, 50, 80, 128]:
        if time.time() > deadline:
            break
        for A in a_combos:
            if time.time() > deadline:
                break
            A = list(A)
            B = fast_find_B(A, limit_b, deadline)
            if B:
                score = check(A, B)
                if score > best_score:
                    best_score = score
                    best_A = A
                    best_B = B
                    if best_score == 9:
                        return [best_A, best_B]

    if best_score < 9 and time.time() < deadline - 5:
        for _ in range(5000):
            if time.time() > deadline:
                break
            A = random.sample(range(1, 61), 3)
            A.sort()
            B = fast_find_B(A, 80, deadline)
            if B:
                score = check(A, B)
                if score > best_score:
                    best_score = score
                    best_A = A
                    best_B = B
                    if best_score == 9:
                        return [best_A, best_B]

    return [best_A, best_B]
