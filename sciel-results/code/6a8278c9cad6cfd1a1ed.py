# EVOLVE-BLOCK-START
import time
from math import gcd
from itertools import combinations

def are_pairwise_coprime(nums):
    unique = list(set(nums))
    n = len(unique)
    for i in range(n):
        for j in range(i + 1, n):
            if gcd(unique[i], unique[j]) != 1:
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    def check_pair(A, B):
        sums = [A[i] + B[j] for i in range(3) for j in range(3)]
        distinct_sums = set(sums)
        if len(distinct_sums) < 9:
            return False, 0
        s_list = list(distinct_sums)
        for i in range(len(s_list)):
            for j in range(i + 1, len(s_list)):
                if gcd(s_list[i], s_list[j]) != 1:
                    return False, len(distinct_sums)
        return True, len(distinct_sums)
    
    # Precompute: which sums in [2, 256] are coprime to all other potential sums?
    # Strategy: Find triples A,B where all 9 sums are distinct primes or 1
    # First, let's try a more targeted search using prime sums
    
    # Generate all valid sum values (primes and 1 in range [2,256])
    def is_prime(n):
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(n**0.5) + 1, 2):
            if n % i == 0:
                return False
        return True
    
    prime_sums = [p for p in range(2, 257) if is_prime(p)]
    
    # Try to find A,B such that A+B contains only primes
    # For each candidate A, check if there exists B such that all A[i]+B[j] are prime
    # and pairwise coprime (which they will be if all distinct primes)
    
    # Quick check: try small ranges first with prime-sum constraint
    for max_val in [20, 30, 50, 100, 128]:
        if time.time() > deadline - 10:
            break
        
        # Generate candidates with values in [1, max_val]
        candidates = []
        for x in range(1, max_val + 1):
            for y in range(x + 1, max_val + 1):
                for z in range(y + 1, max_val + 1):
                    candidates.append([x, y, z])
        
        # For each pair, check if all sums are prime (and hence pairwise coprime if distinct)
        for A in candidates:
            if time.time() > deadline - 10:
                break
            for B in candidates:
                if time.time() > deadline - 10:
                    break
                sums = [A[i] + B[j] for i in range(3) for j in range(3)]
                distinct_sums = set(sums)
                if len(distinct_sums) != 9:
                    continue
                # Check if all are prime
                all_prime = all(is_prime(s) for s in distinct_sums)
                if all_prime:
                    return [A, B]
        
        # If no all-prime solution found, relax to pairwise coprime
        for A in candidates:
            if time.time() > deadline - 5:
                break
            for B in candidates:
                if time.time() > deadline - 5:
                    break
                valid, count = check_pair(A, B)
                if valid and count == 9:
                    return [A, B]
    
    # Fallback: broader random search
    import random
    random.seed(42)
    
    best_valid_A = None
    best_valid_B = None
    best_count = 0
    
    while time.time() < start_time + 50:
        A = sorted(random.sample(range(1, 129), 3))
        B = sorted(random.sample(range(1, 129), 3))
        valid, count = check_pair(A, B)
        if valid and count > best_count:
            best_count = count
            best_valid_A = A[:]
            best_valid_B = B[:]
            if best_count == 9:
                return [best_valid_A, best_valid_B]
        
    if best_valid_A is not None:
        return [best_valid_A, best_valid_B]
    
    return [[1, 3, 5], [4, 6, 8]]
# EVOLVE-BLOCK-END
