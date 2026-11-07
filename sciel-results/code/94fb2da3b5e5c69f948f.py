import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    def is_valid_addition(current_sums, x):
        """
        Check if x can be added to the set with subset sums current_sums.
        """
        # Fast check: s | x implies s | (s+x)
        for s in current_sums:
            if x % s == 0:
                return False
        
        new_sums = set()
        for s in current_sums:
            ns = s + x
            new_sums.add(ns)
        new_sums.add(x)
        
        # Check duplicates
        for ns in new_sums:
            if ns in current_sums:
                return False
        
        # Full divisibility check
        all_sums = list(current_sums | new_sums)
        all_sums.sort()
        
        for i in range(len(all_sums)):
            a = all_sums[i]
            for j in range(i + 1, len(all_sums)):
                b = all_sums[j]
                if b % a == 0:
                    return False
        return True

    def greedy_random(pool, num_iters=50):
        best = []
        for _ in range(num_iters):
            random.shuffle(pool)
            current_set = []
            current_sums = set()
            for x in pool:
                if time.time() > deadline:
                    break
                if is_valid_addition(current_sums, x):
                    current_set.append(x)
                    for s in current_sums:
                        current_sums.add(s + x)
                    current_sums.add(x)
            if len(current_set) > len(best):
                best = list(current_set)
        return best

    # Candidate Pools
    # 1. Primes (good for divisibility)
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251]
    
    # 2. Large numbers
    large_nums = list(range(128, 257))
    
    # 3. All numbers
    all_nums = list(range(1, 257))
    
    # 4. Numbers with few divisors (primes and semi-primes)
    # Just use primes for now.
    
    best_set = []
    
    # Try different pools
    pools = [primes, large_nums, all_nums]
    
    for pool in pools:
        result = greedy_random(pool, 100)
        if len(result) > len(best_set):
            best_set = result
            
    # Try a mixed pool: Primes + Large Composites
    mixed = primes + [x for x in large_nums if x not in primes]
    result = greedy_random(mixed, 100)
    if len(result) > len(best_set):
        best_set = result

    return best_set
