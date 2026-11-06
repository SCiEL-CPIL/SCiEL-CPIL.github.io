import random
import math

def construct():
    MAX_VAL = 10000
    MAX_SUM = 2 * MAX_VAL  # 20000
    
    # Precompute smallest prime factor for all numbers up to MAX_SUM
    spf = list(range(MAX_SUM + 1))
    for i in range(2, int(MAX_SUM**0.5) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i*i, MAX_SUM + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    def get_prime_factors(n):
        """Return set of distinct prime factors using SPF."""
        if n <= 1:
            return set()
        factors = set()
        while n > 1:
            p = spf[n]
            factors.add(p)
            while n % p == 0:
                n //= p
        return factors
    
    def objective(nums):
        prime_set = set()
        n = len(nums)
        for i in range(n):
            for j in range(i+1, n):
                s = nums[i] + nums[j]
                prime_set |= get_prime_factors(s)
        return len(prime_set)

    best = None
    best_score = float('inf')
    
    # Structured Initialization 1: a_i = 210 * k_i
    # Try to make k_i + k_j have few factors
    # k_i in [1, 47]
    # Try k_i all even? Then sum is even.
    # Try k_i all divisible by 3? Then sum div by 3.
    
    candidates = []
    
    # Pattern A: a_i = 210 * k_i, k_i multiples of 6
    # k = [6, 12, 18, 24, 30, 36, 42] -> only 7 values. Not enough.
    # k = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    # Sums range 3 to 23.
    
    # Pattern B: a_i = 210 * k_i + 105
    # k in [0, 47]
    # u_i = k_i + 1 in [1, 48]
    # S_ij = u_i + u_j - 1
    # Try u_i all 2 mod 3.
    
    def generate_structured(pattern):
        if pattern == 0:
            # a_i = 210 * k_i
            # Pick 12 k_i from 1..47
            # Try to cluster k_i to keep sums small
            # e.g., 1..12
            ks = list(range(1, 13))
            return [210 * k for k in ks]
        elif pattern == 1:
            # a_i = 210 * k_i + 105
            # u_i = k_i + 1
            # Pick u_i from [1..48] such that u_i = 2 mod 3
            us = [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35]
            return [210 * (u - 1) + 105 for u in us]
        elif pattern == 2:
            # a_i = 210 * k_i
            # Pick k_i all even
            ks = list(range(2, 25, 2)) # 2, 4, ..., 24 (12 values)
            return [210 * k for k in ks]
        elif pattern == 3:
            # a_i = 210 * k_i
            # Pick k_i all odd
            ks = list(range(1, 24, 2)) # 1, 3, ..., 23 (12 values)
            return [210 * k for k in ks]
        elif pattern == 4:
            # a_i = 210 * k_i + 105
            # Pick u_i all even
            us = list(range(2, 25, 2)) # 2, 4, ..., 24 (12 values)
            return [210 * (u - 1) + 105 for u in us]
        elif pattern == 5:
            # a_i = 210 * k_i + 105
            # Pick u_i all odd
            us = list(range(1, 24, 2)) # 1, 3, ..., 23 (12 values)
            return [210 * (u - 1) + 105 for u in us]
        return None

    for p in range(6):
        cand = generate_structured(p)
        if cand and all(1 <= x <= MAX_VAL for x in cand) and len(set(cand)) == 12:
            score = objective(cand)
            if score < best_score:
                best_score = score
                best = cand[:]

    # Add some random restarts to ensure we don't miss a better random cluster
    for seed in range(20):
        random.seed(seed)
        candidate = random.sample(range(1, MAX_VAL+1), 12)
        score = objective(candidate)
        if score < best_score:
            best_score = score
            best = candidate[:]

    # Local Search
    # We will use a simulated annealing style or simple hill climbing
    # Given the structured start, simple hill climbing should work well.
    
    for restart in range(10):
        current = best[:]
        current_score = objective(current)
        
        # If we start a new restart, maybe perturb the best found so far
        if restart > 0:
            # Perturb 2 random elements
            for _ in range(2):
                idx = random.randint(0, 11)
                current[idx] = random.randint(1, MAX_VAL)
            # Ensure distinctness
            while len(set(current)) < 12:
                current[random.randint(0, 11)] = random.randint(1, MAX_VAL)
            current_score = objective(current)
        
        for iteration in range(5000):
            idx = random.randint(0, 11)
            # Try a small step or a random step
            if random.random() < 0.5:
                # Small step: add/subtract small number
                step = random.randint(-10, 10)
                new_val = current[idx] + step
            else:
                new_val = random.randint(1, MAX_VAL)
            
            if new_val < 1 or new_val > MAX_VAL:
                continue
            if new_val in current and new_val != current[idx]:
                continue
                
            candidate = current[:]
            candidate[idx] = new_val
            
            # Check distinctness
            if len(set(candidate)) != 12:
                continue
                
            new_score = objective(candidate)
            
            if new_score < current_score:
                current = candidate
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best = current[:]
                    if best_score <= 4:
                        return best

    return best
