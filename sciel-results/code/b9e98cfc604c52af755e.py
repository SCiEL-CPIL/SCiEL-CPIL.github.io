import random
import math
import time

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
    
    # Precompute prime factors for all sums
    pf = [None] * (MAX_SUM + 1)
    pf[0] = frozenset()
    pf[1] = frozenset()
    for i in range(2, MAX_SUM + 1):
        n = i
        factors = set()
        while n > 1:
            p = spf[n]
            factors.add(p)
            while n % p == 0:
                n //= p
        pf[i] = frozenset(factors)

    def objective(nums):
        prime_set = set()
        n = len(nums)
        for i in range(n):
            for j in range(i+1, n):
                prime_set |= pf[nums[i] + nums[j]]
        return len(prime_set)

    best = None
    best_score = float('inf')
    start_time = time.time()
    TIME_LIMIT = 55.0

    # Candidate Generation
    candidates = []
    
    # 1. Random unstructured
    for _ in range(20):
        candidates.append(random.sample(range(1, MAX_VAL+1), 12))
        
    # 2. Scaled candidates
    multipliers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 24, 30, 35, 40, 42, 48, 60, 70, 84, 90, 105, 120, 168, 210, 240, 336, 420, 504, 630, 840, 1260, 1680, 2520, 5040]
    for C in multipliers:
        max_b = MAX_VAL // C
        if max_b >= 12:
            for _ in range(10):
                b = random.sample(range(1, max_b+1), 12)
                candidates.append([C*x for x in b])
                
    # 3. Arithmetic Progressions
    for step in [1, 2, 3, 5, 10, 20, 50]:
        for start in range(1, 50):
            a = [start + i*step for i in range(12)]
            if a[-1] <= MAX_VAL:
                candidates.append(a)

    # Evaluate all candidates
    for cand in candidates:
        if len(set(cand)) != 12 or any(x < 1 or x > MAX_VAL for x in cand):
            continue
        score = objective(cand)
        if score < best_score:
            best_score = score
            best = cand[:]
            if best_score <= 4:
                return best

    # Local Search
    def local_search(initial, time_limit):
        current = initial[:]
        current_score = objective(current)
        best_local = current[:]
        best_local_score = current_score
        start = time.time()
        
        while time.time() - start < time_limit:
            improved = False
            for _ in range(100):
                idx = random.randint(0, 11)
                new_val = random.randint(1, MAX_VAL)
                if new_val == current[idx]:
                    continue
                if new_val in current and new_val != current[idx]:
                    continue
                
                candidate = current[:]
                candidate[idx] = new_val
                new_score = objective(candidate)
                
                if new_score < current_score:
                    current = candidate
                    current_score = new_score
                    if current_score < best_local_score:
                        best_local_score = current_score
                        best_local = current[:]
                    improved = True
            if not improved:
                # Perturb
                for _ in range(2):
                    idx = random.randint(0, 11)
                    current[idx] = random.randint(1, MAX_VAL)
                    while len(set(current)) < 12:
                        current[random.randint(0, 11)] = random.randint(1, MAX_VAL)
                current_score = objective(current)
                if current_score > best_local_score + 2:
                    current = best_local[:]
                    current_score = best_local_score
                    
        return best_local, best_local_score

    if best:
        res, res_score = local_search(best, 30.0)
        if res_score < best_score:
            best_score = res_score
            best = res

    # More searches
    for i in range(5):
        if time.time() - start_time > TIME_LIMIT - 5:
            break
        C = random.choice([1, 2, 3, 6, 10, 30])
        max_b = MAX_VAL // C
        if max_b >= 12:
            b = random.sample(range(1, max_b+1), 12)
            cand = [C*x for x in b]
            res, res_score = local_search(cand, 5.0)
            if res_score < best_score:
                best_score = res_score
                best = res

    return best
