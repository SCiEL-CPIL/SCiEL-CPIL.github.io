import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    best_set = []
    best_size = 0
    
    def check_divisibility(sums):
        sorted_sums = sorted(sums)
        n = len(sorted_sums)
        for i in range(n):
            a = sorted_sums[i]
            for j in range(i + 1, n):
                b = sorted_sums[j]
                if b % a == 0:
                    return False
        return True

    def try_add(sums, x):
        new_sums = set(sums)
        
        if x in new_sums:
            return None
        new_sums.add(x)
        
        for s in sums:
            ns = s + x
            if ns in new_sums:
                return None
            new_sums.add(ns)
            
        if not check_divisibility(new_sums):
            return None
            
        return new_sums

    def backtrack(current_elements, current_sums, candidates):
        nonlocal best_set, best_size
        
        if len(current_elements) > best_size:
            best_size = len(current_elements)
            best_set = list(current_elements)
            
        if len(current_elements) + len(candidates) <= best_size:
            return
            
        if not candidates:
            return
            
        for i, x in enumerate(candidates):
            if time.time() > deadline:
                return
            
            new_sums = try_add(current_sums, x)
            if new_sums is not None:
                backtrack(current_elements + [x], new_sums, candidates[i+1:])

    candidates_all = list(range(1, 257))
    
    def is_prime(n):
        if n < 2: return False
        for i in range(2, int(n**0.5)+1):
            if n % i == 0: return False
        return True

    orderings = []
    orderings.append(sorted(candidates_all, reverse=True))
    orderings.append(sorted(candidates_all))
    
    rng = random.Random(42)
    for _ in range(30):
        shuffled = candidates_all[:]
        rng.shuffle(shuffled)
        orderings.append(shuffled)
        
    primes = [x for x in candidates_all if is_prime(x)]
    non_primes = [x for x in candidates_all if not is_prime(x)]
    orderings.append(primes + non_primes)
    
    # Also try an ordering based on "density" of divisors?
    # Numbers with fewer divisors are "safer" as they don't divide as many things.
    # But we care about subset sums, not the elements themselves directly.
    # However, the smallest element is a subset sum.
    
    for order in orderings:
        if time.time() > deadline:
            break
        backtrack([], set(), order)
        
    return list(best_set)
