import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 50.0
    
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True

    candidates = list(range(128, 257))
    primes = [x for x in candidates if is_prime(x)]
    non_primes = [x for x in candidates if not is_prime(x)]
    primes.sort(reverse=True)
    non_primes.sort(reverse=True)
    
    # Strategy 1: Greedy with sorted primes
    # Strategy 2: Randomized
    
    best_set = []
    best_len = 0
    
    def try_build(candidates):
        current_set = []
        sums = set()
        for c in candidates:
            if time.time() - start_time > DEADLINE:
                break
            if c in sums:
                continue
            new_sums = [c] + [s + c for s in sums]
            if len(new_sums) != len(set(new_sums)):
                continue
            valid = True
            ns_sorted = sorted(new_sums)
            # Check new-new
            for i in range(len(ns_sorted)):
                a = ns_sorted[i]
                for j in range(i+1, len(ns_sorted)):
                    if ns_sorted[j] % a == 0:
                        valid = False
                        break
                if not valid: break
            if not valid: continue
            
            # Check cross
            for n in new_sums:
                for s in sums:
                    if n % s == 0 or s % n == 0:
                        valid = False
                        break
                if not valid: break
            if not valid: continue
            
            current_set.append(c)
            sums.update(new_sums)
        return current_set

    # Run a few deterministic tries
    for _ in range(10):
        s = try_build(primes)
        if len(s) > best_len:
            best_set = s
            best_len = len(s)
            
        s = try_build(non_primes)
        if len(s) > best_len:
            best_set = s
            best_len = len(s)
            
        s = try_build(primes + non_primes)
        if len(s) > best_len:
            best_set = s
            best_len = len(s)

    # Run randomized tries
    max_restarts = 1000
    for i in range(max_restarts):
        if time.time() - start_time > DEADLINE:
            break
        cand = primes + non_primes
        random.shuffle(cand)
        s = try_build(cand)
        if len(s) > best_len:
            best_set = s
            best_len = len(s)

    return [int(x) for x in best_set]
