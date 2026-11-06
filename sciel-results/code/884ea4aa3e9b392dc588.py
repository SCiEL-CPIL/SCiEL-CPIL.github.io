import random
import math

def construct():
    n = 4096
    target_size = 40  # Theoretical max is 40
    
    def can_add(x, current_set, existing_sums):
        # Check (x, x, x)
        s = 3 * x
        if s in existing_sums:
            return False, []
        new_sums = [s]
        
        # Check (x, x, y)
        for y in current_set:
            s = 2 * x + y
            if s in existing_sums:
                return False, []
            new_sums.append(s)
            
        # Check (x, y, z)
        k = len(current_set)
        for i in range(k):
            y = current_set[i]
            for j in range(i, k):
                z = current_set[j]
                s = x + y + z
                if s in existing_sums:
                    return False, []
                new_sums.append(s)
                
        return True, new_sums

    def run_greedy(order):
        current_set = []
        existing_sums = set()
        for x in order:
            if x in existing_set: # Optimization: skip if already in set? No, order is unique.
                continue
            ok, new_sums = can_add(x, current_set, existing_sums)
            if ok:
                current_set.append(x)
                existing_sums.update(new_sums)
                if len(current_set) >= target_size:
                    break
        return current_set

    best_set = []
    
    # Strategy 1: Ascending
    res = run_greedy(list(range(1, n + 1)))
    if len(res) > len(best_set):
        best_set = res

    # Strategy 2: Descending
    res = run_greedy(list(range(n, 0, -1)))
    if len(res) > len(best_set):
        best_set = res

    # Strategy 3: Primes
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(math.isqrt(n)) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i, prime in enumerate(is_prime) if prime]
    
    primes = sieve(n)
    res = run_greedy(primes)
    if len(res) > len(best_set):
        best_set = res

    # Strategy 4: Randomized
    for _ in range(200):
        candidates = list(range(1, n + 1))
        random.shuffle(candidates)
        res = run_greedy(candidates)
        if len(res) > len(best_set):
            best_set = res
            if len(best_set) >= target_size:
                break

    return best_set
