import time
import random
from math import isqrt

# EVOLVE-BLOCK-START
P = {'n': 40, 'maximum_size': 18}

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    # Generate candidates 2 to 40
    candidates = list(range(2, P['n'] + 1))
    
    # Precompute prime factors
    def largest_prime_factor(n):
        if n % 2 == 0:
            lp = 2
        else:
            lp = n
        i = 3
        while i * i <= n:
            while n % i == 0:
                lp = i
                n //= i
            i += 2
        if n > 1:
            lp = n
        return lp
    
    def num_prime_factors(n):
        count = 0
        d = 2
        while d * d <= n:
            while n % d == 0:
                count += 1
                n //= d
            d += 1
        if n > 1:
            count += 1
        return count
    
    # Primes and composites
    primes = [a for a in range(2, P['n']+1) if all(a%d for d in range(2,isqrt(a)+1))]
    composites = [a for a in candidates if a not in primes]
    
    best_set = []
    best_size = 0
    
    # Verify primes set
    prods = {1}
    valid = True
    for p in primes:
        new_prods = list(prods)
        for prod in new_prods:
            if p * prod in prods:
                valid = False
                break
            prods.add(p * prod)
        if not valid:
            break
    if valid:
        best_set = primes[:]
        best_size = len(primes)
    
    def check_set(set_vals):
        prods = {1}
        for v in set_vals:
            new_prods = list(prods)
            for prod in new_prods:
                if v * prod in prods:
                    return False
                prods.add(v * prod)
        return True

    def try_greedy(order):
        current = []
        prods = {1}
        for v in order:
            new_prods = []
            valid = True
            for p in prods:
                np = v * p
                if np in prods:
                    valid = False
                    break
                new_prods.append(np)
            if valid:
                current.append(v)
                prods.update(new_prods)
        return current

    # Try deterministic orders
    orders = [
        primes + sorted(composites),
        primes + sorted(composites, reverse=True),
        sorted(candidates, key=lambda x: -largest_prime_factor(x)),
        sorted(candidates, key=lambda x: (num_prime_factors(x), x)),
        sorted(candidates, key=lambda x: (-num_prime_factors(x), x)),
    ]
    
    for order in orders:
        sol = try_greedy(order)
        if len(sol) > best_size:
            best_size = len(sol)
            best_set = sol[:]
    
    # Backtracking search with pruning
    def backtrack(current_set, prods, remaining, depth):
        nonlocal best_set, best_size
        if time.time() > deadline:
            return
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
        if not remaining:
            return
        # Try adding each remaining element
        for i in range(len(remaining)):
            v = remaining[i]
            new_prods = list(prods)
            valid = True
            for p in new_prods:
                np = v * p
                if np in prods:
                    valid = False
                    break
            if valid:
                new_prods_set = set(prods)
                for p in new_prods:
                    new_prods_set.add(v * p)
                backtrack(current_set + [v], new_prods_set, remaining[:i] + remaining[i+1:], depth+1)
                if time.time() > deadline:
                    return
    
    # Try backtracking from different starting points
    # Start with primes only
    if time.time() < deadline:
        remaining = composites[:]
        backtrack(primes[:], {1}, remaining, 0)
    
    # Try backtracking from empty set with limited depth
    if time.time() < deadline:
        # Use a heuristic order for backtracking
        ordered_candidates = sorted(candidates, key=lambda x: -largest_prime_factor(x))
        backtrack([], {1}, ordered_candidates[:], 0)
    
    # Randomized greedy with more restarts
    random.seed(42)
    
    for i in range(5000):
        if time.time() > deadline:
            break
        shuffled = candidates[:]
        random.shuffle(shuffled)
        sol = try_greedy(shuffled)
        if len(sol) > best_size:
            best_size = len(sol)
            best_set = sol[:]
    
    # Local search: try to swap elements to improve
    if time.time() < deadline:
        for _ in range(1000):
            if time.time() > deadline:
                break
            current = best_set[:]
            # Try removing one element and adding another
            for i in range(len(current)):
                if time.time() > deadline:
                    break
                removed = current[i]
                without_removed = current[:i] + current[i+1:]
                # Check if without_removed is valid
                if not check_set(without_removed):
                    continue
                # Try adding each candidate not in current
                for v in candidates:
                    if v in current:
                        continue
                    test_set = without_removed + [v]
                    if len(set(test_set)) != len(test_set):
                        continue
                    if check_set(test_set):
                        if len(test_set) > len(current):
                            current = test_set
                            if len(current) > best_size:
                                best_size = len(current)
                                best_set = current[:]
                            break
    
    # Return the best set found
    # Ensure types are correct
    return [int(x) for x in best_set]
# EVOLVE-BLOCK-END
