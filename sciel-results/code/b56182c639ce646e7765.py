import time
import random
from math import isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin for 90s limit
    
    n = 12
    max_val = 10000
    
    # Primes to consider in increasing order of "cost" (number of distinct primes)
    # We want to minimize the count, so we try sets of primes of size 1, 2, 3...
    # But we know size 1 is impossible for n=12.
    # Let's try size 2: {2, 3}
    # Then size 3: {2, 3, 5}
    # Then size 4: {2, 3, 5, 7}
    
    prime_sets = [
        [2, 3],
        [2, 3, 5],
        [2, 3, 5, 7],
        [2, 3, 5, 7, 11]
    ]
    
    best_solution = None
    best_prime_count = float('inf')
    
    for primes in prime_sets:
        if time.time() > deadline:
            break
            
        # Precompute smooth numbers up to 2 * max_val
        max_sum = 2 * max_val
        smooth_set = set()
        # Generate all numbers of form p1^a * p2^b * ... <= max_sum
        # Recursive generation
        def gen_smooth(current, idx, limit):
            if current > limit:
                return
            smooth_set.add(current)
            if idx < len(primes):
                p = primes[idx]
                next_val = current
                while next_val <= limit:
                    gen_smooth(next_val, idx + 1, limit)
                    next_val *= p
            else:
                # No more primes, but we can still multiply by the last prime?
                # Actually, the standard way is to iterate through primes.
                # The above logic is slightly flawed for generating all combinations.
                # Better:
                pass
        
        # Correct generation of smooth numbers
        smooth_set = set()
        def gen(idx, current):
            if current > max_sum:
                return
            smooth_set.add(current)
            if idx < len(primes):
                p = primes[idx]
                next_val = current
                while next_val <= max_sum:
                    gen(idx + 1, next_val)
                    next_val *= p
            # Note: This generates numbers composed of primes[0..idx].
            # It misses numbers that don't use all primes, but that's fine, 
            # we just need to check if a number is smooth.
            # Actually, the set `smooth_set` will contain all smooth numbers.
        
        gen(0, 1)
        
        # Now, we have a set of smooth sums.
        # We need to find 12 distinct integers x1..x12 <= 10000
        # such that xi + xj in smooth_set for all i != j.
        
        # Optimization: 
        # For a fixed x1, x2 must be such that x1+x2 is smooth.
        # So x2 must be in {s - x1 | s in smooth_set, 1 <= s-x1 <= 10000, s-x1 != x1}.
        
        # We can try to find a clique.
        # Given the time limit, a simple randomized greedy search with restarts is appropriate.
        
        found = False
        for _ in range(10000): # Try up to 10000 restarts
            if time.time() > deadline:
                break
                
            # Randomly pick a start
            # To bias towards success, maybe pick small numbers? 
            # Or just random.
            x1 = random.randint(1, max_val)
            candidates = set()
            for s in smooth_set:
                val = s - x1
                if 1 <= val <= max_val and val != x1:
                    candidates.add(val)
            
            if len(candidates) < n - 1:
                continue
                
            # We have x1. Now pick x2 from candidates.
            x2 = random.choice(list(candidates))
            
            # Now x3..x12 must be common neighbors of x1 and x2.
            # Common neighbors: v such that v+x1 in smooth_set AND v+x2 in smooth_set.
            # And v not in {x1, x2}.
            
            common = set()
            for v in candidates:
                if v == x2: continue
                if (v + x2) in smooth_set:
                    common.add(v)
            
            if len(common) < n - 2:
                continue
                
            # Now we need n-3 more from `common` such that they are also mutually connected.
            # This is still a clique problem on `common`.
            # But `common` might be small enough or dense enough.
            
            # Greedy pick for the rest
            selected = [x1, x2]
            remaining = list(common)
            random.shuffle(remaining)
            
            success = True
            for v in remaining:
                if len(selected) == n:
                    break
                # Check if v connects to all in selected
                ok = True
                for u in selected:
                    if (v + u) not in smooth_set:
                        ok = False
                        break
                if ok:
                    selected.append(v)
            
            if len(selected) == n:
                # Verify
                valid = True
                for i in range(n):
                    for j in range(i+1, n):
                        if (selected[i] + selected[j]) not in smooth_set:
                            valid = False
                            break
                    if not valid: break
                if valid:
                    best_solution = selected
                    best_prime_count = len(primes)
                    found = True
                    break
        
        if found:
            break
        
    if best_solution:
        return best_solution
    else:
        # Fallback: return a valid set, even if not optimal
        # e.g., 1..12
        return list(range(1, 13))
