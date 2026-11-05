import time
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s
    
    MAX_M = 128
    
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return abs(a)
    
    def are_disjoint(r1, m1, r2, m2):
        """Check if residue classes (r1,m1) and (r2,m2) are disjoint."""
        g = gcd(m1, m2)
        return (r1 - r2) % g != 0
    
    # We want to select a subset of moduli from 2..128 and assign residues
    # such that all pairs are disjoint. Maximize the count.
    
    # Strategy: Try to include as many moduli as possible.
    # Use greedy: try adding moduli in some order, assign residues via backtracking.
    
    # First, let's try a simple greedy: iterate through moduli, try to add each one
    # by finding a valid residue.
    
    def try_greedy(moduli_order):
        """Try to build a solution by processing moduli in given order."""
        selected = []  # list of (r, m)
        for m in moduli_order:
            # Try each possible residue 0..m-1
            found = False
            for r in range(m):
                ok = True
                for (r2, m2) in selected:
                    if not are_disjoint(r, m, r2, m2):
                        ok = False
                        break
                if ok:
                    selected.append((r, m))
                    found = True
                    break
            if not found:
                # Can't add this modulus, skip it
                pass
        return selected
    
    # Try different orderings of moduli
    # Start with all moduli 2..128
    all_moduli = list(range(2, MAX_M + 1))
    
    # Order 1: ascending
    best = try_greedy(all_moduli)
    
    if time.time() > deadline:
        return [[r, m] for r, m in best]
    
    # Order 2: descending
    best2 = try_greedy(list(reversed(all_moduli)))
    if len(best2) > len(best):
        best = best2
    
    if time.time() > deadline:
        return [[r, m] for r, m in best]
    
    # Order 3: by number of divisors (fewer divisors first, easier to satisfy)
    def num_divisors(n):
        count = 0
        i = 1
        while i * i <= n:
            if n % i == 0:
                count += 1
                if i != n // i:
                    count += 1
            i += 1
        return count
    
    order3 = sorted(all_moduli, key=lambda m: num_divisors(m))
    best3 = try_greedy(order3)
    if len(best3) > len(best):
        best = best3
    
    if time.time() > deadline:
        return [[r, m] for r, m in best]
    
    # Try a more sophisticated approach: use backtracking with pruning
    # to find a larger set.
    
    # Let's try to find the maximum by using a recursive search with memoization
    # or by trying to add moduli in different orders.
    
    # Try random orderings
    import random
    random.seed(42)
    for trial in range(50):
        if time.time() > deadline:
            break
        order = all_moduli[:]
        random.shuffle(order)
        result = try_greedy(order)
        if len(result) > len(best):
            best = result
    
    # Try to improve by removing some moduli and re-adding others
    # This is getting complex. Let's just return the best we have.
    
    return [[r, m] for r, m in best]
