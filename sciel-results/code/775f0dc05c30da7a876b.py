"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    MAX_MOD = 128
    moduli = list(range(2, MAX_MOD+1))
    
    gcd_table = {}
    for i in range(2, MAX_MOD+1):
        for j in range(2, MAX_MOD+1):
            gcd_table[(i,j)] = math.gcd(i, j)
    
    def is_disjoint(r1, m1, r2, m2):
        g = gcd_table[(m1, m2)]
        return (r1 - r2) % g != 0
    
    def can_add(r, m, selected):
        for (sr, sm) in selected:
            if not is_disjoint(r, m, sr, sm):
                return False
        return True
    
    def num_divisors(n):
        c = 0
        for i in range(1, int(n**0.5)+1):
            if n % i == 0:
                c += 1
                if i != n//i:
                    c += 1
        return c
    
    best_solution = []
    
    # Deterministic greedy with various orderings
    orders = [
        sorted(moduli),
        sorted(moduli, reverse=True),
        sorted(moduli, key=lambda m: num_divisors(m)),
        sorted(moduli, key=lambda m: num_divisors(m), reverse=True),
    ]
    
    for order in orders:
        if time.time() > deadline:
            break
        selected = []
        for m in order:
            if time.time() > deadline:
                break
            for r in range(m):
                if can_add(r, m, selected):
                    selected.append((r, m))
                    break
        if len(selected) > len(best_solution):
            best_solution = selected
    
    # Randomized restarts
    random.seed(42)
    for _ in range(500):
        if time.time() > deadline:
            break
        order = moduli[:]
        random.shuffle(order)
        selected = []
        for m in order:
            if time.time() > deadline:
                break
            valid = [r for r in range(m) if can_add(r, m, selected)]
            if valid:
                selected.append((random.choice(valid), m))
        if len(selected) > len(best_solution):
            best_solution = selected
    
    result = [[r, m] for (r, m) in best_solution]
    if not result:
        result = [[1, 2]]
    return result
# EVOLVE-BLOCK-END
