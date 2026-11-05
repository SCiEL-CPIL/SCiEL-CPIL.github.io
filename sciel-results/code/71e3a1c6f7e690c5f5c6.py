import math
import random
import time

def construct():
    deadline = time.time() + 85
    
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a
    
    def are_disjoint(r1, m1, r2, m2):
        g = gcd(m1, m2)
        return (r1 - r2) % g != 0
    
    def can_add(r_new, m_new, existing):
        for (r, m) in existing:
            if not are_disjoint(r_new, m_new, r, m):
                return False
        return True
    
    def try_greedy_with_backtrack(moduli_order, max_time=5.0):
        start = time.time()
        result = []
        for m in moduli_order:
            if time.time() - start > max_time:
                break
            found = False
            # Try residues in a smart order
            for r in range(m):
                if can_add(r, m, result):
                    result.append((r, m))
                    found = True
                    break
            if not found:
                # Skip this modulus
                pass
        return result
    
    all_moduli = list(range(2, 129))
    best_result = []
    
    # Strategy 1: All even moduli
    evens = [m for m in all_moduli if m % 2 == 0]
    for attempt in range(100):
        if time.time() > deadline:
            break
        order = evens[:]
        random.shuffle(order)
        result = try_greedy_with_backtrack(order, max_time=2.0)
        if len(result) > len(best_result):
            best_result = result
    
    # Strategy 2: Multiples of 3
    threes = [m for m in all_moduli if m % 3 == 0]
    for attempt in range(50):
        if time.time() > deadline:
            break
        order = threes[:]
        random.shuffle(order)
        result = try_greedy_with_backtrack(order, max_time=2.0)
        if len(result) > len(best_result):
            best_result = result
    
    # Strategy 3: Random subsets biased toward even
    for attempt in range(300):
        if time.time() > deadline:
            break
        subset = []
        for m in all_moduli:
            if m % 2 == 0:
                if random.random() < 0.85:
                    subset.append(m)
            elif m % 3 == 0:
                if random.random() < 0.5:
                    subset.append(m)
            else:
                if random.random() < 0.2:
                    subset.append(m)
        random.shuffle(subset)
        result = try_greedy_with_backtrack(subset, max_time=2.0)
        if len(result) > len(best_result):
            best_result = result
    
    # Strategy 4: Try to add one at a time to best found
    # Start from best_result, try to add more moduli
    used_moduli = set(m for (r, m) in best_result)
    remaining = [m for m in all_moduli if m not in used_moduli]
    for m in remaining:
        if time.time() > deadline:
            break
        for r in range(m):
            if can_add(r, m, best_result):
                best_result.append((r, m))
                used_moduli.add(m)
                break
    
    if not best_result:
        return [[1, 2]]
    
    return [[r, m] for (r, m) in best_result]
