"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from itertools import permutations

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _is_disjoint(r1, m1, r2, m2):
    """Check if residue classes (r1,m1) and (r2,m2) are disjoint."""
    g = _gcd(m1, m2)
    return (r1 - r2) % g != 0

def _can_assign(residues, moduli, new_r, new_m):
    """Check if new_r, new_m is disjoint from all existing classes."""
    for i in range(len(moduli)):
        if not _is_disjoint(residues[i], moduli[i], new_r, new_m):
            return False
    return True

def _greedy(moduli_order, all_moduli):
    """Greedy: assign residues in the given order, skip moduli that can't be assigned."""
    residues = []
    moduli = []
    for m in moduli_order:
        if m not in all_moduli:
            continue
        found = False
        for r in range(m):
            if _can_assign(residues, moduli, r, m):
                residues.append(r)
                moduli.append(m)
                found = True
                break
        # If not found, skip this modulus
    return list(zip(residues, moduli))

def _swap_improve(result):
    """Try 1-for-1 swaps to improve the result."""
    if not result:
        return result
    best = list(result)
    improved = True
    while improved:
        improved = False
        # Try replacing each selected modulus with an unselected one
        selected_mods = set(m for _, m in best)
        all_mods = set(range(2, 129))
        unselected = all_mods - selected_mods
        for i in range(len(best)):
            for new_m in unselected:
                # Try to find a residue for new_m that's disjoint from all others
                others = [(r, m) for j, (r, m) in enumerate(best) if j != i]
                found = False
                for r in range(new_m):
                    ok = True
                    for or_, om in others:
                        if not _is_disjoint(r, new_m, or_, om):
                            ok = False
                            break
                    if ok:
                        new_result = best[:i] + [(r, new_m)] + best[i+1:]
                        # This is same size, so only useful if we can then add more
                        # For now, just track it
                        found = True
                        break
                # Swaps don't increase size, so skip for now
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin
    
    all_moduli = list(range(2, 129))
    
    # Try multiple orderings
    best_result = []
    
    # Ordering 1: ascending
    order1 = list(all_moduli)
    r1 = _greedy(order1, set(all_moduli))
    if len(r1) > len(best_result):
        best_result = r1
    
    if time.time() > deadline:
        return [[r, m] for r, m in best_result]
    
    # Ordering 2: descending
    order2 = list(reversed(all_moduli))
    r2 = _greedy(order2, set(all_moduli))
    if len(r2) > len(best_result):
        best_result = r2
    
    if time.time() > deadline:
        return [[r, m] for r, m in best_result]
    
    # Ordering 3: most constrained first (by sum of gcds with others)
    def constraint_score(m):
        total = 0
        for other in all_moduli:
            if other != m:
                total += _gcd(m, other)
        return total
    
    order3 = sorted(all_moduli, key=constraint_score, reverse=True)
    r3 = _greedy(order3, set(all_moduli))
    if len(r3) > len(best_result):
        best_result = r3
    
    if time.time() > deadline:
        return [[r, m] for r, m in best_result]
    
    # Ordering 4: least constrained first
    order4 = sorted(all_moduli, key=constraint_score)
    r4 = _greedy(order4, set(all_moduli))
    if len(r4) > len(best_result):
        best_result = r4
    
    if time.time() > deadline:
        return [[r, m] for r, m in best_result]
    
    # Try random orderings
    import random
    random.seed(42)
    for trial in range(200):
        if time.time() > deadline:
            break
        order = list(all_moduli)
        random.shuffle(order)
        r = _greedy(order, set(all_moduli))
        if len(r) > len(best_result):
            best_result = r
    
    # Try starting with a subset of "easy" moduli (pairwise coprime) then adding more
    # Start with primes, then add composite moduli one by one
    primes = [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,101,103,107,109,113,127]
    
    # Build up from primes
    current = []
    for p in primes:
        for r in range(p):
            if _can_assign([x[0] for x in current], [x[1] for x in current], r, p):
                current.append((r, p))
                break
    
    if len(current) > len(best_result):
        best_result = list(current)
    
    if time.time() > deadline:
        return [[r, m] for r, m in best_result]
    
    # Now try to add non-prime moduli to the current set
    composites = [m for m in all_moduli if m not in set(primes)]
    for m in composites:
        if time.time() > deadline:
            break
        for r in range(m):
            if _can_assign([x[0] for x in current], [x[1] for x in current], r, m):
                current.append((r, m))
                break
    
    if len(current) > len(best_result):
        best_result = list(current)
    
    if time.time() > deadline:
        return [[r, m] for r, m in best_result]
    
    # Final attempt: try all moduli in several heuristic orders
    # Order by number of prime factors (more factors = more constraints)
    def num_prime_factors(n):
        count = 0
        d = 2
        temp = n
        while d * d <= temp:
            while temp % d == 0:
                count += 1
                temp //= d
            d += 1
        if temp > 1:
            count += 1
        return count
    
    order5 = sorted(all_moduli, key=num_prime_factors, reverse=True)
    r5 = _greedy(order5, set(all_moduli))
    if len(r5) > len(best_result):
        best_result = r5
    
    if time.time() > deadline:
        return [[r, m] for r, m in best_result]
    
    # Sort result for canonical output
    best_result.sort(key=lambda x: x[1])
    return [[r, m] for r, m in best_result]

# EVOLVE-BLOCK-END
