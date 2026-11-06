"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
from itertools import combinations

MAX_MOD = 128
DEADLINE = 88.0  # safety margin before 90s kill

def _disjoint(r1, m1, r2, m2):
    g = math.gcd(m1, m2)
    return (r2 - r1) % g != 0

def _is_valid_solution(pairs):
    n = len(pairs)
    for i in range(n):
        r1, m1 = pairs[i]
        for j in range(i+1, n):
            r2, m2 = pairs[j]
            if not _disjoint(r1, m1, r2, m2):
                return False
    return True

def _find_residue(m, existing_pairs):
    """Find a residue r for modulus m that is disjoint from all existing pairs."""
    for r in range(m):
        ok = True
        for (r2, m2) in existing_pairs:
            if not _disjoint(r, m, r2, m2):
                ok = False
                break
        if ok:
            return r
    return None

def _greedy_build(moduli_order):
    pairs = []
    for m in moduli_order:
        r = _find_residue(m, pairs)
        if r is not None:
            pairs.append([r, m])
    return pairs

def _local_search(pairs, rng, time_limit):
    """Try to improve the solution by removing and re-adding moduli."""
    current_pairs = [p[:] for p in pairs]
    current_size = len(current_pairs)
    best_pairs = [p[:] for p in pairs]
    best_size = current_size
    all_moduli = list(range(2, MAX_MOD + 1))
    
    def _try_add(m, ps):
        r = _find_residue(m, ps)
        if r is not None:
            return [p[:] for p in ps] + [[r, m]]
        return None
    
    while time.time() < time_limit:
        current_mods = [m for _, m in current_pairs]
        available = [m for m in all_moduli if m not in current_mods]
        
        if not available:
            break
        
        # Try to add a random available modulus
        m = rng.choice(available)
        new_pairs = _try_add(m, current_pairs)
        if new_pairs is not None:
            current_pairs = new_pairs
            current_size = len(current_pairs)
            if current_size > best_size:
                best_size = current_size
                best_pairs = [p[:] for p in current_pairs]
                if best_size >= 30:  # good enough, stop
                    break
        
        # Occasionally try a swap: remove one, add another
        if rng.random() < 0.3 and len(current_pairs) > 0:
            idx = rng.randrange(len(current_pairs))
            removed = current_pairs[idx]
            reduced = [p[:] for p in current_pairs[:idx]] + [p[:] for p in current_pairs[idx+1:]]
            # Try to add a different modulus
            available_after = [m for m in all_moduli if m not in [p[1] for p in reduced]]
            if available_after:
                m2 = rng.choice(available_after)
                new_pairs = _try_add(m2, reduced)
                if new_pairs is not None and len(new_pairs) >= len(reduced):
                    current_pairs = new_pairs
                    current_size = len(current_pairs)
                    if current_size > best_size:
                        best_size = current_size
                        best_pairs = [p[:] for p in current_pairs]
    
    return best_pairs

def construct():
    start_time = time.time()
    rng = random.Random(42)
    all_moduli = list(range(2, MAX_MOD + 1))
    
    best_pairs = []
    
    # Strategy 1: Multiple greedy orderings
    # Order by number of divisors (fewer divisors = more flexible)
    def _divisor_count(m):
        return sum(1 for d in range(1, m+1) if m % d == 0)
    
    # Sort by divisor count ascending (primes and semiprimes first)
    order_by_divs = sorted(all_moduli, key=_divisor_count)
    p = _greedy_build(order_by_divs)
    if len(p) > len(best_pairs):
        best_pairs = p
    
    # Sort by value descending
    order_desc = sorted(all_moduli, reverse=True)
    p = _greedy_build(order_desc)
    if len(p) > len(best_pairs):
        best_pairs = p
    
    # Sort by value ascending
    order_asc = sorted(all_moduli)
    p = _greedy_build(order_asc)
    if len(p) > len(best_pairs):
        best_pairs = p
    
    # Evens first, then odds
    evens = [m for m in all_moduli if m % 2 == 0]
    odds = [m for m in all_moduli if m % 2 != 0]
    order_eo = sorted(evens, reverse=True) + sorted(odds, reverse=True)
    p = _greedy_build(order_eo)
    if len(p) > len(best_pairs):
        best_pairs = p
    
    # Randomized greedy
    for _ in range(50):
        if time.time() - start_time > DEADLINE - 10:
            break
        order = all_moduli[:]
        rng.shuffle(order)
        p = _greedy_build(order)
        if len(p) > len(best_pairs):
            best_pairs = p
    
    # Local search from best
    if len(best_pairs) > 0:
        improved = _local_search(best_pairs, rng, DEADLINE - 2)
        if len(improved) > len(best_pairs):
            best_pairs = improved
    
    return best_pairs

# EVOLVE-BLOCK-END
