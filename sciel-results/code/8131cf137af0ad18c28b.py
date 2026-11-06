"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
from collections import defaultdict

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

def _greedy_build(moduli_order, rng=None):
    pairs = []
    for m in moduli_order:
        best_r = None
        # Shuffle residues if rng provided to allow diversity
        residues = list(range(m))
        if rng:
            rng.shuffle(residues)
        else:
            residues = sorted(residues)
            
        for r in residues:
            ok = True
            for (r2, m2) in pairs:
                if not _disjoint(r, m, r2, m2):
                    ok = False
                    break
            if ok:
                best_r = r
                break
        if best_r is not None:
            pairs.append([best_r, m])
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

def _simulate_annealing(pairs, rng, time_limit):
    """Simulated annealing style local search with residue changes and swaps."""
    current_pairs = [p[:] for p in pairs]
    best_pairs = [p[:] for p in pairs]
    best_size = len(current_pairs)
    all_moduli = list(range(2, MAX_MOD + 1))
    start_time = time.time()
    
    def _try_add_modulus(pairs, m, rng):
        """Try to find a residue for m that is disjoint from all in pairs."""
        residues = list(range(m))
        rng.shuffle(residues)
        for r in residues:
            ok = True
            for (r2, m2) in pairs:
                if not _disjoint(r, m, r2, m2):
                    ok = False
                    break
            if ok:
                return r
        return None

    def _try_remove_modulus(pairs, index):
        return pairs[:index] + pairs[index+1:]

    # Iterative improvement with simulated annealing moves
    while time.time() < time_limit:
        current_mods = [m for _, m in current_pairs]
        available = [m for m in all_moduli if m not in current_mods]
        
        # Randomly pick an action
        action = rng.random()
        
        if action < 0.4 and available:
            # Try to add a random modulus not in current set
            m = rng.choice(available)
            r = _try_add_modulus(current_pairs, m, rng)
            if r is not None:
                new_pairs = current_pairs + [[r, m]]
                if len(new_pairs) > best_size:
                    best_size = len(new_pairs)
                    best_pairs = [p[:] for p in new_pairs]
                current_pairs = new_pairs
                
        elif action < 0.7 and len(current_pairs) > 0:
            # Try to change a residue
            idx = rng.randrange(len(current_pairs))
            r_old, m = current_pairs[idx]
            # Find a new residue for m that is disjoint from others
            others = [p for i, p in enumerate(current_pairs) if i != idx]
            residues = list(range(m))
            residues.remove(r_old)
            rng.shuffle(residues)
            found = False
            for r in residues:
                ok = True
                for (r2, m2) in others:
                    if not _disjoint(r, m, r2, m2):
                        ok = False
                        break
                if ok:
                    current_pairs[idx] = [r, m]
                    found = True
                    break
            # Even if not found, we might have broken something? No, we only change if ok.
            # But changing residue might allow adding another modulus later.
            
        elif action < 0.9 and len(current_pairs) > 0:
            # Try a swap: remove one, add another
            idx = rng.randrange(len(current_pairs))
            reduced = _try_remove_modulus(current_pairs, idx)
            available_after = [m for m in all_moduli if m not in [p[1] for p in reduced]]
            if available_after:
                m2 = rng.choice(available_after)
                r2 = _try_add_modulus(reduced, m2, rng)
                if r2 is not None:
                    new_pairs = reduced + [[r2, m2]]
                    if len(new_pairs) > best_size:
                        best_size = len(new_pairs)
                        best_pairs = [p[:] for p in new_pairs]
                    # Accept with some probability even if same size (to escape local optima)
                    if len(new_pairs) >= len(current_pairs):
                        current_pairs = new_pairs
                        
        else:
            # Try to remove a modulus to free up constraints
            if len(current_pairs) > 0:
                idx = rng.randrange(len(current_pairs))
                current_pairs = _try_remove_modulus(current_pairs, idx)
    
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
        p = _greedy_build(order, rng)
        if len(p) > len(best_pairs):
            best_pairs = p
    
    # Local search from best
    if len(best_pairs) > 0:
        improved = _local_search(best_pairs, rng, DEADLINE - 5)
        if len(improved) > len(best_pairs):
            best_pairs = improved
    
    # Simulated annealing from best
    if len(best_pairs) > 0:
        improved = _simulate_annealing(best_pairs, rng, DEADLINE - 2)
        if len(improved) > len(best_pairs):
            best_pairs = improved
    
    return best_pairs

# EVOLVE-BLOCK-END
