"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random

MAX_MOD = 128
DEADLINE = 85.0

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

def construct():
    start_time = time.time()
    rng = random.Random(42)
    
    all_moduli = list(range(2, MAX_MOD + 1))
    
    # Pre-filter: We know coprime moduli cannot coexist.
    # But greedy handles it.
    
    best_pairs = []
    
    # Strategy 1: Greedy with specific orderings
    # Evens first (most likely to be compatible)
    evens = [m for m in all_moduli if m % 2 == 0]
    odds = [m for m in all_moduli if m % 2 != 0]
    
    # Order 1: Evens descending
    order1 = sorted(evens, reverse=True) + sorted(odds, reverse=True)
    p1 = _greedy_build(order1, None)
    if len(p1) > len(best_pairs):
        best_pairs = p1
        
    # Order 2: Evens ascending
    order2 = sorted(evens) + sorted(odds)
    p2 = _greedy_build(order2, None)
    if len(p2) > len(best_pairs):
        best_pairs = p2
        
    # Order 3: Randomized Greedy
    for _ in range(100):
        if time.time() - start_time > DEADLINE - 5:
            break
        order = all_moduli[:]
        rng.shuffle(order)
        # Bias towards evens
        evens_in_order = [m for m in order if m % 2 == 0]
        odds_in_order = [m for m in order if m % 2 != 0]
        order = evens_in_order + odds_in_order
        p = _greedy_build(order, rng)
        if len(p) > len(best_pairs):
            best_pairs = p

    # Local Search / Hill Climbing
    # Try to improve best_pairs
    current_pairs = best_pairs[:]
    current_size = len(current_pairs)
    
    # Precompute gcds for all pairs of moduli in all_moduli? 
    # Or just use math.gcd on the fly.
    
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

    # Iterative improvement
    for step in range(10000):
        if time.time() - start_time > DEADLINE - 2:
            break
            
        # Randomly pick an action
        action = rng.random()
        
        if action < 0.5:
            # Try to add a random modulus not in current set
            current_mods = [m for _, m in current_pairs]
            available = [m for m in all_moduli if m not in current_mods]
            if not available:
                continue
            m = rng.choice(available)
            r = _try_add_modulus(current_pairs, m, rng)
            if r is not None:
                new_pairs = current_pairs + [[r, m]]
                if len(new_pairs) > current_size:
                    current_pairs = new_pairs
                    current_size = len(current_pairs)
                    if len(current_pairs) > len(best_pairs):
                        best_pairs = current_pairs[:]
        elif action < 0.8:
            # Try to change a residue
            if not current_pairs:
                continue
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
            # We don't gain size here, but it's a move in search space.
        else:
            # Try to remove a modulus to free up constraints?
            # Only useful if we can add 2 more later.
            if len(current_pairs) > 0:
                idx = rng.randrange(len(current_pairs))
                current_pairs = _try_remove_modulus(current_pairs, idx)
                current_size = len(current_pairs)
                
    return best_pairs

# EVOLVE-BLOCK-END
