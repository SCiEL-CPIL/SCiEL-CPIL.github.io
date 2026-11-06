"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import random

_START_TIME = None

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _disjoint(r1, m1, r2, m2):
    """Check if residue classes (r1 mod m1) and (r2 mod m2) are disjoint."""
    g = _gcd(m1, m2)
    if g == 1:
        return False  # always intersect
    return (r2 - r1) % g != 0

def _can_add(r_new, m_new, existing):
    """Check if new class is disjoint from all existing."""
    for (r, m) in existing:
        if not _disjoint(r_new, m_new, r, m):
            return False
    return True

def construct():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    deadline = _START_TIME + 80.0  # safety margin
    
    # Candidate moduli from 2 to 128
    moduli = list(range(2, 129))
    
    # Heuristic: Sort moduli by "connectivity" or "shared factors".
    # Moduli with more common factors with others are easier to place.
    # Let's sort by number of divisors descending, then value descending.
    def divisor_count(n):
        count = 0
        i = 1
        while i * i <= n:
            if n % i == 0:
                count += 1
                if i != n // i:
                    count += 1
            i += 1
        return count

    # Precompute divisor counts
    div_counts = {m: divisor_count(m) for m in moduli}
    
    # Sort: Higher divisor count first (more likely to share factors), then higher value (larger modulus = more residue choices? 
    # Actually, larger modulus means more residues to check, but also larger gcds with multiples.
    # Let's try sorting by divisor count desc, then value desc.
    sorted_moduli = sorted(moduli, key=lambda m: (-div_counts[m], -m))
    
    best_solution = []
    best_size = 0
    
    # Try a deterministic greedy pass first with the sorted order
    current = []
    used_moduli = set()
    for m in sorted_moduli:
        if m in used_moduli:
            continue
        # Try residues in a specific order? 
        # If we pick r=0, it might conflict.
        # Randomizing residues in the first pass might help.
        # But let's try a fixed heuristic first: try all residues, pick the one that minimizes conflicts with FUTURE moduli?
        # Too expensive. Just pick the first valid one.
        # To break ties, we can shuffle residues slightly or just use 0..m-1.
        found = False
        for r in range(m):
            if _can_add(r, m, current):
                current.append([r, m])
                used_moduli.add(m)
                found = True
                break
        if found:
            if len(current) > best_size:
                best_size = len(current)
                best_solution = [x[:] for x in current]
                
    # If time permits, do randomized restarts
    if time.time() < deadline:
        for restart in range(100):
            if time.time() > deadline:
                break
            
            current = []
            used_moduli = set()
            
            # Shuffle the sorted moduli slightly? Or just shuffle the whole list?
            # Shuffling the whole list might lose the benefit of the heuristic.
            # Let's shuffle the residues instead, and keep the modulus order heuristic.
            # Or, use a different heuristic for some restarts.
            
            # Strategy: Randomize the order of moduli, but bias towards high divisor count.
            # Or just shuffle the residues for the greedy selection.
            
            # Let's try: Keep sorted_moduli, but randomize the residue check order.
            for m in sorted_moduli:
                if time.time() > deadline:
                    break
                if m in used_moduli:
                    continue
                    
                residues = list(range(m))
                random.shuffle(residues)
                
                found = False
                for r in residues:
                    if _can_add(r, m, current):
                        current.append([r, m])
                        used_moduli.add(m)
                        found = True
                        break
                        
                if found:
                    if len(current) > best_size:
                        best_size = len(current)
                        best_solution = [x[:] for x in current]
                        
    if not best_solution:
        return [[1, 2]]
        
    return best_solution
# EVOLVE-BLOCK-END
