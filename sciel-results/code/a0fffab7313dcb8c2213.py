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

def _conflict_count(r_new, m_new, existing):
    """Count how many existing classes conflict with the new one."""
    count = 0
    for (r, m) in existing:
        if not _disjoint(r_new, m_new, r, m):
            count += 1
    return count

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
    
    deadline = _START_TIME + 85.0  # safety margin
    
    # Candidate moduli from 2 to 128
    moduli = list(range(2, 129))
    
    # We will try multiple random restarts and different heuristic orderings
    # to find a large set of disjoint classes.
    
    best_solution = []
    best_size = 0
    
    # Heuristic: Sort moduli by number of prime factors? Or just random?
    # Let's try a greedy approach where we pick the modulus that leaves the most options open.
    # Actually, a simpler and often effective heuristic for packing is to pick the one with the "hardest" constraints first, 
    # but here disjointness is the constraint.
    
    # Let's use a randomized greedy approach with multiple restarts.
    # For each restart, we shuffle the moduli and residues.
    
    for restart in range(200): # Try up to 200 restarts if time permits
        if time.time() > deadline:
            break
            
        current = []
        used_moduli = set()
        
        # Shuffle moduli to break symmetry
        moduli_order = moduli[:]
        random.shuffle(moduli_order)
        
        for m in moduli_order:
            if time.time() > deadline:
                break
            if m in used_moduli:
                continue
                
            # Try all residues for this modulus
            # We want to pick a residue that is disjoint from all existing.
            # To maximize future flexibility, we could try to pick one that conflicts with the fewest *potential* future moduli?
            # That's expensive. Let's just pick the first valid residue, or randomize them.
            
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
        
        # If we found a solution of size 15+ (since 2.14 score suggests ~15 classes), we might stop early?
        # But let's keep trying to maximize.
        
    # If we haven't found anything (unlikely), return a default
    if not best_solution:
        return [[1, 2]]
        
    return best_solution
# EVOLVE-BLOCK-END
