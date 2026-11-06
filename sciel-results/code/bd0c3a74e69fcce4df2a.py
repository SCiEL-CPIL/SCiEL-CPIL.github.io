# EVOLVE-BLOCK-START
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import random
from functools import lru_cache

_START_TIME = None
MAXIMUM = 128

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
    
    deadline = _START_TIME + 85.0  # safety margin
    
    # The set of moduli must have pairwise gcd > 1.
    # The largest such set is the even numbers.
    target_moduli = [m for m in range(2, 129) if m % 2 == 0]
    
    # Sort by value descending? Or random?
    # Let's try sorting by value descending to handle "constrained" ones?
    # Actually, smaller moduli have fewer residues, so they are more constrained.
    # Sort by modulus ascending.
    sorted_moduli = sorted(target_moduli)
    
    best_solution = []
    best_size = 0
    
    # Run many randomized greedy trials
    # Each trial: try to place all moduli in sorted_moduli.
    # If a modulus fails, skip it (or abort? skipping allows finding a large subset).
    # We want to maximize the count, so skipping failed ones is good.
    
    num_trials = 2000
    for trial in range(num_trials):
        if time.time() > deadline:
            break
            
        current = []
        used_moduli = set()
        
        # Shuffle the order of moduli to avoid bias
        trial_moduli = sorted_moduli[:]
        random.shuffle(trial_moduli)
        
        for m in trial_moduli:
            if time.time() > deadline:
                break
                
            # Try to find a residue
            # Randomize residue order
            residues = list(range(m))
            random.shuffle(residues)
            
            found = False
            for r in residues:
                if _can_add(r, m, current):
                    current.append([r, m])
                    used_moduli.add(m)
                    found = True
                    break
            
            # If not found, we skip this modulus in this trial.
            # This allows us to find a large valid subset.
            
        if len(current) > best_size:
            best_size = len(current)
            best_solution = [x[:] for x in current]
            
            # If we found all 64, we can stop early
            if best_size == len(target_moduli):
                break
                
    if not best_solution:
        # Fallback: just return a small valid set
        return [[0, 2], [1, 4]]
        
    return best_solution
# EVOLVE-BLOCK-END
