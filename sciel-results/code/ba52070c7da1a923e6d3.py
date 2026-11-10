"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from math import gcd
from itertools import product

# EVOLVE-BLOCK-START
_BASE = 51975
_DEADLINE = 150.0
_MARGIN = 5.0  # seconds safety margin

def _divisors_odd():
    """Return all odd divisors > 1 of 51975, sorted."""
    divs = []
    for d in range(3, _BASE + 1, 2):
        if _BASE % d == 0:
            divs.append(d)
    return divs

def _count_uncovered(covered, m, r):
    """Count how many uncovered residues would be covered by class (m, r)."""
    return int(np.sum(covered[r::_BASE * 0 + m] == 0))

def construct():
    start_time = time.time()
    deadline = start_time + _DEADLINE - _MARGIN

    base = _BASE
    all_moduli = _divisors_odd()
    
    # Greedy initial solution: moduli in increasing order
    covered = np.zeros(base, dtype=np.uint8)
    classes = []
    for m in all_moduli:
        # For each residue r in 0..m-1, count uncovered in positions r, r+m, r+2m, ...
        best_r = 0
        best_count = -1
        for r in range(m):
            c = int(np.sum(covered[r::m] == 0))
            if c > best_count:
                best_count = c
                best_r = r
        classes.append([m, best_r])
        covered[best_r::m] = 1
    
    current_score = int(np.sum(covered == 0))
    best_classes = [list(c) for c in classes]
    best_score = current_score
    
    # Local search: try swapping residues of existing classes, or adding/removing classes
    # We have distinct moduli; we can change which residue r is used for each m.
    # Also consider subset of moduli (not all need to be used).
    
    rng = random.Random(42)
    
    # Build a mapping from modulus to its index in classes
    def _score(covered_arr):
        return int(np.sum(covered_arr == 0))
    
    # Try to improve by changing residues of individual moduli
    # Order moduli by descending size for potential impact
    order = sorted(range(len(all_moduli)), key=lambda i: -all_moduli[i])
    
    improved = True
    while improved and (time.time() < deadline):
        improved = False
        for idx in order:
            if time.time() >= deadline:
                break
            m = all_moduli[idx]
            # Try all residues for this modulus
            best_r = classes[idx][1]
            best_local = current_score
            for r in range(m):
                # Temporarily set this class to (m, r)
                # Clear current class contribution, then set new
                old_r = classes[idx][1]
                if r == old_r:
                    continue
                # Clear old
                covered[old_r::m] = 0
                # Set new
                covered[r::m] = 1
                new_score = _score(covered)
                if new_score < best_local:
                    best_local = new_score
                    best_r = r
                # Restore
                covered[old_r::m] = 0
                covered[classes[idx][1] if r != classes[idx][1] else old_r::m] = 1
            # Apply best
            if best_r != classes[idx][1]:
                old_r = classes[idx][1]
                covered[old_r::m] = 0
                classes[idx] = [m, best_r]
                covered[best_r::m] = 1
                current_score = best_local
                if current_score < best_score:
                    best_score = current_score
                    best_classes = [list(c) for c in classes]
                improved = True
    
    # Try adding/removing moduli: start from best and see if removing a modulus helps
    # (it won't reduce uncovered count, but let's check if a smaller set suffices)
    # Actually removing a class can only increase or keep same uncovered count.
    # So the greedy full set is at least as good as any subset in terms of score.
    
    # Try local search on the best solution found
    if best_score < current_score:
        # Reconstruct from best
        classes = [list(c) for c in best_classes]
        current_score = best_score
        covered = np.zeros(base, dtype=np.uint8)
        for m, r in classes:
            covered[r::m] = 1
    
    # Second pass of local search on the best solution
    improved = True
    while improved and (time.time() < deadline):
        improved = False
        for idx in order:
            if time.time() >= deadline:
                break
            m = all_moduli[idx]
            best_r = classes[idx][1]
            best_local = current_score
            for r in range(m):
                if r == best_r:
                    continue
                old_r = classes[idx][1]
                covered[old_r::m] = 0
                covered[r::m] = 1
                new_score = _score(covered)
                if new_score < best_local:
                    best_local = new_score
                    best_r = r
                covered[old_r::m] = 0
                covered[classes[idx][1] if r != classes[idx][1] else old_r::m] = 1
            if best_r != classes[idx][1]:
                old_r = classes[idx][1]
                covered[old_r::m] = 0
                classes[idx] = [m, best_r]
                covered[best_r::m] = 1
                current_score = best_local
                if current_score < best_score:
                    best_score = current_score
                    best_classes = [list(c) for c in classes]
                improved = True
    
    # Return the best solution found
    return best_classes
# EVOLVE-BLOCK-END
