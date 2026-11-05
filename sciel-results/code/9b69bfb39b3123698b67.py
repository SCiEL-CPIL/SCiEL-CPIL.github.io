"""Improved program for odd covering systems with moduli dividing 51975."""
import time
import numpy as np
import random

# EVOLVE-BLOCK-START
def construct():
    base = 51975
    deadline = time.time() + 140  # safety margin before 150s limit
    
    # Precompute all odd divisors of 51975 > 1, sorted in increasing order
    divisors = [d for d in range(3, base + 1, 2) if base % d == 0]
    n_div = len(divisors)
    
    def count_uncovered(covered):
        return int(np.sum(covered == 0))
    
    # Greedy initialization: process moduli in increasing order
    moduli = list(divisors)  # sorted increasing
    residues = [0] * n_div
    covered = np.zeros(base, dtype=np.uint8)
    
    for i, m in enumerate(moduli):
        if time.time() > deadline:
            break
        
        best_r = 0
        best_count = -1
        for r in range(m):
            positions = np.arange(r, base, m)
            count = int(np.sum(covered[positions] == 0))
            if count > best_count:
                best_count = count
                best_r = r
        
        residues[i] = best_r
        covered[best_r::m] = 1
    
    current_uncovered = count_uncovered(covered)
    best_uncovered = current_uncovered
    best_residues = residues[:]
    
    # Local search: try swapping residues of pairs
    temp = 100.0
    cooling = 0.99
    
    # Precompute for each modulus the list of positions for each residue
    # This is too much memory, so we'll compute on the fly but cache small moduli
    
    while time.time() < deadline - 5:
        if temp < 0.5:
            break
        
        # Pick two random moduli
        i, j = random.sample(range(n_div), 2)
        mi = moduli[i]
        mj = moduli[j]
        
        old_ri = residues[i]
        old_rj = residues[j]
        
        # Pick new residues
        if random.random() < 0.5:
            new_ri = random.randint(0, mi - 1)
            new_rj = random.randint(0, mj - 1)
        else:
            # Try swapping the residues (if moduli are compatible)
            new_ri = old_rj % mi
            new_rj = old_ri % mj
        
        # Compute delta efficiently
        # Positions affected by changing modulus i from old_ri to new_ri:
        # pos_i_old = {old_ri, old_ri+mi, old_ri+2mi, ...}
        # pos_i_new = {new_ri, new_ri+mi, new_ri+2mi, ...}
        # Similarly for j.
        
        # We need to count:
        # - positions that were covered by i (old) but not by j (old) and become uncovered
        # - positions that become covered by i (new) or j (new) that were uncovered
        
        # This is complex due to overlaps. Let's just recompute for the union of affected positions.
        
        # Get all positions that might change
        pos_i_old = set(range(old_ri, base, mi))
        pos_i_new = set(range(new_ri, base, mi))
        pos_j_old = set(range(old_rj, base, mj))
        pos_j_new = set(range(new_rj, base, mj))
        
        # Positions that might change are in the symmetric difference of (pos_i_old ∪ pos_j_old) and (pos_i_new ∪ pos_j_new)
        old_union = pos_i_old | pos_j_old
        new_union = pos_i_new | pos_j_new
        affected = old_union.symmetric_difference(new_union)
        
        if not affected:
            continue
        
        # For each affected position, determine if it's covered before and after
        # Before: covered by i_old, j_old, or others
        # After: covered by i_new, j_new, or others
        
        # We need to check if any OTHER modulus covers this position
        # This is expensive. Let's use a different approach.
        
        # Just recompute the covered array for the affected region
        # But we need the full covered array to know if others cover it.
        
        # Alternative: compute the change in coverage count
        # delta = (newly covered) - (newly uncovered)
        
        # A position p is:
        # - newly covered: p in new_union, p not in old_union, and p not covered by others
        #   Wait, that's not right either.
        
        # Let me think again. The covered array has contributions from all moduli.
        # When we change i and j, a position p's coverage changes only if:
        # 1. p is in pos_i_old or pos_i_new (i's contribution changes)
        # 2. p is in pos_j_old or pos_j_new (j's contribution changes)
        # 3. p was covered only by i or j (not by others)
        
        # This is getting complex. Let's just do a full recompute for small changes.
        
        # Actually, let's just copy and modify
        old_covered = covered.copy()
        
        covered[old_ri::mi] = 0
        covered[old_rj::mj] = 0
        covered[new_ri::mi] = 1
        covered[new_rj::mj] = 1
        
        new_uncovered = count_uncovered(covered)
        delta = new_uncovered - current_uncovered
        
        if delta < 0 or random.random() < np.exp(-delta / temp):
            residues[i] = new_ri
            residues[j] = new_rj
            current_uncovered = new_uncovered
            
            if current_uncovered < best_uncovered:
                best_uncovered = current_uncovered
                best_residues = residues[:]
        else:
            covered[:] = old_covered
        
        temp *= cooling
    
    # Return the best solution found
    classes = [[moduli[i], best_residues[i]] for i in range(n_div)]
    return classes
# EVOLVE-BLOCK-END
