import numpy as np
import random
import time

def construct():
    base = 51975
    # Get all divisors > 1 of base
    divisors = [d for d in range(2, base + 1) if base % d == 0]
    # We use all available divisors to maximize coverage potential
    # Shuffle to randomize order for initialization? No, order doesn't matter for final set.
    
    best_score = base
    best_classes = None
    
    # Time limit check
    start_time = time.time()
    time_limit = 100.0 # Leave 20s buffer
    
    # Number of restarts
    # Each restart: Initialize random residues, run hill climbing
    # Hill climbing: Randomly change residue of a random modulus, accept if better
    
    for restart in range(50): # Try up to 50 restarts or until time limit
        if time.time() - start_time > time_limit:
            break
            
        # Initialize
        classes = [[m, random.randint(0, m - 1)] for m in divisors]
        cover_count = np.zeros(base, dtype=np.int32)
        
        # Apply initial classes
        for m, r in classes:
            cover_count[r::m] += 1
            
        uncovered_count = np.sum(cover_count == 0)
        
        # Hill Climbing
        no_improve_count = 0
        max_no_improve = 10000 # Stop if stuck
        
        while no_improve_count < max_no_improve:
            if time.time() - start_time > time_limit:
                break
                
            # Pick a random class
            idx = random.randint(0, len(classes) - 1)
            m, r_old = classes[idx]
            
            # Pick a random new residue
            r_new = random.randint(0, m - 1)
            if r_new == r_old:
                continue
                
            # Calculate change
            # Indices in old slice: r_old::m
            # Indices in new slice: r_new::m
            
            old_slice = slice(r_old, base, m)
            new_slice = slice(r_new, base, m)
            
            # We need to know how many 0s and 1s are in these slices relative to the rest.
            # Change in cover_count for old_slice: -1
            # Change in cover_count for new_slice: +1
            
            # Count transitions
            # 1. In old_slice:
            #    If cover_count was 1, it becomes 0 -> uncovered +1
            #    If cover_count was >1, it becomes >0 -> uncovered unchanged
            #    (cover_count cannot be 0 because this class was covering it)
            
            # 2. In new_slice:
            #    If cover_count was 0, it becomes 1 -> uncovered -1
            #    If cover_count was >0, it becomes >0 -> uncovered unchanged
            
            # Note: If old_slice and new_slice overlap, we must be careful.
            # But since m is the same, and r_old != r_new, the slices are disjoint?
            # No! $r_1 \pmod m$ and $r_2 \pmod m$ with $r_1 \neq r_2$ are disjoint sets of indices?
            # Yes, indices $x$ such that $x \equiv r_1 \pmod m$ and $x \equiv r_2 \pmod m$ are empty if $r_1 \neq r_2$.
            # So the sets of indices are disjoint.
            
            old_vals = cover_count[old_slice]
            new_vals = cover_count[new_slice]
            
            # Transitions in old_slice (decrement by 1)
            # 1 -> 0: count of 1s in old_slice
            lost_coverage = np.sum(old_vals == 1)
            
            # Transitions in new_slice (increment by 1)
            # 0 -> 1: count of 0s in new_slice
            gained_coverage = np.sum(new_vals == 0)
            
            delta_uncovered = lost_coverage - gained_coverage
            
            if delta_uncovered < 0:
                # Improvement
                # Apply change
                cover_count[old_slice] -= 1
                cover_count[new_slice] += 1
                uncovered_count += delta_uncovered
                classes[idx][1] = r_new
                no_improve_count = 0
            else:
                no_improve_count += 1
                
        # End Hill Climbing
        if uncovered_count < best_score:
            best_score = uncovered_count
            best_classes = [c[:] for c in classes] # Deep copy
            
        # If we found 0, we can stop
        if best_score == 0:
            break
            
    return best_classes
