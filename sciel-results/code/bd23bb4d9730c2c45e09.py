# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    denominators = list(range(2, 49))  # [2, 3, ..., 48]
    n = len(denominators)
    
    # Precompute reciprocals as Fractions for exact arithmetic
    reciprocals = [Fraction(1, d) for d in denominators]
    
    # Beam search: maintain top-K candidate subsets
    # Each beam entry: (deficit, frozenset_of_indices, sum_fraction)
    BEAM_WIDTH = 5000
    
    # Initialize with all single-element subsets
    initial_beams = []
    for i in range(n):
        s = reciprocals[i]
        if s < 1:
            deficit = 1 - s
            initial_beams.append((deficit, frozenset([i]), s))
    
    # Sort by deficit (ascending) and keep top BEAM_WIDTH
    initial_beams.sort(key=lambda x: x[0])
    beams = initial_beams[:BEAM_WIDTH]
    
    best_indices = list(beams[0][1]) if beams else []
    best_deficit = beams[0][0] if beams else Fraction(1)
    
    # Process elements in order, at each step try adding next element to each beam
    # We process denominators from small to large (indices 0 to n-1)
    # At step i, we consider adding element i to beams that don't already contain it
    
    # To make this tractable, we'll do iterative deepening:
    # For each element index, expand all beams by optionally including that element
    
    current_beams = beams[:]
    
    for elem_idx in range(n):
        if time.time() > deadline - 5:
            break
        
        new_beams = []
        elem_frac = reciprocals[elem_idx]
        
        for deficit, index_set, total in current_beams:
            # Option 1: don't include elem_idx
            new_beams.append((deficit, index_set, total))
            
            # Option 2: include elem_idx (if not already present)
            if elem_idx not in index_set:
                new_total = total + elem_frac
                if new_total < 1:
                    new_deficit = 1 - new_total
                    new_beams.append((new_deficit, index_set | {elem_idx}, new_total))
        
        # Sort and prune to BEAM_WIDTH
        new_beams.sort(key=lambda x: x[0])
        current_beams = new_beams[:BEAM_WIDTH]
        
        # Update best
        if current_beams:
            if current_beams[0][0] < best_deficit:
                best_deficit = current_beams[0][0]
                best_indices = list(current_beams[0][1])
    
    # Final local search on the best solution found
    if time.time() < deadline - 10:
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            current_set = set(best_indices)
            current_sum = sum(reciprocals[i] for i in best_indices)
            
            # Try adding one element
            for add_idx in range(n):
                if add_idx in current_set:
                    continue
                new_sum = current_sum + reciprocals[add_idx]
                if new_sum < 1:
                    new_deficit = 1 - new_sum
                    if new_deficit < best_deficit:
                        best_indices.append(add_idx)
                        best_deficit = new_deficit
                        improved = True
                        break
            if improved:
                continue
            
            # Try 1-swap
            for remove_idx in list(current_set):
                for add_idx in range(n):
                    if add_idx in current_set:
                        continue
                    new_sum = current_sum - reciprocals[remove_idx] + reciprocals[add_idx]
                    if new_sum < 1:
                        new_deficit = 1 - new_sum
                        if new_deficit < best_deficit:
                            best_indices = [i for i in best_indices if i != remove_idx]
                            best_indices.append(add_idx)
                            best_deficit = new_deficit
                            improved = True
                            break
                if improved:
                    break
    
    # Final check
    if not best_indices:
        return [0]
    
    return best_indices
# EVOLVE-BLOCK-END
