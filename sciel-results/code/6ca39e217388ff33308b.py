"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math

def construct():
    n_elements = 12
    universe_size = 1 << n_elements  # 4096
    family_size = 40
    
    all_masks = list(range(universe_size))
    
    def count_comparable(family):
        count = 0
        n = len(family)
        for i in range(n):
            a = family[i]
            for j in range(i + 1, n):
                b = family[j]
                if (a & b) == a or (a & b) == b:
                    count += 1
        return count

    # Greedy Construction
    family = []
    remaining = set(all_masks)
    
    for _ in range(family_size):
        best_candidate = None
        best_score = -1
        # Optimization: We can pre-compare or just iterate. 
        # 4096 * 40 (avg family size) * 40 (iterations) is ~ 6.5M ops.
        for c in remaining:
            score = 0
            for f in family:
                if (c & f) == c or (c & f) == f:
                    score += 1
            if score > best_score:
                best_score = score
                best_candidate = c
        
        if best_candidate is not None:
            family.append(best_candidate)
            remaining.remove(best_candidate)
        else:
            # Fallback if something goes wrong (shouldn't happen)
            candidate = random.choice(all_masks)
            while candidate in family:
                candidate = random.choice(all_masks)
            family.append(candidate)
            remaining.remove(candidate)

    current_family = family
    current_score = count_comparable(current_family)
    
    # Local Search
    # We try to swap any element in the family with any element not in the family
    # to see if it increases the total comparable count.
    
    in_family = set(current_family)
    not_in_family = list(remaining)
    
    improved = True
    max_passes = 100 # Safety limit
    passes = 0
    
    while improved and passes < max_passes:
        improved = False
        passes += 1
        for i in range(family_size):
            old_set = current_family[i]
            # Calculate contribution of old_set to total score
            # It's easier to just recalculate total score for candidate swaps if N is small
            # But we can be smarter.
            # Let's just try all swaps.
            for new_set in not_in_family:
                if new_set == old_set:
                    continue
                # Tentative swap
                candidate_family = current_family.copy()
                candidate_family[i] = new_set
                
                # Calculate new score
                # Optimization: We only need to recompute pairs involving index i
                # Pairs not involving i remain same.
                # Old contribution of i: sum(1 for j != i if comparable(i, j))
                # New contribution of i: sum(1 for j != i if comparable(new_set, j))
                
                old_contrib = 0
                new_contrib = 0
                for j in range(family_size):
                    if i == j:
                        continue
                    other = current_family[j]
                    if (old_set & other) == old_set or (old_set & other) == other:
                        old_contrib += 1
                    if (new_set & other) == new_set or (new_set & other) == other:
                        new_contrib += 1
                
                if new_contrib > old_contrib:
                    # Accept swap
                    current_family[i] = new_set
                    in_family.remove(old_set)
                    in_family.add(new_set)
                    # Update not_in_family list (remove new_set, add old_set)
                    # This is O(N) to update list, which is fine.
                    not_in_family.remove(new_set)
                    not_in_family.append(old_set)
                    current_score += (new_contrib - old_contrib)
                    improved = True
                    break # Restart i loop to handle interactions
                    
    result = []
    for mask in current_family:
        subset = []
        for i in range(n_elements):
            if mask & (1 << i):
                subset.append(i)
        result.append(subset)
        
    return result
# EVOLVE-BLOCK-END
