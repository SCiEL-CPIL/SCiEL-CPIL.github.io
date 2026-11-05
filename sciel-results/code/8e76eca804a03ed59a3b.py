"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill

    denominators = list(range(2, 49))
    n = len(denominators)
    # Precompute unit fractions
    unit_fracs = [Fraction(1, d) for d in denominators]
    
    # Total sum of all unit fractions
    total_all = sum(unit_fracs)
    
    best_deficit = None
    best_subset = None
    best_sum = None

    # Greedy initialization: take largest fractions that keep sum < 1
    greedy_indices = []
    greedy_sum = Fraction(0)
    # Sort by fraction value descending (i.e., by denominator ascending)
    sorted_indices = sorted(range(n), key=lambda i: -unit_fracs[i])
    for i in sorted_indices:
        if greedy_sum + unit_fracs[i] < 1:
            greedy_indices.append(i)
            greedy_sum += unit_fracs[i]
    
    if greedy_indices:
        best_subset = greedy_indices
        best_sum = greedy_sum
        best_deficit = 1 - greedy_sum

    # Local search: try to improve by swapping or adding/removing elements
    # We want to minimize deficit = 1 - sum, so maximize sum while keeping sum < 1
    
    def try_improve(indices, current_sum, deadline):
        nonlocal best_deficit, best_subset, best_sum
        if time.time() > deadline:
            return indices, current_sum
        
        improved = True
        while improved:
            improved = False
            if time.time() > deadline:
                break
            idx_set = set(indices)
            # Try adding each element not in the set
            for i in range(n):
                if i in idx_set:
                    continue
                new_sum = current_sum + unit_fracs[i]
                if new_sum < 1:
                    # This is an improvement (larger sum, smaller deficit)
                    indices.append(i)
                    idx_set.add(i)
                    current_sum = new_sum
                    if best_deficit is None or (1 - current_sum) < best_deficit:
                        best_deficit = 1 - current_sum
                        best_subset = list(indices)
                        best_sum = current_sum
                    improved = True
                    if time.time() > deadline:
                        return indices, current_sum
            
            # Try swapping: remove one, add another
            if time.time() > deadline:
                break
            for i in list(indices):
                if time.time() > deadline:
                    break
                for j in range(n):
                    if j in idx_set:
                        continue
                    # Swap i out, j in
                    new_sum = current_sum - unit_fracs[i] + unit_fracs[j]
                    if new_sum < 1 and new_sum > current_sum:
                        # Improvement
                        indices[indices.index(i)] = j
                        idx_set.remove(i)
                        idx_set.add(j)
                        current_sum = new_sum
                        if best_deficit is None or (1 - current_sum) < best_deficit:
                            best_deficit = 1 - current_sum
                            best_subset = list(indices)
                            best_sum = current_sum
                        improved = True
                        if time.time() > deadline:
                            return indices, current_sum
                        break  # restart after a swap
    
        return indices, current_sum

    # Start local search from greedy
    curr_indices = list(best_subset)
    curr_sum = best_sum
    curr_indices, curr_sum = try_improve(curr_indices, curr_sum, deadline)

    # Try multiple random restarts if time permits
    import random
    random.seed(42)
    
    for restart in range(200):
        if time.time() > deadline:
            break
        
        # Random subset of moderate size
        size = random.randint(3, 12)
        subset = random.sample(range(n), size)
        s = sum(unit_fracs[i] for i in subset)
        
        if s >= 1:
            # Try to reduce
            # Sort by fraction descending, remove until sum < 1
            subset.sort(key=lambda i: -unit_fracs[i])
            s = Fraction(0)
            valid_subset = []
            for i in subset:
                if s + unit_fracs[i] < 1:
                    valid_subset.append(i)
                    s += unit_fracs[i]
            if not valid_subset:
                continue
            subset = valid_subset
            s = sum(unit_fracs[i] for i in subset)
        
        if s >= 1:
            continue
        
        if best_deficit is None or (1 - s) < best_deficit:
            best_deficit = 1 - s
            best_subset = list(subset)
            best_sum = s
        
        # Local search from this start
        curr_indices, curr_sum = try_improve(list(subset), s, deadline)
    
    # Final check
    if time.time() > deadline:
        pass  # just return what we have

    # Validate best_subset
    if best_subset is None or len(best_subset) == 0:
        # Fallback: just return [0] (denominator 2, sum = 1/2 < 1)
        return [0]
    
    # Ensure distinct and valid
    best_subset = list(set(best_subset))
    final_sum = sum(unit_fracs[i] for i in best_subset)
    if final_sum >= 1:
        # Shouldn't happen, but fallback
        return [0]
    
    return best_subset

# EVOLVE-BLOCK-END
