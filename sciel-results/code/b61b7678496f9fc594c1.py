import time
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    time_limit = 55.0  # Safety margin for 60s limit
    
    denominators = list(range(2, 49))
    n = len(denominators)
    reciprocals = [1.0 / d for d in denominators]
    
    # Helper to calculate exact deficit using Fraction for final verification or comparison if needed
    # But for search, we use floats.
    
    def get_deficit_float(indices):
        return 1.0 - sum(reciprocals[i] for i in indices)
    
    def is_valid_float(indices, eps=1e-12):
        return sum(reciprocals[i] for i in indices) < 1.0 - eps

    # 1. Generate Initial Candidates
    
    # Candidate A: Greedy Ascending (smallest denominators first)
    cand_a = []
    sum_a = 0.0
    for i in range(n):
        if sum_a + reciprocals[i] < 1.0 - 1e-12:
            cand_a.append(i)
            sum_a += reciprocals[i]
            
    # Candidate B: Greedy Descending (largest denominators first)
    cand_b = []
    sum_b = 0.0
    for i in range(n - 1, -1, -1):
        if sum_b + reciprocals[i] < 1.0 - 1e-12:
            cand_b.append(i)
            sum_b += reciprocals[i]
            
    # Candidate C: Start with all, remove largest until valid
    cand_c = list(range(n))
    sum_c = sum(reciprocals)
    # Sort indices by reciprocal descending (denominator ascending) to remove largest first
    sorted_indices = sorted(range(n), key=lambda i: -reciprocals[i])
    for i in sorted_indices:
        if sum_c < 1.0 - 1e-12:
            break
        if i in cand_c:
            cand_c.remove(i)
            sum_c -= reciprocals[i]
            
    # Candidate D: Random Greedy
    random.seed(42)
    cand_d = []
    sum_d = 0.0
    perm = list(range(n))
    random.shuffle(perm)
    for i in perm:
        if sum_d + reciprocals[i] < 1.0 - 1e-12:
            cand_d.append(i)
            sum_d += reciprocals[i]

    candidates = [
        (cand_a, sum_a),
        (cand_b, sum_b),
        (cand_c, sum_c),
        (cand_d, sum_d)
    ]
    
    # Filter valid candidates
    valid_candidates = []
    for c, s in candidates:
        if s < 1.0 - 1e-12 and len(c) > 0:
            valid_candidates.append((c, s))
            
    if not valid_candidates:
        # Fallback: return a single small fraction
        return [n - 1] # index for 48
        
    # Select best initial candidate
    best_set, best_sum = min(valid_candidates, key=lambda x: 1.0 - x[1])
    best_set = set(best_set)
    best_deficit = 1.0 - best_sum
    
    # Local Search
    current_set = set(best_set)
    current_sum = best_sum
    
    while time.time() - start_time < time_limit:
        improved = False
        
        # Try adding items
        for j in range(n):
            if j not in current_set:
                new_sum = current_sum + reciprocals[j]
                if new_sum < 1.0 - 1e-12:
                    new_deficit = 1.0 - new_sum
                    if new_deficit < best_deficit:
                        current_set.add(j)
                        current_sum = new_sum
                        best_deficit = new_deficit
                        best_set = set(current_set)
                        improved = True
        
        # Try swapping items
        # To speed up, we can limit the number of swaps or just do it all
        if not improved:
            # If no single add improved, try swaps
            # We need to iterate over items in current_set and items not in current_set
            # To avoid modifying set during iteration, we make copies
            items_in = list(current_set)
            items_out = [i for i in range(n) if i not in current_set]
            
            for i in items_in:
                for j in items_out:
                    new_sum = current_sum - reciprocals[i] + reciprocals[j]
                    if new_sum < 1.0 - 1e-12:
                        new_deficit = 1.0 - new_sum
                        if new_deficit < best_deficit:
                            current_set.remove(i)
                            current_set.add(j)
                            current_sum = new_sum
                            best_deficit = new_deficit
                            best_set = set(current_set)
                            improved = True
                            # Break out of inner loop? Or continue? 
                            # Breaking inner loop is safer to re-evaluate state
                            break
                if improved:
                    break
            
            if improved:
                continue
        
        # If no improvement found in add or swap, try a perturbation
        if not improved:
            # Random restart or perturbation
            # Remove 2 random items, add 2 random items
            if len(current_set) > 2:
                to_remove = random.sample(list(current_set), 2)
                for i in to_remove:
                    current_set.remove(i)
                    current_sum -= reciprocals[i]
                
                # Try to add back some items
                for j in range(n):
                    if j not in current_set:
                        if current_sum + reciprocals[j] < 1.0 - 1e-12:
                            current_set.add(j)
                            current_sum += reciprocals[j]
            
            # If perturbation didn't help or wasn't possible, try a new random greedy
            if time.time() - start_time > time_limit - 5:
                break
                
            # Occasionally restart with a new random greedy to escape deep local minima
            if random.random() < 0.1:
                cand_new = []
                sum_new = 0.0
                perm = list(range(n))
                random.shuffle(perm)
                for i in perm:
                    if sum_new + reciprocals[i] < 1.0 - 1e-12:
                        cand_new.append(i)
                        sum_new += reciprocals[i]
                
                if cand_new and sum_new < 1.0 - 1e-12:
                    new_deficit = 1.0 - sum_new
                    if new_deficit < best_deficit:
                        best_set = set(cand_new)
                        best_deficit = new_deficit
                        current_set = set(best_set)
                        current_sum = best_sum
                    else:
                        # Keep current, but maybe mix?
                        pass

    # Final verification with Fraction to ensure validity
    # The task requires STRICTLY BELOW 1.
    # If float says < 1 but fraction says >= 1, it's invalid.
    # Given the precision of 1e-12 and small N, this is unlikely, but let's check the best candidate.
    
    final_indices = list(best_set)
    exact_sum = sum(Fraction(1, denominators[i]) for i in final_indices)
    
    if exact_sum >= 1:
        # Fallback to a known valid small set
        # e.g., just index for 48
        return [n - 1]
        
    return final_indices
