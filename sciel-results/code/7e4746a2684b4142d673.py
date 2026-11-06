import math
import time

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin from 150s
    
    # Candidates are 1..511
    # Weights are 1/(512-a)
    # Sort by weight descending (which means a descending)
    candidates = list(range(511, 0, -1))
    
    # Precompute weights for fast access
    weights = {a: 1.0 / (512.0 - a) for a in candidates}
    
    # Greedy Selection
    selected = []
    for a in candidates:
        # Check if a is coprime to all already selected
        is_coprime = True
        for b in selected:
            if math.gcd(a, b) != 1:
                is_coprime = False
                break
        if is_coprime:
            selected.append(a)
            
    # Calculate initial score
    current_score = sum(weights[a] for a in selected)
    
    # Local Search: Try swapping one element out and one in
    # We represent the set for fast lookup
    selected_set = set(selected)
    
    improved = True
    while improved and (time.time() - start_time) < deadline:
        improved = False
        best_swap = None
        best_new_score = current_score
        
        # For each element in selected, try removing it
        for a_out in selected:
            # Remaining set
            remaining = [b for b in selected if b != a_out]
            remaining_set = set(remaining)
            
            # Try adding any candidate not in selected
            # To be efficient, we can iterate through candidates in descending weight order
            # and break early if potential gain is too small, but simple iteration is fine
            for a_in in candidates:
                if a_in in selected_set:
                    continue
                    
                # Check if a_in is coprime to all in remaining
                is_coprime = True
                for b in remaining:
                    if math.gcd(a_in, b) != 1:
                        is_coprime = False
                        break
                
                if is_coprime:
                    new_score = current_score - weights[a_out] + weights[a_in]
                    if new_score > best_new_score + 1e-12:
                        best_new_score = new_score
                        best_swap = (a_out, a_in)
                        improved = True
        
        if improved:
            a_out, a_in = best_swap
            selected.remove(a_out)
            selected.append(a_in)
            selected_set.discard(a_out)
            selected_set.add(a_in)
            current_score = best_new_score
    
    return selected
