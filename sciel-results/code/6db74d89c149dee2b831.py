import time
import random
from fractions import Fraction

def construct():
    # Denominators are 2 to 48
    # Indices 0 to 46 correspond to d = 2 to 48
    denominators = list(range(2, 49))
    n = len(denominators)
    reciprocals = [1.0 / d for d in denominators]
    
    # Precompute exact fractions for final check
    exact_reciprocals = [Fraction(1, d) for d in denominators]
    
    # Deadline guard
    start_time = time.time()
    time_limit = 50.0  # Keep 10s margin
    
    # Initial greedy solution
    selected = []
    current_sum_float = 0.0
    for i in range(n):
        if current_sum_float + reciprocals[i] < 1.0:
            selected.append(i)
            current_sum_float += reciprocals[i]
    
    selected_set = set(selected)
    best_sum_float = current_sum_float
    best_selected = selected.copy()
    
    def get_sum_float(indices):
        return sum(reciprocals[i] for i in indices)
        
    def is_valid(indices):
        # Check strictly below 1 using float first for speed
        s = sum(reciprocals[i] for i in indices)
        if s >= 1.0:
            return False
        # Verify with Fraction to be safe against floating point errors near 1
        # This is expensive, so only do it periodically or at the end?
        # Given n=47, summing 47 fractions is fast.
        # But doing it every step might be slow.
        # Let's rely on float for search, and verify at the end.
        # However, if float says < 1 but fraction is >= 1, we are in trouble.
        # The gap is likely small. 1/d for d<=48 is at least 1/48 ~ 0.02.
        # Float precision is ~1e-16. So float is safe for "strictly below" unless sum is extremely close to 1.
        # We will verify the final result.
        return True

    # Local Search
    while time.time() - start_time < time_limit:
        improved = False
        
        # 1. Try to add the best fitting element
        best_add_idx = -1
        best_new_sum = current_sum_float
        for j in range(n):
            if j not in selected_set:
                new_sum = current_sum_float + reciprocals[j]
                if new_sum < 1.0 and new_sum > best_new_sum:
                    best_new_sum = new_sum
                    best_add_idx = j
        
        if best_add_idx != -1:
            selected.append(best_add_idx)
            selected_set.add(best_add_idx)
            current_sum_float = best_new_sum
            improved = True
            if current_sum_float > best_sum_float:
                best_sum_float = current_sum_float
                best_selected = selected.copy()
            continue # Restart loop to try adding more
        
        # 2. Try to swap
        # For each selected i, try to replace with unselected j
        swap_made = False
        for i in list(selected_set):
            temp_sum = current_sum_float - reciprocals[i]
            best_swap_j = -1
            best_swap_sum = temp_sum # We want to beat current_sum_float
            for j in range(n):
                if j not in selected_set or j == i:
                    new_sum = temp_sum + reciprocals[j]
                    if new_sum < 1.0 and new_sum > best_swap_sum:
                        best_swap_sum = new_sum
                        best_swap_j = j
            
            if best_swap_j != -1 and best_swap_sum > current_sum_float:
                # Perform swap
                selected_set.remove(i)
                selected_set.add(best_swap_j)
                selected = list(selected_set)
                current_sum_float = best_swap_sum
                improved = True
                swap_made = True
                if current_sum_float > best_sum_float:
                    best_sum_float = current_sum_float
                    best_selected = selected.copy()
                break # Restart swap search after a swap
        
        if swap_made:
            continue
            
        if not improved:
            # Stuck in local optimum, perturb
            # Remove a random element
            if selected:
                idx_to_remove = random.choice(selected)
                selected.remove(idx_to_remove)
                selected_set.remove(idx_to_remove)
                current_sum_float = sum(reciprocals[i] for i in selected)
                # This might drop sum significantly, but allows escape
            else:
                # If empty, restart with greedy
                selected = []
                selected_set = set()
                current_sum_float = 0.0
                for i in range(n):
                    if current_sum_float + reciprocals[i] < 1.0:
                        selected.append(i)
                        selected_set.add(i)
                        current_sum_float += reciprocals[i]

    # Final verification and cleanup
    # Ensure the best found is valid
    # Re-calculate exact sum for best_selected
    exact_sum = sum(exact_reciprocals[i] for i in best_selected)
    if exact_sum < 1:
        return best_selected
    else:
        # Fallback to a simple valid solution if something went wrong
        # Greedy from start
        selected_fallback = []
        total = Fraction(0)
        for i, d in enumerate(denominators):
            if total + Fraction(1, d) < 1:
                selected_fallback.append(i)
                total += Fraction(1, d)
        return selected_fallback
