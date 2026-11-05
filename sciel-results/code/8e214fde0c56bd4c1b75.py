"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    DEADLINE = 85.0
    
    denominators = list(range(2, 49))
    n = len(denominators)
    reciprocals = [Fraction(1, d) for d in denominators]
    
    # Sort by reciprocal descending (larger reciprocals first) for better pruning
    order = sorted(range(n), key=lambda i: -reciprocals[i])
    sorted_recs = [reciprocals[i] for i in order]
    
    # Suffix sums of sorted reciprocals
    suffix_sum = [Fraction(0)] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + sorted_recs[i]
    
    # Greedy initial solution (in original order for the answer)
    best_deficit = None
    best_indices = None
    selected = []
    total = Fraction(0)
    for i in range(n):
        if total + reciprocals[i] < 1:
            selected.append(i)
            total += reciprocals[i]
    if selected:
        best_deficit = 1 - total
        best_indices = selected[:]
    best_deficit_val = best_deficit
    
    # Branch and bound on sorted order
    # We track which original indices we've selected
    best_indices_set = set(best_indices) if best_indices else set()
    
    def solve(pos, current_sum, selected_set):
        nonlocal best_deficit_val, best_indices, best_deficit, best_indices_set
        
        if time.time() > DEADLINE:
            return
        
        if pos == n:
            if current_sum < 1 and selected_set:
                deficit = 1 - current_sum
                if best_deficit_val is None or deficit < best_deficit_val:
                    best_deficit_val = deficit
                    best_deficit = deficit
                    best_indices = [order[i] for i in selected_set]
                    best_indices_set = set(best_indices)
            return
        
        # Pruning: max possible sum from here
        max_sum = current_sum + suffix_sum[pos]
        if max_sum < 1:
            # Taking all remaining is still < 1
            deficit = 1 - max_sum
            if best_deficit_val is not None and deficit >= best_deficit_val:
                return
            # Take all remaining
            for j in range(pos, n):
                selected_set.add(j)
            # Record if this is better
            if deficit < best_deficit_val:
                best_deficit_val = deficit
                best_deficit = deficit
                best_indices = [order[i] for i in selected_set]
                best_indices_set = set(best_indices)
            for j in range(pos, n):
                selected_set.discard(j)
            return
        
        # Pruning: even taking all remaining exceeds 1, but maybe a subset works
        # If current_sum >= 1, prune (shouldn't happen due to our constraint)
        if current_sum >= 1:
            return
        
        # Option: include sorted_recs[pos]
        new_sum = current_sum + sorted_recs[pos]
        if new_sum < 1:
            selected_set.add(pos)
            solve(pos + 1, new_sum, selected_set)
            selected_set.discard(pos)
        
        # Option: exclude sorted_recs[pos]
        # Pruning for exclude: if current_sum + suffix_sum[pos+1] < 1 and deficit >= best, prune
        max_sum_excl = current_sum + suffix_sum[pos + 1]
        if max_sum_excl < 1:
            deficit_excl = 1 - max_sum_excl
            if best_deficit_val is not None and deficit_excl >= best_deficit_val:
                return
        solve(pos + 1, current_sum, selected_set)
    
    solve(0, Fraction(0), set())
    
    if best_indices is None:
        best_indices = [0, 1, 5, 41]
    
    return best_indices
# EVOLVE-BLOCK-END
