"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import heapq
from fractions import Fraction

def construct():
    denominators = list(range(2, 49))
    n = len(denominators)
    
    start_time = time.time()
    deadline = start_time + 80.0  # 10s margin from 90s limit
    
    # Precompute unit fractions
    unit_fracs = [Fraction(1, d) for d in denominators]
    
    # Best solution found so far
    best_deficit = None
    best_indices = None
    
    # Phase 1: Greedy initial solution (take all possible, then remove smallest that keeps sum < 1)
    # Actually, greedy: iterate and add if sum stays < 1
    selected, total = [], Fraction(0)
    for i, d in enumerate(denominators):
        if total + unit_fracs[i] < 1:
            selected.append(i)
            total += unit_fracs[i]
    
    best_deficit = 1 - total
    best_indices = selected[:]
    
    # Phase 2: Dijkstra-like search on deficit space
    # State: (deficit, frozenset_of_selected)
    # Start: deficit = 0, empty set
    # Goal: minimal positive deficit where sum < 1
    
    # Use a priority queue ordered by deficit
    # To keep state space manageable, we limit depth and use beam search
    
    # Actually, let's do a more targeted approach:
    # Local search with multiple restarts and better swap heuristics
    
    def local_search(initial_selected, max_time):
        """Improve a starting solution via swap and insert/delete operations."""
        selected_set = set(initial_selected)
        total = sum(unit_fracs[i] for i in selected_set)
        deficit = 1 - total
        
        if deficit <= 0:
            return list(initial_selected), deficit
        
        improved = True
        while improved and (time.time() - start_time) < max_time:
            improved = False
            
            # Try all single swaps: remove one, add one
            unselected = [i for i in range(n) if i not in selected_set]
            
            # Sort candidates by potential improvement
            # For each pair (i_out, i_in), check if it improves
            best_swap = None
            best_new_deficit = deficit
            
            for i_out in list(selected_set):
                out_frac = unit_fracs[i_out]
                for i_in in unselected:
                    in_frac = unit_fracs[i_in]
                    new_total = total - out_frac + in_frac
                    if new_total < 1:
                        new_deficit = 1 - new_total
                        if new_deficit < best_new_deficit:
                            best_new_deficit = new_deficit
                            best_swap = (i_out, i_in)
            
            if best_swap is not None:
                i_out, i_in = best_swap
                selected_set.remove(i_out)
                selected_set.add(i_in)
                total = total - unit_fracs[i_out] + unit_fracs[i_in]
                deficit = 1 - total
                improved = True
                continue
            
            # Try single inserts (if sum < 1 after adding)
            best_insert = None
            best_insert_deficit = deficit
            for i_in in unselected:
                new_total = total + unit_fracs[i_in]
                if new_total < 1:
                    new_deficit = 1 - new_total
                    if new_deficit < best_insert_deficit:
                        best_insert_deficit = new_deficit
                        best_insert = i_in
            
            if best_insert is not None:
                selected_set.add(best_insert)
                total = total + unit_fracs[best_insert]
                deficit = 1 - total
                improved = True
                continue
            
            # Try single deletes (if it reduces deficit... wait, deleting increases deficit)
            # Actually deleting makes deficit larger, so we don't want that unless it enables better swaps
            # But sometimes we need to delete to make room for better inserts
            # This is handled by the swap above
            
            # Try 2-swap: remove two, add two
            if len(selected_set) >= 2:
                best_2swap = None
                best_2swap_deficit = deficit
                
                sel_list = list(selected_set)
                # Limit combinations for performance
                for idx1 in range(len(sel_list)):
                    for idx2 in range(idx1 + 1, len(sel_list)):
                        i_out1 = sel_list[idx1]
                        i_out2 = sel_list[idx2]
                        base_total = total - unit_fracs[i_out1] - unit_fracs[i_out2]
                        
                        for i_in in unselected:
                            t1 = base_total + unit_fracs[i_in]
                            if t1 >= 1:
                                continue
                            for j_in in unselected:
                                if j_in == i_in:
                                    continue
                                new_total = t1 + unit_fracs[j_in]
                                if new_total < 1:
                                    new_deficit = 1 - new_total
                                    if new_deficit < best_2swap_deficit:
                                        best_2swap_deficit = new_deficit
                                        best_2swap = (i_out1, i_out2, i_in, j_in)
            
            if best_2swap is not None:
                i_out1, i_out2, i_in1, i_in2 = best_2swap
                selected_set.remove(i_out1)
                selected_set.remove(i_out2)
                selected_set.add(i_in1)
                selected_set.add(i_in2)
                total = sum(unit_fracs[i] for i in selected_set)
                deficit = 1 - total
                improved = True
                continue
        
        return list(selected_set), deficit
    
    # Run local search from the greedy solution
    result, deficit = local_search(best_indices, deadline)
    if deficit < best_deficit:
        best_deficit = deficit
        best_indices = result
    
    # Phase 3: Randomized restarts
    random.seed(42)
    remaining_time = deadline - (time.time() - start_time)
    restarts_done = 0
    
    while (time.time() - start_time) < deadline - 5.0:
        # Generate a random starting solution
        # Strategy: random subset, then repair
        num_selected = random.randint(10, 30)
        indices = random.sample(range(n), num_selected)
        
        # Repair: ensure sum < 1
        # Remove largest elements until sum < 1
        indices.sort(key=lambda i: -unit_fracs[i])  # descending by fraction value
        total = sum(unit_fracs[i] for i in indices)
        for i in indices[:]:
            if total >= 1:
                total -= unit_fracs[i]
                indices.remove(i)
        
        if not indices:
            continue
        
        # Run local search
        result, deficit = local_search(indices, deadline)
        if deficit < best_deficit:
            best_deficit = deficit
            best_indices = result
        
        restarts_done += 1
    
    return best_indices
# EVOLVE-BLOCK-END
