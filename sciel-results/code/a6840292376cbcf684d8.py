"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
import random

def construct():
    t0 = time.time()
    deadline = t0 + 80.0  # safety margin before 90s kill
    
    denominators = list(range(2, 49))
    n = len(denominators)
    vals = [Fraction(1, d) for d in denominators]
    
    # Greedy initialization: add fractions in order if sum stays < 1
    def greedy_init():
        selected = set()
        total = Fraction(0)
        for i in range(n):
            if total + vals[i] < 1:
                selected.add(i)
                total += vals[i]
        return selected, total
    
    best_set, best_total = greedy_init()
    best_deficit = 1 - best_total
    
    # Local search: try swaps and single additions/removals
    def local_search(selected, total, time_limit):
        nonlocal best_set, best_total, best_deficit
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Try single additions
            for i in range(n):
                if i not in selected:
                    new_total = total + vals[i]
                    if new_total < 1:
                        new_deficit = 1 - new_total
                        if new_deficit < best_deficit:
                            best_deficit = new_deficit
                            best_total = new_total
                            best_set = selected | {i}
                            selected = best_set
                            total = best_total
                            improved = True
                            if time.time() > time_limit:
                                return
            # Try single removals (to free up space for better additions)
            # Only useful if we can then add something better
            # Try swaps: remove j, add i
            for j in list(selected):
                if time.time() > time_limit:
                    return
                base_total = total - vals[j]
                for i in range(n):
                    if i not in selected and i != j:
                        new_total = base_total + vals[i]
                        if new_total < 1:
                            new_deficit = 1 - new_total
                            if new_deficit < best_deficit:
                                best_deficit = new_deficit
                                best_total = new_total
                                best_set = (selected - {j}) | {i}
                                selected = best_set
                                total = best_total
                                improved = True
                                break  # restart inner loop with new state
                    if time.time() > time_limit:
                        return
            # Try 2-for-1 swaps: remove j1, j2, add i1, i2
            if time.time() < time_limit:
                sel_list = list(selected)
                for a_idx in range(len(sel_list)):
                    for b_idx in range(a_idx + 1, len(sel_list)):
                        j1, j2 = sel_list[a_idx], sel_list[b_idx]
                        base_total = total - vals[j1] - vals[j2]
                        found = False
                        for i1 in range(n):
                            if i1 in selected:
                                continue
                            t1 = base_total + vals[i1]
                            if t1 >= 1:
                                continue
                            for i2 in range(n):
                                if i2 in selected or i2 == i1:
                                    continue
                                new_total = t1 + vals[i2]
                                if new_total < 1:
                                    new_deficit = 1 - new_total
                                    if new_deficit < best_deficit:
                                        best_deficit = new_deficit
                                        best_total = new_total
                                        best_set = (selected - {j1, j2}) | {i1, i2}
                                        selected = best_set
                                        total = best_total
                                        improved = True
                                        found = True
                                        break
                            if found or time.time() > time_limit:
                                break
                        if found or time.time() > time_limit:
                            break
                    if improved or time.time() > time_limit:
                        break
        return selected, total
    
    # Run local search
    time_limit = deadline - 5.0
    selected, total = local_search(best_set, best_total, time_limit)
    
    # Multiple restarts with different initializations
    if time.time() < deadline - 10.0:
        for restart in range(20):
            if time.time() > deadline - 5.0:
                break
            # Different greedy strategies
            strategy = restart % 4
            if strategy == 0:
                # Greedy by smallest denominator first
                order = list(range(n))
            elif strategy == 1:
                # Greedy by largest denominator first
                order = list(range(n - 1, -1, -1))
            elif strategy == 2:
                # Random order
                order = list(range(n))
                random.shuffle(order)
            else:
                # Greedy by value (same as smallest denom for unit fractions)
                order = list(range(n))
            
            selected_r = set()
            total_r = Fraction(0)
            for i in order:
                if total_r + vals[i] < 1:
                    selected_r.add(i)
                    total_r += vals[i]
            
            # Local search from this start
            time_limit_r = min(deadline - 5.0, time.time() + 3.0)
            selected_r, total_r = local_search(selected_r, total_r, time_limit_r)
            
            deficit_r = 1 - total_r
            if deficit_r < best_deficit:
                best_deficit = deficit_r
                best_total = total_r
                best_set = selected_r
    
    return sorted(best_set)
# EVOLVE-BLOCK-END
