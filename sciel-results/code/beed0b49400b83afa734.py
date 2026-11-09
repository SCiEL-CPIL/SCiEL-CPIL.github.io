"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from math import gcd

def construct():
    N = 256
    # 1. Greedy Initialization: Pick largest numbers that are coprime to selected ones
    selected = []
    for k in range(N, 0, -1):
        if all(gcd(k, s) == 1 for s in selected):
            selected.append(k)
    
    # 2. Local Search to improve the solution
    # We try to swap elements to increase the sum.
    # To ensure we finish within time limits, we limit the search depth/iterations.
    
    def is_valid_candidate(candidate, current_selected, exclude_idx=None):
        # Check if candidate is coprime to all in current_selected except the one being removed
        for i, s in enumerate(current_selected):
            if i == exclude_idx:
                continue
            if gcd(candidate, s) != 1:
                return False
        return True

    improved = True
    iterations = 0
    max_iterations = 1000 # Safety cap
    
    while improved and iterations < max_iterations:
        improved = False
        iterations += 1
        
        # Try single swap: remove one, add one
        best_gain = 0
        best_swap = None
        for i, s_remove in enumerate(selected):
            # Find best candidate to add
            for s_add in range(N, 0, -1):
                if s_add in selected:
                    continue
                if is_valid_candidate(s_add, selected, i):
                    gain = s_add - s_remove
                    if gain > best_gain:
                        best_gain = gain
                        best_swap = (i, s_add)
        
        if best_swap:
            i, s_add = best_swap
            selected[i] = s_add
            improved = True
            continue
            
        # Try double swap: remove one, add two
        best_gain = 0
        best_swap2 = None
        for i, s_remove in enumerate(selected):
            # We need to find two numbers a, b not in selected
            # such that a,b coprime to selected (except i) and a,b coprime to each other
            # And a+b > s_remove
            candidates = []
            for s_add in range(N, 0, -1):
                if s_add in selected:
                    continue
                if is_valid_candidate(s_add, selected, i):
                    candidates.append(s_add)
            
            # Sort candidates descending to check larger sums first
            candidates.sort(reverse=True)
            # Check pairs in candidates
            # Optimization: break early if max possible sum < s_remove
            # Max sum is candidates[0] + candidates[1]
            if len(candidates) >= 2:
                for j in range(len(candidates)):
                    for k in range(j+1, len(candidates)):
                        a = candidates[j]
                        b = candidates[k]
                        if gcd(a, b) != 1:
                            continue
                        gain = a + b - s_remove
                        if gain > best_gain:
                            best_gain = gain
                            best_swap2 = (i, a, b)
                        # Optional: break if a+b - s_remove < best_gain and a is small
                        # But we just want ANY improvement, so we can break inner loop if a is too small?
                        # No, just let it run, N is small.
        
        if best_swap2:
            i, a, b = best_swap2
            selected[i] = a
            selected.append(b)
            improved = True
            continue

    return selected
# EVOLVE-BLOCK-END
