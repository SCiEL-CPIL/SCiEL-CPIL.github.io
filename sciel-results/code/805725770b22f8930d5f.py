"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from math import gcd

def construct():
    N = 256
    
    # 1. Greedy Initialization: Largest to smallest
    selected = []
    for k in range(N, 0, -1):
        if all(gcd(k, s) == 1 for s in selected):
            selected.append(k)
    
    # Precompute coprime check for speed in local search
    # We will use a set for O(1) lookups for selected numbers
    selected_set = set(selected)
    unselected = set(range(1, N + 1)) - selected_set
    
    # Helper to check if a candidate is coprime with all in a set
    def is_coprime_with_set(candidate, others):
        for o in others:
            if gcd(candidate, o) != 1:
                return False
        return True
        
    improved = True
    while improved:
        improved = False
        
        # 1-move local search: Replace x with y (y > x)
        # Iterate through selected numbers
        for i, x in enumerate(selected):
            # Find potential replacements y > x
            # We can optimize by checking unselected numbers > x
            best_y = -1
            # Check unselected numbers greater than x
            # To be efficient, we can sort unselected or just iterate
            # Since N is small, iterating unselected is fine
            for y in unselected:
                if y <= x:
                    continue
                # Check if y is coprime with all other selected numbers (excluding x)
                # If yes, y is a valid replacement
                # We don't need to check y vs x because y != x and we are replacing x
                valid = True
                for j, s in enumerate(selected):
                    if i == j:
                        continue
                    if gcd(y, s) != 1:
                        valid = False
                        break
                if valid:
                    best_y = y
                    break # Found the smallest valid y > x? 
                    # Actually, we want the LARGEST valid y to maximize sum immediately.
                    # But finding the largest might take longer. 
                    # Let's just find ANY valid y > x first, then we can iterate again.
                    # Or, better: iterate unselected in descending order to find the best replacement.
            
            # Let's refine: Find the BEST replacement (largest y) for x
            best_y = -1
            # Iterate unselected in descending order
            # Creating a sorted list each time is expensive. 
            # Let's just iterate all unselected and keep the max valid one.
            for y in unselected:
                if y <= x:
                    continue
                valid = True
                for j, s in enumerate(selected):
                    if i == j:
                        continue
                    if gcd(y, s) != 1:
                        valid = False
                        break
                if valid and y > best_y:
                    best_y = y
            
            if best_y > 0:
                # Perform swap
                selected[i] = best_y
                selected_set.remove(x)
                selected_set.add(best_y)
                unselected.remove(best_y)
                unselected.add(x)
                improved = True
                break # Restart loop after a change
        
        # 2-move local search: Replace {x, z} with {y, w}
        # This is O(N^4) worst case, but N is small and selected is sparse.
        # Selected size is approx N / log(N) ~ 256 / 5 ~ 50.
        # 50^2 = 2500 pairs.
        # Unselected size ~ 200.
        # 200^2 = 40000 pairs.
        # 2500 * 40000 = 100M checks. Might be too slow in Python if done naively.
        # Optimization: Only check pairs where sum(y,w) > sum(x,z).
        
        # To keep it simple and fast, let's stick to 1-moves first. 
        # If 1-moves don't improve, try 2-moves but with a limit or smarter pruning.
        
        # Let's implement a limited 2-move search
        # Sort selected to help with pruning?
        # Let's try a few random 2-moves or a systematic small window.
        
        # Given the constraints and time, 1-move is often sufficient for this specific problem structure (intervals).
        # However, 2-moves can unlock significant gains.
        
        # Efficient 2-move:
        # For each pair (x, z) in selected:
        #   Find best pair (y, w) in unselected such that y+w > x+z and coprime constraints met.
        
        # To avoid O(N^4), we can pre-filter.
        # Let's just do a few iterations of 1-move. If it converges, we are likely at a local optimum.
        
        # If we want to be thorough, we can add a simple 2-move.
        # Let's add a basic 2-move check for the first few iterations if 1-move fails.
        
        if not improved:
            # Try 2-moves
            # Get list of selected indices
            n_sel = len(selected)
            found_2move = False
            for i in range(n_sel):
                for j in range(i + 1, n_sel):
                    x = selected[i]
                    z = selected[j]
                    target_sum = x + z
                    
                    best_y = -1
                    best_w = -1
                    
                    # Iterate unselected
                    # Optimization: Only consider y, w > min(x,z) is not enough, need sum > target_sum
                    # Let's just iterate all pairs in unselected
                    # To speed up, we can iterate y, then find best w
                    # But checking coprimality is expensive.
                    
                    # Let's just do a limited search: try replacing with the largest available numbers
                    # This is a heuristic.
                    
                    # For robustness and speed, let's stick to 1-move for now as it's O(N^2 * K) roughly.
                    # If the score is already 1.21, 1-move might push it higher.
                    
                    pass # Skip 2-move for now to ensure speed and simplicity, unless 1-move stalls early.
                    # Actually, let's implement a simple 2-move but break early.
                    
            # If we found a 2-move, we would update and set improved=True
            # Since we skipped it, improved remains False and loop ends.
            
    return [int(x) for x in selected]
# EVOLVE-BLOCK-END
