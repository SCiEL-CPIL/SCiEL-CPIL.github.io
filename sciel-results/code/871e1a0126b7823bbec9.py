"""
Optimized solver for the 4-term progression coloring problem.
Uses fast local search (Simulated Annealing) with NumPy vectorization to minimize the number of colors used.
"""
import time
import random
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit
    
    # 1. Precompute all 4-term arithmetic progressions
    progressions = []
    for i in range(128):
        for d in range(1, (128 - i) // 3 + 1):
            j = i + d
            k = i + 2 * d
            l = i + 3 * d
            if l < 128:
                progressions.append((i, j, k, l))
    
    n_progressions = len(progressions)
    
    # Convert to NumPy arrays for fast access
    # Shape: (n_progressions, 4)
    prog_arr = np.array(progressions, dtype=np.int32)
    
    # Map position to list of progression indices
    # We need this for local updates, but for the initial fast check we can just use the array.
    # For the SA loop, we will stick to a hybrid approach: 
    # 1. Fast global check using NumPy to see if valid.
    # 2. If not valid, we need to pick a position to change. 
    #    Picking a position involved in violations is better.
    #    Let's precompute pos_to_progs for efficient delta calculation if we go back to Python loops,
    #    OR we can use a simpler heuristic: pick a random position, change color, check global validity (fast in NumPy).
    #    Given 128 positions, checking global validity is ~1000 ops. Very fast.
    
    def check_validity(colors_np):
        """
        Returns number of violations.
        colors_np: array of shape (128,) with integer colors.
        """
        # Get colors for each progression
        # colors_np[prog_arr] has shape (n_progressions, 4)
        prog_colors = colors_np[prog_arr]
        
        # Count distinct colors in each row
        # A row is valid if it has at least 3 distinct colors.
        # It is invalid if it has 1 or 2 distinct colors.
        # 1 distinct: all same.
        # 2 distinct: not all same, but only 2 unique values.
        
        # Let's compute unique counts per row.
        # np.unique is slow on 2D arrays row-by-row.
        # Alternative: 
        # A 4-tuple has < 3 distinct colors iff:
        # (c0==c1==c2==c3) OR 
        # (c0==c1==c2 != c3) OR (c0==c1!=c2==c3) ... etc.
        # Basically, it's invalid if the number of distinct elements is 1 or 2.
        
        # We can check:
        # invalid if:
        # (c0==c1 and c1==c2 and c2==c3) # 1 color
        # OR (c0==c1 and c1==c2 and c2!=c3) # 2 colors (3 same, 1 diff)
        # OR (c0==c1 and c1!=c2 and c2==c3) # 2 colors (2 same, 1 diff, 1 same)
        # OR (c0==c1 and c1!=c2 and c2!=c3 and c3!=c0) # This is 4 distinct? No, c0=c1, c2, c3 distinct from c0 and each other? 
        # Let's just use a vectorized distinct count.
        
        # Sort each row
        sorted_colors = np.sort(prog_colors, axis=1)
        
        # Count distincts by looking at differences
        # distinct_count = 1 + sum(sorted_colors[:, 1:] != sorted_colors[:, :-1])
        distinct_counts = 1 + np.sum(sorted_colors[:, 1:] != sorted_colors[:, :-1], axis=1)
        
        # Violation if distinct_counts < 3
        violations = np.sum(distinct_counts < 3)
        return violations

    def search_k_colors(k, time_budget, initial_colors=None):
        """
        Try to find a valid coloring using exactly k colors (0..k-1).
        Returns a numpy array of colors if found, else None.
        """
        start_search = time.time()
        max_restarts = 20
        T_initial = 1.0
        cooling_rate = 0.99 # Slower cooling for more exploration
        
        # If no initial colors, start random
        if initial_colors is None:
            current_colors = np.random.randint(0, k, 128)
        else:
            current_colors = initial_colors.copy()
            
        # Ensure we only use colors 0..k-1
        # If initial colors had more, we might need to map them? 
        # The task says "labels need not be consecutive", but "Minimize the number of labels actually used".
        # If we search for k colors, we should restrict to 0..k-1.
        # If initial_colors has values >= k, we should probably map them or restart.
        # Let's assume initial_colors is already in 0..k-1 or we handle it.
        if np.any(current_colors >= k):
             # Map existing colors to 0..k-1? Or just restart?
             # Better to restart with random if range mismatch is large.
             current_colors = np.random.randint(0, k, 128)

        current_violations = check_validity(current_colors)
        
        if current_violations == 0:
            return current_colors

        T = T_initial
        
        for restart in range(max_restarts):
            if time.time() > start_search + time_budget:
                break
                
            if restart > 0 and current_violations > 0:
                # Restart with randomization
                current_colors = np.random.randint(0, k, 128)
                current_violations = check_validity(current_colors)
                if current_violations == 0:
                    return current_colors
                T = T_initial
            
            steps_in_restart = 0
            max_steps = 50000 # More steps possible with NumPy
            
            while current_violations > 0 and steps_in_restart < max_steps:
                if time.time() > start_search + time_budget:
                    break
                
                # Pick a random position
                pos = np.random.randint(0, 128)
                old_color = current_colors[pos]
                
                # Pick a new color
                new_color = np.random.randint(0, k)
                if new_color == old_color:
                    continue
                
                # Try the move
                current_colors[pos] = new_color
                new_violations = check_validity(current_colors)
                
                delta = new_violations - current_violations
                
                if delta <= 0 or (T > 1e-9 and random.random() < math.exp(-delta / T)):
                    # Accept
                    current_violations = new_violations
                    # If we accepted a bad move, we keep the new colors
                    # If we rejected, we revert
                else:
                    # Reject, revert
                    current_colors[pos] = old_color
                
                steps_in_restart += 1
                T *= cooling_rate
                
                if current_violations == 0:
                    return current_colors
                    
        return None

    # Strategy:
    # 1. Try k=10 with a short budget. It should be easy to find a valid coloring.
    # 2. Use that result to initialize searches for k=9, 8, ..., 3.
    
    best_colors = None
    best_count = 128
    
    # Time allocation:
    # k=10: 5s (easy)
    # k=9: 5s
    # k=8: 5s
    # k=7: 5s
    # k=6: 5s
    # k=5: 5s
    # k=4: 5s
    # k=3: 5s
    # Total 40s.
    
    current_colors = None
    
    for k in range(10, 2, -1):
        if time.time() > deadline - 5.0:
            break
            
        budget = 5.0
        remaining = deadline - time.time() - 5.0
        if budget > remaining:
            budget = remaining
            
        if budget < 1.0:
            break
            
        result = search_k_colors(k, budget, initial_colors=current_colors)
        
        if result is not None:
            # Check if it's actually valid and uses <= k colors
            # result contains values 0..k-1
            if check_validity(result) == 0:
                used_colors = sorted(list(set(result)))
                m = len(used_colors)
                
                if m < best_count:
                    # Map to 0..m-1 for compactness? 
                    # The task says "labels need not be consecutive". 
                    # But "Minimize the number of labels actually used".
                    # If I return labels 0, 2, 5, the count is 3.
                    # If I return 0, 1, 2, the count is 3.
                    # The evaluator counts distinct labels. So mapping doesn't change the score, 
                    # but keeps numbers small.
                    mapping = {c: i for i, c in enumerate(used_colors)}
                    optimized_result = np.array([mapping[c] for c in result])
                    
                    best_count = m
                    best_colors = optimized_result
                    current_colors = result # Keep the raw result for next iteration (same color space)
                    
                    if m == 3:
                        # Can't do better than 3 for non-constant 4-term AP?
                        # Actually, can we do 2? No, 2 colors always allow a monochromatic AP of length 4? 
                        # Van der Waerden number W(2,4) is 16. So 2 colors fail for N>=16.
                        # So 3 is the theoretical minimum if we assume we can avoid 2.
                        # If we find 3, we can return immediately.
                        return optimized_result

    if best_colors is None:
        # Fallback: Just use 10 colors randomly if we didn't find anything (unlikely)
        # Or use a greedy construction?
        # Let's just return a random 10-coloring if we failed, though it should be valid.
        res = search_k_colors(10, 5.0)
        if res is not None and check_validity(res) == 0:
             best_colors = res
        else:
             # Last resort: 128 distinct colors (always valid)
             best_colors = np.arange(128)

    return best_colors
