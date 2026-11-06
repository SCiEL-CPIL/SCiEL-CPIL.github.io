import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    n = 128
    
    # Precompute all nonconstant 4-term APs
    # ap_list[i] = (a, a+d, a+2d, a+3d)
    ap_list = []
    for a in range(n):
        for d in range(1, (n - a - 3) // 3 + 1):
            if a + 3 * d < n:
                ap_list.append((a, a + d, a + 2 * d, a + 3 * d))
                
    num_aps = len(ap_list)
    
    # For fast updates, map each position to the list of AP indices it participates in
    # pos_to_aps[i] = list of indices in ap_list where i is a term
    pos_to_aps = [[] for _ in range(n)]
    for idx, ap in enumerate(ap_list):
        for pos in ap:
            pos_to_aps[pos].append(idx)
    
    # Precompute for each AP, the other 3 positions for each of the 4 slots
    # ap_others[idx][slot] = tuple of the other 3 positions
    # This allows fast calculation of distinct colors among the other 3
    ap_others = [None] * num_aps
    for idx, ap in enumerate(ap_list):
        ap_others[idx] = [
            tuple(x for j, x in enumerate(ap) if j != 0),
            tuple(x for j, x in enumerate(ap) if j != 1),
            tuple(x for j, x in enumerate(ap) if j != 2),
            tuple(x for j, x in enumerate(ap) if j != 3)
        ]

    def count_violations(colors):
        """Count the number of 4-term APs that violate the condition."""
        violations = 0
        for idx, ap in enumerate(ap_list):
            c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
            if len({c0, c1, c2, c3}) < 3:
                violations += 1
        return violations

    def get_color_count(colors):
        """Count distinct colors used."""
        return len(set(colors))

    # Initialize with a random assignment of k colors
    def initialize(k):
        return [random.randint(0, k - 1) for _ in range(n)]

    # Simulated Annealing
    def simulated_annealing(k, max_iter, start_temp, end_temp, cooling_rate):
        colors = initialize(k)
        current_violations = count_violations(colors)
        current_colors_used = get_color_count(colors)
        
        best_colors = colors.copy()
        best_violations = current_violations
        best_colors_used = current_colors_used
        
        # We want to minimize colors_used, but violations must be 0.
        # Objective: minimize (violations * large_penalty + colors_used)
        # But since we start with random, violations are likely high.
        # Let's prioritize reducing violations first, then colors.
        # Actually, a better objective for SA is to minimize a score:
        # score = violations * 100 + colors_used
        # If we can get violations to 0, we minimize colors_used.
        
        def score(v, c):
            return v * 1000 + c

        current_score = score(current_violations, current_colors_used)
        best_score = current_score
        
        for i in range(max_iter):
            if time.time() > deadline:
                break
            
            # Choose a random position to change
            pos = random.randint(0, n - 1)
            old_color = colors[pos]
            
            # Choose a new color randomly from 0..k-1
            # To encourage color reduction, we might bias towards existing colors or lower indices?
            # For now, uniform random is standard.
            new_color = random.randint(0, k - 1)
            
            if new_color == old_color:
                continue
            
            # Calculate delta in violations
            # Only APs involving 'pos' can change status
            delta_violations = 0
            affected_aps = pos_to_aps[pos]
            
            for idx in affected_aps:
                ap = ap_list[idx]
                # Determine slot of pos in this AP
                # We can just recalculate the violation status for this AP
                # Before change:
                old_vals = {colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]}
                old_bad = 1 if len(old_vals) < 3 else 0
                
                # Apply change temporarily for calculation? No, just compute new state
                # We know colors[pos] changes to new_color
                new_vals = set()
                for j, p in enumerate(ap):
                    if p == pos:
                        new_vals.add(new_color)
                    else:
                        new_vals.add(colors[p])
                new_bad = 1 if len(new_vals) < 3 else 0
                
                delta_violations += new_bad - old_bad
            
            # Apply change
            colors[pos] = new_color
            new_violations = current_violations + delta_violations
            new_colors_used = get_color_count(colors) # This is O(n), might be slow. 
            # Optimization: maintain a count of colors? 
            # For now, n=128 is small, O(n) is fast enough.
            
            new_score = score(new_violations, new_colors_used)
            delta_score = new_score - current_score
            
            temp = start_temp * (cooling_rate ** i)
            
            if delta_score < 0 or random.random() < np.exp(-delta_score / temp):
                current_violations = new_violations
                current_colors_used = new_colors_used
                current_score = new_score
                
                if current_score < best_score:
                    best_score = current_score
                    best_violations = current_violations
                    best_colors_used = current_colors_used
                    best_colors = colors.copy()
            else:
                # Revert
                colors[pos] = old_color
        
        # Return best found
        if best_violations == 0:
            return best_colors, best_colors_used
        return None, None

    # Try different numbers of colors, starting from low
    # The theoretical minimum is likely 3 or 4.
    # Let's try 3, then 4, then 5, etc.
    
    best_result = None
    best_k = 128
    
    # Strategy: 
    # 1. Try k=3 with SA. If it finds a valid solution, return it.
    # 2. Try k=4 with SA.
    # 3. If no valid solution found for small k, fall back to a greedy approach with a larger palette 
    #    or just return a valid coloring with more colors if time permits.
    
    # To ensure we always return a valid coloring, we first need a valid one.
    # The seed program's greedy approach is reliable for finding *a* valid coloring with many colors.
    # Let's first run a fast greedy to get a baseline valid coloring.
    
    # Fast Greedy (similar to seed but simpler, just to get a valid state)
    def fast_greedy(k):
        colors = [None] * n
        for i in range(n):
            valid = []
            for c in range(k):
                ok = True
                for idx in pos_to_aps[i]:
                    ap = ap_list[idx]
                    # Check if assigning c to i violates the AP
                    # The other 3 positions are already assigned
                    others = [colors[p] for p in ap if p != i]
                    if len({c} | set(others)) < 3:
                        ok = False
                        break
                if ok:
                    valid.append(c)
            if not valid:
                return None
            colors[i] = random.choice(valid)
        return colors

    # Get a baseline valid coloring
    baseline = None
    for k in range(3, 129):
        if time.time() > deadline - 10:
            break
        baseline = fast_greedy(k)
        if baseline is not None:
            best_k = k
            best_result = baseline
            break
    
    if best_result is None:
        # Fallback: all distinct
        return list(range(n))
        
    # Now, try to improve using SA with smaller k
    # Start from k-1, k-2, etc., down to 3
    # Or start from the current best_k and try to reduce?
    # SA starts from random, so it doesn't inherit the structure of the greedy solution.
    # However, we can try SA with k = best_k - 1, best_k - 2, ...
    
    # If best_k is already small (e.g., 3), we are done.
    if best_k == 3:
        return best_result
        
    # Try SA for k from best_k - 1 down to 3
    # Allocate time for each k
    time_per_k = 10  # seconds
    
    for k in range(min(best_k - 1, 10), 2, -1): # Try up to 10 colors max for SA to be efficient
        if time.time() > deadline - 5:
            break
            
        # Run SA for this k
        # Number of iterations depends on time
        # Let's run for a fixed number of iterations or until time runs out
        max_iter = 50000
        start_temp = 10.0
        end_temp = 0.01
        cooling_rate = (end_temp / start_temp) ** (1.0 / max_iter)
        
        res, k_used = simulated_annealing(k, max_iter, start_temp, end_temp, cooling_rate)
        
        if res is not None:
            # Found a valid coloring with k colors
            # Check if it uses fewer colors than current best
            if k_used < best_k:
                best_k = k_used
                best_result = res
                # If we reached 3, we can stop
                if best_k == 3:
                    break
            # Even if not fewer, it's a valid coloring with k colors.
            # We want to minimize colors, so we only update if strictly better.
            # But wait, if SA finds a valid coloring with k colors, it means k is sufficient.
            # We are iterating downwards, so the first valid one we find is the minimum for that search.
            # Actually, if we find a valid one with k, we should record it and continue to try k-1.
            
    # If we found a better result, return it
    # Otherwise return the baseline
    return best_result
