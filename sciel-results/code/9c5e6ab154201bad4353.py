import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    n = 128
    # Precompute all 4-APs
    aps = []
    for d in range(1, n // 3 + 1):
        for a in range(1, n - 3 * d + 1):
            # 0-indexed
            aps.append((a - 1, a - 1 + d, a - 1 + 2 * d, a - 1 + 3 * d))
            
    def get_violations(colors):
        count = 0
        for ap in aps:
            # Check distinct colors
            # Unpack for speed?
            c0 = colors[ap[0]]
            c1 = colors[ap[1]]
            c2 = colors[ap[2]]
            c3 = colors[ap[3]]
            # Distinct count
            # If all same: 1
            # If 2 same, 2 diff: 2
            # If 3 same, 1 diff: 2
            # If 2 pairs: 2
            # If 1 pair, 2 diff: 3
            # If all diff: 4
            # We need < 3
            if c0 == c1 == c2 == c3:
                count += 1
            else:
                # Check if only 2 distinct
                # Use set
                if len({c0, c1, c2, c3}) < 3:
                    count += 1
        return count

    def get_distinct_count(colors):
        return len(set(colors))

    # Step 1: Greedy construction to get a valid baseline
    # This ensures we always have a valid solution to return
    greedy_colors = []
    max_color = 0
    for x in range(1, n + 1):
        found = False
        for c in range(max_color + 1):
            # Check all APs ending at x-1 (0-indexed x-1)
            valid = True
            # An AP (a, a+d, a+2d, a+3d) ends at x-1 if x-1 = a+3d
            # So a = x-1-3d
            # We need a >= 1 (0-indexed a >= 0)
            # x-1-3d >= 0 => 3d <= x-1 => d <= (x-1)//3
            for d in range(1, (x - 1) // 3 + 1):
                a = x - 1 - 3 * d
                # Indices: a, a+d, a+2d, a+3d
                # a+3d is x-1
                # Check colors at a, a+d, a+2d
                c_a = greedy_colors[a]
                c_ad = greedy_colors[a + d]
                c_a2d = greedy_colors[a + 2 * d]
                # New color c
                # Set of colors in AP: {c_a, c_ad, c_a2d, c}
                # Must have size >= 3
                if len({c_a, c_ad, c_a2d, c}) < 3:
                    valid = False
                    break
            if valid:
                greedy_colors.append(c)
                found = True
                break
        if not found:
            # Use new color
            c = max_color + 1
            max_color = c
            greedy_colors.append(c)
            
    best_colors = greedy_colors
    best_violations = 0 # Greedy is valid
    best_distinct = get_distinct_count(best_colors)
    
    # Step 2: Try to reduce colors using local search
    # We will try K = 4, 5, ... up to best_distinct
    # For each K, we run local search restricted to colors 0..K-1
    
    # If greedy used few colors, we might not need to try lower K
    # But greedy might use more than necessary.
    
    # Optimization: If greedy used <= 4, return it.
    if best_distinct <= 4:
        return best_colors

    # Try K from 4 to best_distinct - 1
    # We give each K a time budget
    total_time_budget = deadline - time.time()
    if total_time_budget < 2:
        return best_colors
        
    # We'll try 4, 5, 6...
    # If we find a valid coloring with K colors, we return it.
    # We should try smallest K first.
    
    # To speed up, we can initialize the local search state with the greedy solution
    # but mapped to 0..K-1? No, greedy uses specific colors.
    # Better: Random initialization with K colors.
    
    # Let's try K=4 first with a decent time budget.
    # If we fail, try K=5, etc.
    
    # We can also try to merge colors in the greedy solution.
    # But local search with fixed domain is cleaner.
    
    # Strategy:
    # For K in [4, 5, ... best_distinct-1]:
    #   Run local search for time_slice
    #   If valid found, return
    
    # How to split time?
    # Give 20% of remaining time to each K?
    # Or fixed 10s per K?
    # 10s is plenty for 128 vars.
    
    # Let's try K=4 for 15s.
    # If not found, K=5 for 15s.
    # If not found, return greedy.
    
    # Actually, we can run local search on the GREEDY solution itself,
    # allowing color changes to any existing color, and see if we can
    # reduce the distinct count while keeping violations 0.
    # This is safer.
    
    # Let's do that.
    # Current state: greedy_colors (valid)
    # Objective: Minimize distinct colors.
    # Penalty for violations: Huge.
    
    current_colors = greedy_colors[:]
    current_violations = 0
    current_distinct = best_distinct
    current_score = current_violations * 10000 + current_distinct
    
    best_score = current_score
    best_final_colors = current_colors[:]
    
    # Simulated Annealing
    T = 1.0
    # We want to reduce distinct colors.
    # Moves: Change color of a random position to a random existing color.
    # If we introduce a new color, distinct count increases (bad).
    # So we only pick from existing colors.
    
    existing_colors = set(current_colors)
    
    steps = 0
    while time.time() < deadline:
        steps += 1
        i = random.randint(0, n - 1)
        old_c = current_colors[i]
        
        # Pick a new color from existing colors
        # To encourage merging, pick a color that is NOT old_c
        # If only 1 color exists, we can't merge.
        if len(existing_colors) == 1:
            break
            
        new_c = random.choice(list(existing_colors - {old_c}))
        
        current_colors[i] = new_c
        
        # Calculate new violations
        new_violations = get_violations(current_colors)
        new_distinct = get_distinct_count(current_colors)
        
        # Update existing colors set if distinct count changed?
        # get_distinct_count is O(N).
        # We can maintain a count of each color.
        # But for simplicity, O(N) is fine for N=128.
        
        new_score = new_violations * 10000 + new_distinct
        
        if new_score < current_score or random.random() < pow(2.718, (current_score - new_score) / T):
            current_score = new_score
            current_violations = new_violations
            current_distinct = new_distinct
            
            if new_score < best_score:
                best_score = new_score
                best_final_colors = current_colors[:]
                # Update existing colors if we found a better solution
                # Actually, we should restrict to colors in best_final_colors?
                # No, we can use any color that appeared.
                # But if we drop a color, it's gone.
                # We should probably recompute existing_colors from best_final_colors
                # to avoid using colors that are no longer "active" in the best solution?
                # But if we use a color that was dropped, we increase distinct count.
                # So it's fine.
                
        else:
            current_colors[i] = old_c
            
        # Cool down
        T *= 0.9995
        
        # If we reached 0 violations and distinct count is 4, we can stop?
        # Or keep going to see if 3 is possible?
        # 3 is likely impossible.
        # If distinct count is 4 and violations 0, it's a great result.
        # We can try to see if 3 is possible by forcing domain to 3 colors?
        # But local search with existing colors might get stuck at 4.
        
        # If we have a valid 4-coloring, we can try to improve to 3?
        # Unlikely.
        
        # If time is running out, break.
        
    return best_final_colors

