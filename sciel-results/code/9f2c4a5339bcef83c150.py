import random
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill
    
    # List of all K4s: indices of edges in the 120-edge array
    # Edges are (i, j) with i < j. Index = i * 16 - i * (i + 1) / 2 + (j - i - 1)
    # Let's precompute edge index mapping
    def get_edge_idx(i, j):
        if i > j: i, j = j, i
        return i * 16 - i * (i + 1) // 2 + (j - i - 1)
    
    # Precompute all K4s and their constituent edge indices
    vertices = list(range(16))
    k4s = []
    for quad in itertools.combinations(vertices, 4):
        v0, v1, v2, v3 = quad
        e1 = get_edge_idx(v0, v1)
        e2 = get_edge_idx(v0, v2)
        e3 = get_edge_idx(v0, v3)
        e4 = get_edge_idx(v1, v2)
        e5 = get_edge_idx(v1, v3)
        e6 = get_edge_idx(v2, v3)
        k4s.append((e1, e2, e3, e4, e5, e6))
        
    num_k4s = len(k4s)
    
    # For fast update, map edge index to list of K4 indices containing it
    edge_to_k4s = [[] for _ in range(120)]
    for k4_idx, edges in enumerate(k4s):
        for e in edges:
            edge_to_k4s[e].append(k4_idx)

    best_solution = None
    best_color_count = 18 # Worse than seed (17)

    # Initial valid solution: The seed (17 colors)
    seed_solution = [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
    best_solution = seed_solution[:]
    best_color_count = len(set(seed_solution)) # 17
    
    # Helper to check validity and count colors
    def check_and_get_colors(colors):
        # colors is list of 120 ints
        # Check if any K4 has < 5 colors
        for edges in k4s:
            c = len(set(colors[e] for e in edges))
            if c < 5:
                return False, -1
        return True, len(set(colors))

    # Validate seed just in case
    is_valid, cnt = check_and_get_colors(best_solution)
    if is_valid:
        best_color_count = cnt
    else:
        # Should not happen, but if it does, we have a problem.
        # The seed is theoretically valid for 17.
        pass

    targets = [13, 14, 15]
    
    # Allocate time: 
    # 60s for 13, 30s for 14, 20s for 15, 10s buffer.
    # If we find a valid 13, we stop.
    # If not, we try 14. If we find a valid 14, we update best.
    # If we don't find 14, we try 15.
    
    # We'll run a local search for each target.
    # The local search tries to repair a random coloring using `target` colors.
    
    for target in targets:
        if time.time() > deadline - 5: # Keep 5s buffer for final check/return
            break
            
        # Time slice for this target
        # Allocate remaining time proportionally or fixed chunks
        # If 13 fails, we move to 14.
        slice_duration = 40 if target == 13 else (20 if target == 14 else 15)
        slice_end = min(deadline - 2, time.time() + slice_duration)
        
        found_valid_for_target = False
        
        while time.time() < slice_end:
            # Random initialization
            colors = [random.randint(0, target - 1) for _ in range(120)]
            
            # Calculate initial violations
            # We maintain:
            # 1. color_counts[k4_idx][color] -> count of edges in k4_idx with that color
            # 2. num_colors_in_k4[k4_idx] -> number of distinct colors in k4_idx
            # 3. violating_k4s -> list of indices of k4s with num_colors < 5
            
            color_counts = [None] * num_k4s
            num_colors_in_k4 = [0] * num_k4s
            violating_k4s = []
            
            for k4_idx, edges in enumerate(k4s):
                counts = [0] * target
                distinct = 0
                for e in edges:
                    c = colors[e]
                    if counts[c] == 0:
                        distinct += 1
                    counts[c] += 1
                color_counts[k4_idx] = counts
                num_colors_in_k4[k4_idx] = distinct
                if distinct < 5:
                    violating_k4s.append(k4_idx)
            
            num_violations = len(violating_k4s)
            
            if num_violations == 0:
                # Valid solution with target colors (or fewer, if some colors unused)
                actual_colors = len(set(colors))
                if actual_colors < best_color_count:
                    best_solution = colors[:]
                    best_color_count = actual_colors
                    # If we found 13, we can break out of everything
                    if best_color_count == 13:
                        return best_solution
                found_valid_for_target = True
                continue # Restart with new random init to see if we can get fewer distinct colors? 
                         # Actually, if valid, we can try to reduce distinct colors used?
                         # But the constraint is just "at least 5". 
                         # Minimizing "number of colours used" means minimizing the size of the image of the coloring function.
                         # If we use 13 slots but only 10 distinct colors, that's 10.
                         # However, usually random init with target colors uses all or most.
                         # If we find a valid one, we record it.
                         # We can try to polish to remove unused colors, but that's complex.
                         # Let's just restart to find potentially better (lower distinct count) solutions.
            
            # Local Search Loop: Greedy Hill Climbing
            local_iterations = 0
            max_local_iterations = 20000
            no_improve_counter = 0
            max_no_improve = 500
            
            while num_violations > 0 and local_iterations < max_local_iterations and time.time() < slice_end:
                local_iterations += 1
                
                if no_improve_counter >= max_no_improve:
                    break # Restart
                
                # Pick a random violating K4
                if not violating_k4s:
                    break
                k4_idx = random.choice(violating_k4s)
                edges = k4s[k4_idx]
                edge_idx = edges[random.randint(0, 5)]
                
                old_color = colors[edge_idx]
                
                # Try to find a new color that reduces violations
                # We will try a few random colors
                move_made = False
                for _ in range(10): # Try 10 candidates
                    new_color = random.randint(0, target - 1)
                    if new_color == old_color:
                        continue
                    
                    # Calculate impact on violations
                    # Affected K4s are those containing edge_idx
                    affected_k4s = edge_to_k4s[edge_idx]
                    
                    delta_violations = 0
                    valid_move = True
                    
                    # We need to check if the move reduces the total number of violations.
                    # A K4 is a violation if distinct_colors < 5.
                    # Change in distinct_colors for a K4:
                    # Let C be the set of colors in the K4.
                    # Old: c_old is in C.
                    # New: c_new is in C.
                    # If c_old is removed from C (count becomes 0) and c_new is added to C (count was 0), distinct stays same.
                    # If c_old is removed and c_new was already in C, distinct decreases by 1.
                    # If c_old is NOT removed (count > 1) and c_new is added (count was 0), distinct increases by 1.
                    # If c_old is NOT removed and c_new was already in C, distinct stays same.
                    
                    # We want to minimize violations.
                    # So we prefer moves that increase distinct counts for violating K4s, 
                    # and don't decrease distinct counts for non-violating K4s (or if they do, it doesn't drop below 5).
                    
                    # Actually, just calculate the new num_violations.
                    # If new_num_violations < old_num_violations, accept.
                    
                    new_num_violations = num_violations
                    
                    # We need to simulate the change for each affected K4
                    # and see how the violation count changes.
                    
                    # For each k4 in affected_k4s:
                    #   old_distinct = num_colors_in_k4[k4]
                    #   old_is_violating = old_distinct < 5
                    #   
                    #   Calculate new_distinct:
                    #   counts = color_counts[k4]
                    #   old_count_old_color = counts[old_color]
                    #   old_count_new_color = counts[new_color]
                    #   
                    #   new_count_old_color = old_count_old_color - 1
                    #   new_count_new_color = old_count_new_color + 1
                    #   
                    #   new_distinct = old_distinct
                    #   if new_count_old_color == 0: new_distinct -= 1
                    #   if old_count_new_color == 0: new_distinct += 1
                    #   
                    #   new_is_violating = new_distinct < 5
                    #   
                    #   if old_is_violating and not new_is_violating: new_num_violations -= 1
                    #   elif not old_is_violating and new_is_violating: new_num_violations += 1
                    
                    for k4_i in affected_k4s:
                        old_distinct = num_colors_in_k4[k4_i]
                        old_is_violating = old_distinct < 5
                        
                        counts = color_counts[k4_i]
                        c_old = old_color
                        c_new = new_color
                        
                        old_count_old = counts[c_old]
                        old_count_new = counts[c_new]
                        
                        new_count_old = old_count_old - 1
                        new_count_new = old_count_new + 1
                        
                        new_distinct = old_distinct
                        if new_count_old == 0:
                            new_distinct -= 1
                        if old_count_new == 0:
                            new_distinct += 1
                            
                        new_is_violating = new_distinct < 5
                        
                        if old_is_violating and not new_is_violating:
                            new_num_violations -= 1
                        elif not old_is_violating and new_is_violating:
                            new_num_violations += 1
                    
                    if new_num_violations < num_violations:
                        # Accept move
                        colors[edge_idx] = new_color
                        
                        # Update global state
                        for k4_i in affected_k4s:
                            counts = color_counts[k4_i]
                            old_distinct = num_colors_in_k4[k4_i]
                            
                            counts[old_color] -= 1
                            counts[new_color] += 1
                            
                            new_distinct = old_distinct
                            if counts[old_color] == 0:
                                new_distinct -= 1
                            if counts[new_color] == 1: # Was 0
                                new_distinct += 1
                                
                            num_colors_in_k4[k4_i] = new_distinct
                            
                            # Update violating_k4s list
                            old_is_violating = old_distinct < 5
                            new_is_violating = new_distinct < 5
                            
                            if old_is_violating and not new_is_violating:
                                violating_k4s.remove(k4_i)
                            elif not old_is_violating and new_is_violating:
                                violating_k4s.append(k4_i)
                                
                        num_violations = new_num_violations
                        no_improve_counter = 0
                        move_made = True
                        break # Break out of candidate loop
                    else:
                        # Reject candidate
                        pass
                        
                if not move_made:
                    no_improve_counter += 1
                
            # End local search for this restart
            
    return best_solution
