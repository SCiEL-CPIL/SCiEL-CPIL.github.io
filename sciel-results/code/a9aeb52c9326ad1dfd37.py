import random
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin before 150s kill

    # Precompute edge indices
    # Edge (i, j) with 0 <= i < j < 16
    # Index mapping:
    # Row 0: (0,1)..(0,15) -> 15 edges
    # Row 1: (1,2)..(1,15) -> 14 edges
    # ...
    # Index of (i, j) = sum_{k=0}^{i-1} (15-k) + (j - i - 1)
    # Sum_{k=0}^{i-1} (15-k) = 15*i - i*(i-1)/2
    def get_edge_idx(i, j):
        if i > j: i, j = j, i
        return 15*i - i*(i-1)//2 + (j - i - 1)

    # Precompute all K4s
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
    
    # Map edge to list of K4 indices
    edge_to_k4s = [[] for _ in range(120)]
    for k4_idx, edges in enumerate(k4s):
        for e in edges:
            edge_to_k4s[e].append(k4_idx)

    # Fallback solution: Seed (17 colors)
    # (i+j) % 17
    seed_solution = [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
    best_solution = seed_solution[:]
    best_color_count = len(set(seed_solution)) # 17

    # Helper to check validity
    def is_valid(colors):
        for edges in k4s:
            if len(set(colors[e] for e in edges)) < 5:
                return False
        return True

    # Verify seed
    if is_valid(best_solution):
        best_color_count = len(set(best_solution))
    else:
        # This shouldn't happen, but if it does, we have a bug in seed generation or check
        # Force a valid high-color solution just in case
        # Using 16 colors: (i+j) % 16
        # Check if valid
        alt = [(i + j) % 16 for i in range(16) for j in range(i + 1, 16)]
        if is_valid(alt):
            best_solution = alt
            best_color_count = 16
        else:
            # Use 15 colors?
            alt2 = [(i + j) % 15 for i in range(16) for j in range(i + 1, 16)]
            if is_valid(alt2):
                best_solution = alt2
                best_color_count = 15
            else:
                # Use distinct colors for all edges? 120 colors.
                # But we need to minimize.
                # Just return seed.
                pass

    # Try to find better solutions with C colors
    # Prioritize 15, then 14, then 13
    targets = [15, 14, 13]
    
    # Time allocation
    # 15: 60s, 14: 40s, 13: 15s, buffer: 5s
    time_alloc = {15: 60, 14: 40, 13: 15}
    
    for C in targets:
        if time.time() > deadline - 5:
            break
            
        alloc = time_alloc[C]
        slice_end = min(deadline - 2, time.time() + alloc)
        
        # Local Search / Simulated Annealing
        # We will run multiple restarts within the time slice
        restarts = 0
        while time.time() < slice_end:
            restarts += 1
            # Random initialization
            colors = [random.randint(0, C - 1) for _ in range(120)]
            
            # Calculate initial state
            k4_sets = [None] * num_k4s
            for i, edges in enumerate(k4s):
                k4_sets[i] = set(colors[e] for e in edges)
                
            num_violations = sum(1 for s in k4_sets if len(s) < 5)
            
            if num_violations == 0:
                actual_colors = len(set(colors))
                if actual_colors < best_color_count:
                    best_solution = colors[:]
                    best_color_count = actual_colors
                    if best_color_count == 13:
                        return best_solution
                continue
            
            # Annealing parameters
            temp = 1.0
            cooling = 0.999
            min_temp = 0.01
            
            iterations = 0
            max_iterations = 50000 # Cap per restart
            
            while num_violations > 0 and iterations < max_iterations and time.time() < slice_end:
                iterations += 1
                
                # Pick a random violating K4
                # To speed up, we could maintain a list of violating indices, but random pick is simpler
                # and 1820 is small.
                while True:
                    k4_idx = random.randint(0, num_k4s - 1)
                    if len(k4_sets[k4_idx]) < 5:
                        break
                
                edges = k4s[k4_idx]
                edge_idx = edges[random.randint(0, 5)]
                
                old_color = colors[edge_idx]
                
                # Generate new color
                # Bias towards colors not in the current K4 set to help fix violation
                current_colors_in_k4 = k4_sets[k4_idx]
                
                if random.random() < 0.7:
                    # Pick a color NOT in current_colors_in_k4
                    # If all C colors are in the set (impossible if C<5 and set<5, but if C>=5 and set has <5, there are C-len(set) options)
                    # If set has 4 colors, and C=5, there is 1 option.
                    available = [c for c in range(C) if c not in current_colors_in_k4]
                    if available:
                        new_color = random.choice(available)
                    else:
                        new_color = random.randint(0, C - 1)
                else:
                    new_color = random.randint(0, C - 1)
                
                if new_color == old_color:
                    continue
                
                # Calculate delta
                # Update affected K4s
                affected_k4s = edge_to_k4s[edge_idx]
                
                # Store old state to rollback
                old_states = []
                new_violations_delta = 0
                
                for k4_i in affected_k4s:
                    s = k4_sets[k4_i]
                    old_len = len(s)
                    
                    # Temporarily apply change to calculate new set
                    # We need to know if old_color is still in the set after removing one instance?
                    # The set is just the set of colors present.
                    # If we change an edge from old to new:
                    # 1. If old_color is only on this edge in this K4, it might disappear.
                    # 2. If new_color is not present, it might appear.
                    
                    # Efficient check:
                    # Count occurrences of old_color and new_color in the 6 edges of this K4.
                    k4_edges = k4s[k4_i]
                    old_count = 0
                    new_count = 0
                    for e in k4_edges:
                        if colors[e] == old_color:
                            old_count += 1
                        if colors[e] == new_color:
                            new_count += 1
                    
                    # Note: colors[edge_idx] is still old_color at this point in the loop?
                    # We haven't updated colors[] yet.
                    # So old_count includes the edge we are changing.
                    # new_count does NOT include the edge we are changing (unless new_color==old_color, which we skipped).
                    
                    # After change:
                    # old_color count becomes old_count - 1
                    # new_color count becomes new_count + 1
                    
                    new_set = set(s)
                    
                    # Remove old_color if it was unique to this edge in this K4
                    if old_count == 1:
                        new_set.discard(old_color)
                    else:
                        # old_color remains
                        pass
                    
                    # Add new_color
                    new_set.add(new_color)
                    
                    new_len = len(new_set)
                    
                    # Calculate violation change for this K4
                    old_violation = 1 if old_len < 5 else 0
                    new_violation = 1 if new_len < 5 else 0
                    new_violations_delta += (new_violation - old_violation)
                    
                    old_states.append((k4_i, s, new_set))
                
                # Acceptance criterion
                # If delta < 0, always accept
                # If delta >= 0, accept with probability exp(-delta / temp)
                accept = False
                if new_violations_delta < 0:
                    accept = True
                else:
                    if new_violations_delta == 0:
                        accept = True # Neutral move, helps exploration
                    else:
                        # Simulated annealing step
                        # delta is positive integer
                        prob = pow(2.71828, -new_violations_delta / temp)
                        if random.random() < prob:
                            accept = True
                
                if accept:
                    # Apply change
                    colors[edge_idx] = new_color
                    num_violations += new_violations_delta
                    for k4_i, old_s, new_s in old_states:
                        k4_sets[k4_i] = new_s
                else:
                    # Rollback is implicit as we didn't modify colors or k4_sets permanently
                    pass
                
                temp *= cooling
                if temp < min_temp:
                    temp = min_temp

            # End local search for this restart
            if num_violations == 0:
                actual_colors = len(set(colors))
                if actual_colors < best_color_count:
                    best_solution = colors[:]
                    best_color_count = actual_colors
                    if best_color_count == 13:
                        return best_solution
                # If we found a valid solution for C, we can break out of restarts for this C
                # and move to next C (to try to find fewer colors)
                break

    return best_solution
