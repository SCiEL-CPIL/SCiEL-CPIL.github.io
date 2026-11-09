import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # 1. Precompute edge indices and K4 structures
    # 120 edges in K16
    edges = []
    edge_idx_map = {}
    idx = 0
    for i in range(16):
        for j in range(i + 1, 16):
            edges.append((i, j))
            edge_idx_map[(i, j)] = idx
            edge_idx_map[(j, i)] = idx
            idx += 1
            
    # Generate all 1820 K4s
    # Each K4 is a tuple of 6 edge indices
    k4s_all = []
    # For each edge, list of K4 indices it belongs to
    edge_k4s = [[] for _ in range(120)]
    
    for quad in combinations(range(16), 4):
        a, b, c, d = quad
        e_indices = [
            edge_idx_map[(a, b)],
            edge_idx_map[(a, c)],
            edge_idx_map[(a, d)],
            edge_idx_map[(b, c)],
            edge_idx_map[(b, d)],
            edge_idx_map[(c, d)]
        ]
        k4_idx = len(k4s_all)
        k4s_all.append(e_indices)
        for ei in e_indices:
            edge_k4s[ei].append(k4_idx)

    # 2. Initial Solution
    # Start with a valid 17-color solution: (i+j) % 17
    # This is known to be valid because in any K4, the sum of vertices mod 17
    # allows distinct colors for disjoint edges unless specific symmetries exist,
    # but (i+j) mod 17 is a standard construction for this type of problem.
    # Let's verify validity of this base.
    colors = [0] * 120
    for idx, (i, j) in enumerate(edges):
        colors[idx] = (i + j) % 17
        
    # Helper to check if a K4 is valid (>= 5 distinct colors)
    def is_valid_k4(k4_idx, cols):
        e_indices = k4s_all[k4_idx]
        c_set = set()
        for ei in e_indices:
            c_set.add(cols[ei])
            if len(c_set) >= 5:
                return True
        return len(c_set) >= 5

    # Helper to check validity of all K4s affected by a change
    # Returns True if all affected K4s are valid
    def check_local_validity(e_idx, cols):
        for k4_idx in edge_k4s[e_idx]:
            if not is_valid_k4(k4_idx, cols):
                return False
        return True

    # 3. Local Search / Simulated Annealing
    # Objective: Minimize number of distinct colors.
    # We maintain a current solution and a best solution.
    # We try to recolor edges.
    # To reduce colors, we bias towards existing low colors.
    
    best_colors = colors[:]
    best_num_colors = len(set(colors))
    
    # Precompute list of currently used colors for bias? 
    # Dynamic set of used colors is expensive to update per step if we do it naively.
    # Instead, we can just sample from a small range [0, max_color+1].
    
    # Let's use a simple hill climbing with occasional random jumps.
    # Since the constraint is hard (must be valid), we only accept moves that maintain validity.
    # This is a constrained local search.
    
    # To escape local minima, we can allow temporary violations? 
    # No, the problem requires a valid solution at the end. 
    # If we allow violations, we need to repair. 
    # Given the tight constraint (>=5 colors in every K4), violations are likely hard to repair quickly.
    # Let's stick to valid-only moves but use a smart proposal distribution.
    
    # Proposal: Pick a random edge. Pick a new color.
    # Bias: Try colors that are already in use, especially lower ones.
    # If we pick a color not in use, we increase distinct count (bad).
    # If we pick a color in use, distinct count stays same or decreases (if old color becomes unused).
    
    # Strategy:
    # 1. Identify a "victim" color: a color used by few edges.
    # 2. Try to recolor edges with that color to other existing colors.
    
    # Simpler: Random edge, random color from [0, current_max].
    # If new color == old color, skip.
    # If valid, accept.
    # Periodically check if distinct count decreased.
    
    # Optimization: 
    # Maintaining a count of how many edges have each color allows us to know 
    # if removing a color reduces the total count.
    
    color_counts = [0] * 120
    for c in colors:
        color_counts[c] += 1
        
    def get_distinct_count(counts):
        return sum(1 for c in counts if c > 0)

    current_distinct = get_distinct_count(color_counts)
    
    # Simulated Annealing parameters
    # Since we only accept valid moves, "temperature" doesn't apply to accepting bad moves.
    # Instead, we can use temperature to decide *which* valid move to try or how aggressive the search is.
    # Or we can just run for a fixed time.
    
    # Let's add a mechanism to try to "merge" colors.
    # If we have colors 0..k-1, we try to map color k-1 to 0..k-2.
    
    # Loop
    step = 0
    while time.time() < deadline:
        step += 1
        
        # Determine max color currently in use
        # To avoid O(120) scan every step, we can track max_color.
        # But color_counts changes.
        # Let's just compute max_color occasionally or keep it.
        
        # Pick a random edge
        e_idx = random.randint(0, 119)
        old_color = colors[e_idx]
        
        # Candidate colors:
        # 1. Random color in [0, max_color_used]
        # 2. Specifically, try to replace old_color with a lower color if possible.
        
        # Find max color used
        max_c = -1
        for c in range(119, -1, -1):
            if color_counts[c] > 0:
                max_c = c
                break
        
        if max_c < 0:
            break
            
        # Pick new color
        # Bias towards lower colors to encourage compression
        # Probabilistic choice:
        # With prob 0.5, pick a random color from [0, max_c]
        # With prob 0.5, pick a random color from [0, min(10, max_c)] to force low indices
        
        if random.random() < 0.5:
            new_color = random.randint(0, max_c)
        else:
            new_color = random.randint(0, min(15, max_c))
            
        if new_color == old_color:
            continue
            
        # Try the move
        colors[e_idx] = new_color
        color_counts[old_color] -= 1
        color_counts[new_color] += 1
        
        if check_local_validity(e_idx, colors):
            # Valid move, accept
            # Check if distinct count decreased
            # We don't need to recompute full distinct count if we know:
            # If color_counts[old_color] becomes 0, distinct count decreases by 1.
            # If color_counts[new_color] was 0, distinct count increases by 1.
            
            # Update best
            # We can track distinct count incrementally
            # But for simplicity and safety, let's just check periodically or use the counts.
            
            # Incremental distinct count update:
            # delta = (1 if color_counts[new_color] == 1 else 0) - (1 if color_counts[old_color] == 0 else 0)
            # Wait, color_counts are already updated.
            # Let's re-evaluate:
            # Before: old_c had count > 0, new_c had count >= 0.
            # After: old_c count decreased, new_c count increased.
            
            # It's easier to just compute distinct count every N steps or when a change happens.
            # Given 120 edges, computing distinct count is fast.
            
            if step % 10 == 0 or color_counts[old_color] == 0:
                current_distinct = get_distinct_count(color_counts)
                if current_distinct < best_num_colors:
                    best_num_colors = current_distinct
                    best_colors = colors[:]
        else:
            # Invalid move, revert
            colors[e_idx] = old_color
            color_counts[old_color] += 1
            color_counts[new_color] -= 1
    
    # Final compression
    # Map used colors in best_colors to 0..k-1
    unique_colors = sorted(list(set(best_colors)))
    mapping = {c: i for i, c in enumerate(unique_colors)}
    final_colors = [mapping[c] for c in best_colors]
    
    return final_colors

# EVOLVE-BLOCK-END
