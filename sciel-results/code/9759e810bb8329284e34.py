import time
import random
import math
import numpy as np

N = 16
NUM_EDGES = N * (N - 1) // 2
MAX_COLOR = 120

def edge_idx(i, j):
    if i > j:
        i, j = j, i
    return i * (N - 1) - i * (i - 1) // 2 + (j - i - 1)

# Precompute cycles and their edge indices
CYCLES = []
for a in range(N):
    for b in range(a + 1, N):
        for c in range(b + 1, N):
            for d in range(c + 1, N):
                # Cycle 1: a-b-c-d-a
                CYCLES.append((edge_idx(a, b), edge_idx(b, c), edge_idx(c, d), edge_idx(d, a)))
                # Cycle 2: a-b-d-c-a
                CYCLES.append((edge_idx(a, b), edge_idx(b, d), edge_idx(d, c), edge_idx(c, a)))
                # Cycle 3: a-c-b-d-a
                CYCLES.append((edge_idx(a, c), edge_idx(c, b), edge_idx(b, d), edge_idx(d, a)))

NUM_CYCLES = len(CYCLES)
cycle_edges = np.array(CYCLES, dtype=np.int32)

# For each edge, which cycles does it belong to?
edge_to_cycles = [[] for _ in range(NUM_EDGES)]
for c_idx in range(NUM_CYCLES):
    for e in cycle_edges[c_idx]:
        edge_to_cycles[e].append(c_idx)

# Precompute for each edge, the indices of the other 3 edges in each cycle it belongs to
# This allows O(1) or O(k) validity check where k is number of cycles containing the edge
edge_cycle_neighbors = []
for e in range(NUM_EDGES):
    neighbors = []
    for c_idx in edge_to_cycles[e]:
        cycle = cycle_edges[c_idx]
        others = [x for x in cycle if x != e]
        neighbors.append(others)
    edge_cycle_neighbors.append(neighbors)

def is_valid_assignment(colors, e, new_color):
    """
    Check if setting colors[e] = new_color results in any rainbow cycle.
    Only needs to check cycles containing e.
    """
    for others in edge_cycle_neighbors[e]:
        # The cycle consists of 'e' and 'others' (3 edges)
        # Colors of the 4 edges:
        c1 = new_color
        c2 = colors[others[0]]
        c3 = colors[others[1]]
        c4 = colors[others[2]]
        
        # Check if all 4 are distinct
        # Fast check: 
        if c1 != c2 and c1 != c3 and c1 != c4 and c2 != c3 and c2 != c4 and c3 != c4:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    best_score = 0
    best_colors = None
    
    # GRASP: Multiple iterations of Constructive + Local Search
    # We will run several iterations, each starting with a different randomized greedy construction
    
    num_grasp_iterations = 0
    
    while time.time() < deadline:
        num_grasp_iterations += 1
        
        # --- Phase 1: Greedy Construction ---
        # Initialize with -1 (unassigned)
        current_colors = np.full(NUM_EDGES, -1, dtype=np.int32)
        global_color_counts = np.zeros(MAX_COLOR, dtype=np.int32)
        
        # We assign edges in a random order to break symmetry and allow different paths
        edge_order = list(range(NUM_EDGES))
        random.shuffle(edge_order)
        
        # To make it "adaptive", we prioritize colors that lead to more diversity later?
        # Simple adaptive heuristic: When choosing a color for edge e, 
        # prefer colors that are currently less frequent, or new colors if possible.
        
        feasible = True
        for e in edge_order:
            # Determine candidate colors
            # A color is valid if is_valid_assignment(current_colors, e, c) is True
            # We want to maximize distinct colors.
            
            # Heuristic: 
            # 1. Try new colors first (if they don't break validity)
            # 2. Try existing rare colors
            # 3. Try existing common colors
            
            # To speed up, we don't check all 120 colors for every edge.
            # We pick a small set of candidates.
            
            # Strategy: 
            # If we can use a new color, we should try to.
            # But using a new color might block future edges.
            # Let's try a mix: 
            # - With prob 0.7, try to pick a color that maximizes local diversity (e.g., a color not used in neighbors)
            # - With prob 0.3, pick a random valid color
            
            # Actually, let's just iterate through a subset of colors.
            # For speed, we check a random permutation of colors, stop at first valid?
            # That's too slow if many are invalid.
            
            # Better: Pre-compute which colors are "safe" for this edge given current state?
            # That's expensive.
            
            # Let's use a simple greedy:
            # Try colors in order of (frequency ascending, random tie-break).
            # This encourages using rare/new colors.
            
            # Get current frequencies
            # To avoid sorting 120 colors every time, we can maintain a list or just sample.
            
            # Optimization: 
            # Check a small number of candidate colors (e.g., 20) sampled from the 120.
            # If none are valid, try all 120.
            
            candidates = list(range(MAX_COLOR))
            random.shuffle(candidates)
            
            # Sort candidates by frequency (ascending) to prefer rare/new
            # But sorting 120 items * 120 edges * many iterations might be slow.
            # Let's just sample 10 candidates, check validity, pick the one with lowest freq.
            
            best_cand = -1
            best_freq = MAX_COLOR + 1
            best_cand_is_new = False
            
            # Check 10 random candidates
            for c in candidates[:10]:
                if is_valid_assignment(current_colors, e, c):
                    freq = global_color_counts[c]
                    if freq < best_freq:
                        best_freq = freq
                        best_cand = c
                        if freq == 0:
                            best_cand_is_new = True
                            # If we found a new color, it's great, but maybe check one more just in case?
                            # No, new color is best for diversity. Break?
                            # Actually, any new color is equivalent for score. 
                            # But different new colors might have different future implications.
                            # Let's keep searching for a moment? No, keep it fast.
                            pass
            
            # If no candidate found in top 10, check the rest
            if best_cand == -1:
                for c in candidates[10:]:
                    if is_valid_assignment(current_colors, e, c):
                        freq = global_color_counts[c]
                        if freq < best_freq:
                            best_freq = freq
                            best_cand = c
            
            if best_cand == -1:
                # This shouldn't happen if color 0 is always valid? 
                # Actually, color 0 might not be valid if it creates a rainbow.
                # But there must be at least one valid color for any edge in a valid partial assignment?
                # Not necessarily. But with 120 colors, it's very likely.
                # If we get stuck, we have to backtrack or restart.
                # For simplicity, we'll restart the construction if we get stuck.
                feasible = False
                break
            
            current_colors[e] = best_cand
            global_color_counts[best_cand] += 1
        
        if not feasible:
            continue
            
        current_score = np.count_nonzero(global_color_counts)
        
        if current_score > best_score:
            best_score = current_score
            best_colors = current_colors.copy()
        
        # --- Phase 2: Local Search (Hill Climbing) ---
        # Improve the constructed solution
        T = 2.0
        T_min = 0.1
        alpha = 0.99
        no_improve = 0
        max_no_improve = 2000
        
        while time.time() < deadline and no_improve < max_no_improve:
            e = random.randint(0, NUM_EDGES - 1)
            old_color = current_colors[e]
            
            # Choose a new color
            # Prefer colors that increase diversity or maintain it
            new_color = random.randint(0, MAX_COLOR - 1)
            
            if new_color == old_color:
                continue
            
            if is_valid_assignment(current_colors, e, new_color):
                old_count = global_color_counts[old_color]
                new_count = global_color_counts[new_color]
                
                delta = 0
                if new_count == 0: delta += 1
                if old_count == 1: delta -= 1
                
                new_score = current_score + delta
                
                if new_score > best_score:
                    best_score = new_score
                    best_colors = current_colors.copy()
                    no_improve = 0
                else:
                    no_improve += 1
                    
                if delta >= 0:
                    current_score = new_score
                    global_color_counts[old_color] -= 1
                    global_color_counts[new_color] += 1
                else:
                    if random.random() < math.exp(delta / T):
                        current_score = new_score
                        global_color_counts[old_color] -= 1
                        global_color_counts[new_color] += 1
                    else:
                        pass # Revert (do nothing)
            else:
                no_improve += 1
                
            T *= alpha
            if T < T_min:
                T = T_min
                
        # If we exited due to no_improve, restart GRASP

    return best_colors.tolist() if best_colors is not None else [0]*NUM_EDGES
