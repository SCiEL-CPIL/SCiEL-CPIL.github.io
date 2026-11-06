import time
import random
import sys

N = 16
NUM_EDGES = 120

def get_edge_idx(i, j):
    if i > j:
        i, j = j, i
    return 15 * i - i * (i - 1) // 2 + (j - i - 1)

# Precompute edges
EDGES = []
for i in range(N):
    for j in range(i + 1, N):
        EDGES.append((i, j))

# Precompute K4s
K4_LIST = []
K4_EDGE_INDICES = []
for a in range(N):
    for b in range(a + 1, N):
        for c in range(b + 1, N):
            for d in range(c + 1, N):
                edges = [(a, b), (a, c), (a, d), (b, c), (b, d), (c, d)]
                e_idxs = [get_edge_idx(u, v) for u, v in edges]
                K4_EDGE_INDICES.append(e_idxs)

NUM_K4 = len(K4_EDGE_INDICES)

# Map edge to list of K4 indices
EDGE_TO_K4 = [[] for _ in range(NUM_EDGES)]
for k4_idx, e_idxs in enumerate(K4_EDGE_INDICES):
    for e_idx in e_idxs:
        EDGE_TO_K4[e_idx].append(k4_idx)

def solve_backtracking(K, time_limit):
    start_time = time.time()
    colors = [-1] * NUM_EDGES
    # k4_colors[k4_idx] is a list of colors used in assigned edges of that K4
    # We need to track distinct colors, so a set or bitmask is better.
    # But for speed, we can just recalculate or maintain a count.
    # Let's maintain a bitmask for each K4 of colors used.
    k4_masks = [0] * NUM_K4
    # Count of assigned edges per K4
    k4_assigned_count = [0] * NUM_K4
    
    # Order of edges to assign
    # Simple order: 0 to 119
    # Better order: maybe group by vertices?
    # Let's stick to 0..119 for now, but shuffle? No, deterministic is better for backtracking.
    # However, a bad order can be very slow.
    # Let's try an order that completes K4s quickly.
    # Edges (0,1), (0,2), (1,2) complete a triangle.
    # (0,3) completes K4(0,1,2,3) with (1,3), (2,3).
    
    # Let's use a static order based on vertex index sum?
    # Or just 0..119.
    
    def is_consistent(edge_idx, color):
        # Check all K4s containing this edge
        for k4_idx in EDGE_TO_K4[edge_idx]:
            e_idxs = K4_EDGE_INDICES[k4_idx]
            # Check if adding this color violates the max possible distinct colors
            # Current mask: k4_masks[k4_idx]
            # Current assigned count: k4_assigned_count[k4_idx]
            # If we add this color, new_mask = k4_masks[k4_idx] | (1 << color)
            # new_assigned_count = k4_assigned_count[k4_idx] + 1
            # unassigned_count = 6 - new_assigned_count
            # max_possible_distinct = bit_count(new_mask) + unassigned_count
            # If max_possible_distinct < 5, return False
            
            new_mask = k4_masks[k4_idx] | (1 << color)
            new_assigned = k4_assigned_count[k4_idx] + 1
            unassigned = 6 - new_assigned
            distinct_so_far = bin(new_mask).count('1')
            if distinct_so_far + unassigned < 5:
                return False
        return True

    def backtrack(idx):
        if time.time() - start_time > time_limit:
            return False
            
        if idx == NUM_EDGES:
            return True
            
        e_idx = idx
        # Try colors 0..K-1
        # Randomize color order to avoid bias?
        colors_to_try = list(range(K))
        random.shuffle(colors_to_try)
        
        for color in colors_to_try:
            if is_consistent(e_idx, color):
                # Assign
                colors[e_idx] = color
                # Update K4 masks
                for k4_idx in EDGE_TO_K4[e_idx]:
                    k4_masks[k4_idx] |= (1 << color)
                    k4_assigned_count[k4_idx] += 1
                    
                if backtrack(idx + 1):
                    return True
                else:
                    # Undo
                    colors[e_idx] = -1
                    for k4_idx in EDGE_TO_K4[e_idx]:
                        # We need to recalculate mask? 
                        # Removing a bit from a mask is tricky if other edges in the K4 have the same color.
                        # So we should NOT use a simple bitmask for undo if colors can repeat.
                        # We need to count occurrences.
                        pass
        return False

    # The undo logic for bitmask is flawed if colors repeat in a K4.
    # We need to maintain a count of each color in each K4.
    # k4_color_counts[k4_idx][color] = count
    # This is heavy.
    
    # Alternative: Recalculate the mask from scratch for the K4s affected?
    # There are only 6 edges in a K4.
    # So in is_consistent, we can just look at the colors of the other 5 edges.
    
    def is_consistent_v2(edge_idx, color):
        for k4_idx in EDGE_TO_K4[edge_idx]:
            e_idxs = K4_EDGE_INDICES[k4_idx]
            # Get colors of other edges
            other_colors = []
            unassigned = 0
            for e in e_idxs:
                if e == edge_idx:
                    continue
                if colors[e] == -1:
                    unassigned += 1
                else:
                    other_colors.append(colors[e])
            
            # Add the new color
            all_colors = other_colors + [color]
            distinct_so_far = len(set(all_colors))
            # Total assigned in this K4 will be 1 (new) + (6 - 1 - unassigned) = 6 - unassigned
            # Wait, unassigned is the count of other unassigned edges.
            # So total unassigned after assignment is unassigned.
            # Max possible distinct = distinct_so_far + unassigned
            if distinct_so_far + unassigned < 5:
                return False
        return True

    def backtrack_v2(idx):
        if time.time() - start_time > time_limit:
            return False
            
        if idx == NUM_EDGES:
            return True
            
        e_idx = idx
        colors_to_try = list(range(K))
        random.shuffle(colors_to_try)
        
        for color in colors_to_try:
            if is_consistent_v2(e_idx, color):
                colors[e_idx] = color
                if backtrack_v2(idx + 1):
                    return True
                else:
                    colors[e_idx] = -1
        return False

    # Increase recursion limit
    sys.setrecursionlimit(10000)
    
    if backtrack_v2(0):
        return colors
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Try 13, 14, 15
    for K in [13, 14, 15]:
        remaining_time = deadline - time.time()
        if remaining_time < 10:
            break
        # Allocate time for this K
        time_for_K = min(remaining_time - 5, 30.0)
        
        # Try a few restarts if backtracking fails quickly?
        # Backtracking is deterministic if order is fixed.
        # We shuffled colors, so it's randomized.
        
        colors = solve_backtracking(K, time_for_K)
        if colors is not None:
            # Verify
            # (Optional)
            return colors
            
    # Fallback: Seed solution (17 colors)
    # Or a known 15 color solution?
    # Let's just return the seed fallback to be safe, though 15 is better.
    # But if we couldn't find 15, maybe it's hard.
    # The seed is (i+j)%17.
    return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
