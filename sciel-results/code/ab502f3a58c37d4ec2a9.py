import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 40s safety margin
    
    n = 16
    num_edges = n * (n - 1) // 2
    
    # Generate all edges (i, j) with i < j
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    edge_index = {e: idx for idx, e in enumerate(edges)}
    
    # Generate all K4s (sets of 4 vertices)
    k4s = []
    k4_edge_indices = []
    from itertools import combinations
    for quad in combinations(range(n), 4):
        k4s.append(quad)
        e_indices = []
        for i in range(4):
            for j in range(i + 1, 4):
                e_indices.append(edge_index[(quad[i], quad[j])])
        k4_edge_indices.append(e_indices)
        
    num_k4s = len(k4s)
    
    # Map edge to list of K4 indices containing it
    edge_to_k4s = [[] for _ in range(num_edges)]
    for k4_idx, e_indices in enumerate(k4_edge_indices):
        for e_idx in e_indices:
            edge_to_k4s[e_idx].append(k4_idx)
            
    # Precompute K4 edge indices as numpy arrays for faster access? 
    # Or just lists. Lists are fine.
    
    def count_distinct_colors(e_indices, colors):
        # e_indices: list of 6 edge indices
        # colors: list of current colors
        c0 = colors[e_indices[0]]
        c1 = colors[e_indices[1]]
        c2 = colors[e_indices[2]]
        c3 = colors[e_indices[3]]
        c4 = colors[e_indices[4]]
        c5 = colors[e_indices[5]]
        # Unroll for speed? Or use set?
        # Set is slow. 
        # Since 6 elements, manual comparison is faster.
        # But let's try a small helper.
        s = {c0, c1, c2, c3, c4, c5}
        return len(s)
        
    def get_cost(colors):
        cost = 0
        for e_indices in k4_edge_indices:
            if count_distinct_colors(e_indices, colors) < 5:
                cost += 1
        return cost

    def delta_cost(e_idx, new_color, colors):
        delta = 0
        for k4_idx in edge_to_k4s[e_idx]:
            e_indices = k4_edge_indices[k4_idx]
            old_distinct = count_distinct_colors(e_indices, colors)
            
            # Simulate change
            # Find position of e_idx in e_indices
            # Since e_indices is small (6), linear search is fine
            pos = -1
            for i, ei in enumerate(e_indices):
                if ei == e_idx:
                    pos = i
                    break
            if pos == -1:
                continue # Should not happen
                
            # Create a modified list of colors for this K4
            # To avoid allocation, we can just check the set logic
            old_c = colors[e_idx]
            
            # Efficient distinct count with one change
            # Get the 6 colors
            cs = [colors[ei] for ei in e_indices]
            cs[pos] = new_color
            
            # Count distinct in cs
            # Using set is likely fast enough for 6 items
            new_distinct = len(set(cs))
            
            old_bad = 1 if old_distinct < 5 else 0
            new_bad = 1 if new_distinct < 5 else 0
            delta += new_bad - old_bad
            
        return delta

    best_coloring = None
    best_cost = float('inf')
    
    # Try to find a 13-color solution
    num_colors = 13
    
    # Initialize with random colors
    colors = [random.randint(0, num_colors - 1) for _ in range(num_edges)]
    current_cost = get_cost(colors)
    
    if current_cost == 0:
        best_coloring = colors[:]
        best_cost = 0
    
    # Local Search
    while time.time() < deadline:
        if current_cost == 0:
            # Found a valid solution!
            # Check if we can reduce colors further? 
            # Lower bound is 13, so we can't go below 13.
            # But we might be using fewer than 13? 
            # The "value" is the number of distinct colors used.
            # If we use 12, that's better. But LB is 13.
            # So 13 is the target.
            # We should return immediately or try to optimize for fewer distinct colors if possible?
            # No, LB is 13.
            best_coloring = colors[:]
            best_cost = 0
            break
            
        # Pick a random edge
        e_idx = random.randint(0, num_edges - 1)
        # Pick a random new color
        new_color = random.randint(0, num_colors - 1)
        
        if new_color == colors[e_idx]:
            continue
            
        dc = delta_cost(e_idx, new_color, colors)
        
        # Accept if delta <= 0 (Hill Climbing)
        # Or use SA: accept if delta < 0 or random() < exp(-delta/T)
        # Given the tight time, Hill Climbing is safer to ensure progress.
        # But we might get stuck in local minima.
        # Let's use a simple SA with cooling.
        
        # Actually, let's just try Hill Climbing first. 
        # If stuck, restart?
        
        if dc <= 0:
            colors[e_idx] = new_color
            current_cost += dc
            
    # If we didn't find a 0 cost solution, try 14 colors?
    # Or just return the seed if no valid solution found.
    
    if best_coloring is not None:
        # Ensure colors are in 0..119
        # They are 0..12, so they are fine.
        return best_coloring
    else:
        # Fallback to seed
        return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
