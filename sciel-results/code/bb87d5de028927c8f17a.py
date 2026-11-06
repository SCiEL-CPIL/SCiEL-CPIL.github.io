import time
import random
from itertools import combinations
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s margin
    
    n = 16
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((i, j))
            
    # Map (i, j) to edge index
    ij_to_edge_idx = {e: idx for idx, e in enumerate(edges)}
    
    # Precompute for each K4, the indices of its 6 edges
    k4s = list(combinations(range(n), 4))
    k4_edge_indices = []
    for verts in k4s:
        edges_in_k4 = []
        for i in range(4):
            for j in range(i + 1, 4):
                u = verts[i]
                v = verts[j]
                edges_in_k4.append(ij_to_edge_idx[(u, v)])
        k4_edge_indices.append(edges_in_k4)
    
    # Convert to numpy array for faster operations
    k4_edge_indices_np = np.array(k4_edge_indices, dtype=np.int32)
    
    def is_valid_fast(col):
        """Check if every K4 has at least 5 distinct colors using vectorized numpy."""
        col_arr = np.array(col, dtype=np.int32)
        k4_colors = col_arr[k4_edge_indices_np]  # shape: (1820, 6)
        
        # For each K4, check if it has < 5 distinct colors
        # A K4 has < 5 distinct colors if some color appears >= 2 times
        # and the total distinct count is < 5
        
        # More efficient: count how many pairs of edges in each K4 have the same color
        # If a K4 has >= 5 distinct colors, then at most 1 pair of edges can share a color
        # (since 6 edges, 5+ distinct means at most 1 duplicate pair)
        
        # Actually, let's use a different approach:
        # For each K4, compute the number of distinct colors by checking all 15 pairs
        
        # Simpler vectorized approach: for each K4, check if any color appears 3+ times
        # or if two different colors each appear 2+ times (which would give <= 4 distinct)
        
        # Let me use a direct approach with numpy
        # For each K4 (row), sort the 6 colors and check distinct count
        sorted_colors = np.sort(k4_colors, axis=1)
        
        # Count distinct: number of positions where color changes + 1
        # For each row, count how many times adjacent elements differ
        diffs = np.diff(sorted_colors, axis=1)  # shape: (1820, 5)
        distinct_counts = np.sum(diffs != 0, axis=1) + 1  # shape: (1820,)
        
        # Valid if all K4s have >= 5 distinct colors
        return bool(np.all(distinct_counts >= 5))
    
    def is_valid(col):
        """Check if every K4 has at least 5 distinct colors."""
        return is_valid_fast(col)
    
    def count_colors(col):
        """Count distinct colors used."""
        return len(set(col))
    
    # Try multiple initial colorings
    best_colors = None
    best_count = 999
    
    # Initial coloring strategies
    strategies = [
        lambda i, j: (i + j) % 17,
        lambda i, j: (i * 2 + j) % 17,
        lambda i, j: (i + 2*j) % 17,
        lambda i, j: (i * j) % 17,
        lambda i, j: (i + j + i*j) % 17,
        lambda i, j: (i * 3 + j * 2) % 17,
        lambda i, j: (i + j + 1) % 17,
    ]
    
    for strategy in strategies:
        if time.time() > deadline:
            break
        
        colors = [0] * 120
        for idx, (i, j) in enumerate(edges):
            colors[idx] = strategy(i, j)
        
        if is_valid(colors):
            c = count_colors(colors)
            if c < best_count:
                best_count = c
                best_colors = colors[:]
    
    if best_colors is None:
        # Fallback to simple coloring
        colors = [0] * 120
        for idx, (i, j) in enumerate(edges):
            colors[idx] = (i + j) % 17
        return colors
    
    colors = best_colors[:]
    used_colors = set(colors)
    best_count = len(used_colors)
    
    # Merging loop with restarts
    max_restarts = 100
    restarts = 0
    
    while time.time() < deadline and restarts < max_restarts:
        merged_any = False
        current_colors_list = sorted(used_colors)
        
        # Try all pairs (u, v) with u != v
        # We try to eliminate 'u' by mapping it to 'v'
        for u in current_colors_list:
            if time.time() > deadline:
                break
            for v in current_colors_list:
                if u == v:
                    continue
                if time.time() > deadline:
                    break
                
                # Check if u is still present
                if u not in used_colors:
                    continue
                
                # Create a temporary copy
                temp_colors = colors[:]
                # Find edges with color u and change to v
                for k in range(120):
                    if temp_colors[k] == u:
                        temp_colors[k] = v
                        
                # Check validity
                if is_valid(temp_colors):
                    # Accept merge
                    colors = temp_colors
                    used_colors.remove(u)
                    merged_any = True
                    # Restart the pair loop with the new state
                    break
            
            if merged_any:
                break
            if time.time() > deadline:
                break
                
        if not merged_any:
            # No simple merge possible.
            # Try random restart with perturbation
            restarts += 1
            if restarts < max_restarts and time.time() < deadline:
                # Perturb the current solution
                num_perturb = random.randint(1, 4)
                for _ in range(num_perturb):
                    # Pick a random edge and change its color
                    edge_idx = random.randint(0, 119)
                    old_color = colors[edge_idx]
                    new_color = random.choice(list(used_colors))
                    if new_color != old_color:
                        colors[edge_idx] = new_color
                        used_colors.add(new_color)
                        if old_color not in colors:
                            used_colors.discard(old_color)
                
                # Try to merge again
                current_colors_list = sorted(used_colors)
                for u in current_colors_list:
                    if time.time() > deadline:
                        break
                    for v in current_colors_list:
                        if u == v:
                            continue
                        if time.time() > deadline:
                            break
                        
                        if u not in used_colors:
                            continue
                        
                        temp_colors = colors[:]
                        for k in range(120):
                            if temp_colors[k] == u:
                                temp_colors[k] = v
                                
                        if is_valid(temp_colors):
                            colors = temp_colors
                            used_colors.remove(u)
                            merged_any = True
                            break
                    
                    if merged_any:
                        break
                    if time.time() > deadline:
                        break
        
        # Update best
        current_count = len(used_colors)
        if current_count < best_count:
            best_count = current_count
            best_colors = colors[:]
        
        if not merged_any and restarts >= max_restarts:
            break

    # Final check
    if not is_valid(best_colors):
        return colors
        
    return best_colors
