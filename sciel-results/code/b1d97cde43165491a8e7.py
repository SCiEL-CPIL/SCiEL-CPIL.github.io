import time
import random
from itertools import combinations
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
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
        """Check if every K4 has at least 5 distinct colors using numpy."""
        col_arr = np.array(col, dtype=np.int32)
        # Get all 6 colors for each K4
        k4_colors = col_arr[k4_edge_indices_np]  # shape: (1820, 6)
        
        # For each K4, count distinct colors
        # A K4 is valid if it has at least 5 distinct colors
        # We can check: for each K4, the number of distinct colors >= 5
        
        # Efficient way: for each K4, check if any color appears 3+ times
        # If a color appears 3+ times, then max distinct = 4 (since 6 edges, 3 of one color leaves 3 for at most 3 other colors)
        # Actually, we need to be more careful.
        
        # Let's use a different approach: for each K4, compute the number of distinct colors
        # We can do this by checking all pairs
        
        # Simpler: just check if len(set) >= 5 for each K4
        # But that's slow in pure Python. Let's vectorize differently.
        
        # Alternative: A K4 is invalid if it has <= 4 distinct colors.
        # This happens if:
        # - All 6 edges same color
        # - 5+ edges same color (so 1 or 2 distinct)
        # - 4+ edges same color (so 2 or 3 distinct)
        # - 3+ edges same color AND another 3+ edges same color (so 2 distinct)
        # - Various other patterns with 3 or 4 distinct colors
        
        # Let's just use the set approach but optimized
        for k in range(len(k4s)):
            colors_in_k4 = k4_colors[k]
            if len(set(colors_in_k4.tolist())) < 5:
                return False
        return True
    
    def is_valid(col):
        """Check if every K4 has at least 5 distinct colors."""
        return is_valid_fast(col)
    
    # Initial coloring: (i + j) % 17
    colors = [0] * 120
    for idx, (i, j) in enumerate(edges):
        colors[idx] = (i + j) % 17
        
    # Current set of used colors
    used_colors = set(colors)
    
    # Verify initial
    if not is_valid(colors):
        # Should not happen, but fallback
        return colors
    
    best_colors = colors[:]
    best_count = len(used_colors)
    
    # Merging loop with restarts
    max_restarts = 50
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
                num_perturb = random.randint(1, 3)
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
