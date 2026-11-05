import time
import random
import numpy as np
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n = 16
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_edges = len(edges)
    
    edge_idx = {}
    for idx, (a, b) in enumerate(edges):
        edge_idx[(a, b)] = idx
        
    k4_edges_list = []
    for quad in combinations(range(n), 4):
        v0, v1, v2, v3 = quad
        e_indices = [
            edge_idx[(v0, v1)],
            edge_idx[(v0, v2)],
            edge_idx[(v0, v3)],
            edge_idx[(v1, v2)],
            edge_idx[(v1, v3)],
            edge_idx[(v2, v3)],
        ]
        k4_edges_list.append(e_indices)
        
    num_k4s = len(k4_edges_list)
    k4_edges = np.array(k4_edges_list)
    
    pair_indices = np.array([(i, j) for i in range(6) for j in range(i+1, 6)])
    
    def count_violations_vec(col_np):
        k4_colors = col_np[k4_edges]
        c1 = k4_colors[:, pair_indices[:, 0]]
        c2 = k4_colors[:, pair_indices[:, 1]]
        collisions = (c1 == c2)
        num_collisions = np.sum(collisions, axis=1)
        violations = np.sum(num_collisions >= 2)
        return violations
        
    def count_colors(col_np):
        return len(np.unique(col_np))
        
    # Base coloring: (i+j) % 17
    # This is valid and uses 17 colors
    base_col = np.array([(i + j) % 17 for i in range(n) for j in range(i + 1, n)])
    
    # Verify base is valid
    v = count_violations_vec(base_col)
    if v != 0:
        # Fallback if base is invalid (should not happen)
        # Use random search
        pass
        
    current_col = base_col.copy()
    current_colors = count_colors(current_col) # 17
    
    best_col = current_col.copy()
    best_colors = current_colors
    
    random.seed(42)
    
    # Try to reduce colors by merging
    # We have 17 colors. We want to get to 15.
    # We can try to merge colors one by one.
    
    for target_colors in range(16, 12, -1):
        if time.time() > deadline - 10.0:
            break
            
        # Shuffle colors to try different merging orders
        # Actually, just try to merge the rarest color into others
        
        colors_in_use = np.unique(current_col)
        
        # Sort colors by frequency (ascending)
        freqs = np.array([np.sum(current_col == c) for c in colors_in_use])
        sorted_indices = np.argsort(freqs)
        
        merged = False
        for idx in sorted_indices:
            if time.time() > deadline - 5.0:
                break
                
            c_to_remove = colors_in_use[idx]
            
            # Try to merge c_to_remove into each other color
            for c_target in colors_in_use:
                if c_target == c_to_remove:
                    continue
                
                # Create new coloring
                new_col = current_col.copy()
                new_col[new_col == c_to_remove] = c_target
                
                v = count_violations_vec(new_col)
                if v == 0:
                    # Valid merge
                    current_col = new_col
                    current_colors = count_colors(current_col)
                    if current_colors < best_colors:
                        best_colors = current_colors
                        best_col = current_col.copy()
                    merged = True
                    break # Restart merging process with new state
                    
            if merged:
                break # Restart the outer loop iteration for this target? 
                # Actually, just continue the main loop.
                
        if current_colors <= target_colors:
            continue
            
    return best_col.tolist()
# EVOLVE-BLOCK-END
