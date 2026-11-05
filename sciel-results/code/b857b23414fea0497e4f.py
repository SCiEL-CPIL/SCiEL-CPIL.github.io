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
    base_col = np.array([(i + j) % 17 for i in range(n) for j in range(i + 1, n)])
    
    # Verify base is valid
    v = count_violations_vec(base_col)
    if v != 0:
        pass
        
    current_col = base_col.copy()
    current_colors = count_colors(current_col)
    
    best_col = current_col.copy()
    best_colors = current_colors
    
    random.seed(42)
    
    # Try to reduce colors by merging
    for target_colors in range(16, 12, -1):
        if time.time() > deadline - 10.0:
            break
            
        colors_in_use = np.unique(current_col)
        
        freqs = np.array([np.sum(current_col == c) for c in colors_in_use])
        sorted_indices = np.argsort(freqs)
        
        merged = False
        for idx in sorted_indices:
            if time.time() > deadline - 5.0:
                break
                
            c_to_remove = colors_in_use[idx]
            
            for c_target in colors_in_use:
                if c_target == c_to_remove:
                    continue
                
                new_col = current_col.copy()
                new_col[new_col == c_to_remove] = c_target
                
                v = count_violations_vec(new_col)
                if v == 0:
                    current_col = new_col
                    current_colors = count_colors(current_col)
                    if current_colors < best_colors:
                        best_colors = current_colors
                        best_col = current_col.copy()
                    merged = True
                    break
                    
            if merged:
                break
                
        if current_colors <= target_colors:
            continue
            
    # Simulated annealing to find better solutions
    if time.time() < deadline - 30.0:
        for max_k in range(15, 12, -1):
            if time.time() > deadline - 10.0:
                break
                
            found = False
            for restart in range(50):
                if time.time() > deadline - 10.0:
                    break
                    
                # Start from best known or random
                if restart == 0:
                    sa_col = best_col.copy()
                else:
                    sa_col = np.random.randint(0, max_k, size=num_edges)
                    
                sa_violations = count_violations_vec(sa_col)
                sa_colors = count_colors(sa_col)
                
                if sa_violations == 0 and sa_colors <= max_k:
                    if sa_colors < best_colors:
                        best_colors = sa_colors
                        best_col = sa_col.copy()
                        found = True
                        break
                    
                # Simulated annealing
                temp = 10.0
                cool_rate = 0.98
                for step in range(5000):
                    if time.time() > deadline - 10.0:
                        break
                        
                    if temp < 0.01:
                        break
                        
                    # Pick a random edge and change its color
                    edge_i = random.randint(0, num_edges - 1)
                    old_color = sa_col[edge_i]
                    new_color = random.randint(0, max_k - 1)
                    
                    if new_color == old_color:
                        continue
                        
                    sa_col[edge_i] = new_color
                    new_violations = count_violations_vec(sa_col)
                    new_colors = count_colors(sa_col)
                    
                    # Acceptance criterion
                    delta_v = new_violations - sa_violations
                    delta_c = new_colors - sa_colors
                    
                    if delta_v < 0 or (delta_v == 0 and delta_c < 0):
                        sa_violations = new_violations
                        sa_colors = new_colors
                        if sa_violations == 0 and sa_colors < best_colors:
                            best_colors = sa_colors
                            best_col = sa_col.copy()
                            found = True
                            break
                    else:
                        # Reject
                        sa_col[edge_i] = old_color
                        
                    temp *= cool_rate
                    
                if found:
                    break
                    
            if found:
                continue
                
    return best_col.tolist()
# EVOLVE-BLOCK-END
