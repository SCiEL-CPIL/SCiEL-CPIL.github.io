"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin
    
    n = 16
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_edges = len(edges)  # 120
    
    # Precompute all K4s and their edge indices
    k4s = []
    edge_idx_map = {e: i for i, e in enumerate(edges)}
    for quad in combinations(range(n), 4):
        v0, v1, v2, v3 = quad
        e_indices = [
            edge_idx_map[(v0, v1)],
            edge_idx_map[(v0, v2)],
            edge_idx_map[(v0, v3)],
            edge_idx_map[(v1, v2)],
            edge_idx_map[(v1, v3)],
            edge_idx_map[(v2, v3)]
        ]
        k4s.append(e_indices)
    
    num_k4s = len(k4s)  # 1820
    
    # Precompute which K4s contain each edge
    edge_to_k4s = [[] for _ in range(num_edges)]
    for k4_idx in range(num_k4s):
        for e_idx in k4s[k4_idx]:
            edge_to_k4s[e_idx].append(k4_idx)
    
    # Fallback solution: (i+j) % 17
    seed_col = [(i + j) % 17 for i in range(n) for j in range(i + 1, n)]
    
    def count_colors(col):
        return len(set(col))
    
    def is_valid(col):
        for k in range(num_k4s):
            if len(set(col[e] for e in k4s[k])) < 5:
                return False
        return True

    best_result = None
    best_color_count = float('inf')
    
    # Check seed
    if is_valid(seed_col):
        best_result = seed_col
        best_color_count = count_colors(seed_col)
        
    # Max colors we might use for state tracking
    MAX_COLORS = 20
    
    # Fast local search with optimized state tracking
    def fast_local_search(initial_col, max_colors, max_iters, time_limit):
        col = initial_col[:]
        # k4_color_counts[k][c] = count of color c in k4 k
        k4_color_counts = [[0] * MAX_COLORS for _ in range(num_k4s)]
        k4_distinct = [0] * num_k4s
        current_violations = 0
        
        # Initialize state
        for k in range(num_k4s):
            counts = [0] * MAX_COLORS
            for e in k4s[k]:
                c = col[e]
                if c < MAX_COLORS:
                    counts[c] += 1
            k4_color_counts[k] = counts
            d = sum(1 for x in counts if x > 0)
            k4_distinct[k] = d
            if d < 5:
                current_violations += 1
                
        if current_violations == 0:
            return col, count_colors(col)
            
        T = 1.0
        T_min = 0.01
        cooling = 0.999
        
        for it in range(max_iters):
            if time.time() > time_limit:
                break
                
            e_idx = random.randint(0, num_edges - 1)
            old_color = col[e_idx]
            new_color = random.randint(0, max_colors - 1)
            
            if old_color == new_color:
                continue
            
            if old_color >= MAX_COLORS or new_color >= MAX_COLORS:
                # Should not happen if max_colors <= MAX_COLORS
                continue
                
            delta_viol = 0
            affected_k4s = edge_to_k4s[e_idx]
            
            # Apply change
            col[e_idx] = new_color
            
            for k in affected_k4s:
                counts = k4_color_counts[k]
                old_d = k4_distinct[k]
                
                # Update old color
                counts[old_color] -= 1
                if counts[old_color] == 0:
                    # Distinct count decreases
                    pass # We'll compute new_d at the end
                    
                # Update new color
                counts[new_color] += 1
                
                # Recalculate distinct count for this k4
                new_d = sum(1 for x in counts if x > 0)
                k4_distinct[k] = new_d
                
                if old_d < 5 and new_d >= 5:
                    delta_viol -= 1
                elif old_d >= 5 and new_d < 5:
                    delta_viol += 1
                    
            if delta_viol <= 0:
                current_violations += delta_viol
            else:
                if random.random() < np.exp(-delta_viol / T):
                    current_violations += delta_viol
                else:
                    # Revert
                    col[e_idx] = old_color
                    for k in affected_k4s:
                        counts = k4_color_counts[k]
                        old_d = k4_distinct[k] # This is the new_d from the forward pass
                        
                        # Revert counts
                        counts[new_color] -= 1
                        counts[old_color] += 1
                        
                        # Recalculate distinct
                        new_d = sum(1 for x in counts if x > 0)
                        k4_distinct[k] = new_d
                        
                        # Note: current_violations was not updated, so we don't need to adjust it here
                        # The state is back to before the move.
                        
            T *= cooling
            if T < T_min:
                T = T_min
                
            if current_violations == 0:
                return col, count_colors(col)
                
        return None, float('inf')

    # Try to find better solutions
    for target_colors in [13, 14, 15]:
        if time.time() > deadline - 20.0:
            break
            
        # Multiple restarts
        for restart in range(100):
            if time.time() > deadline - 10.0:
                break
                
            # Random initialization
            col = [random.randint(0, target_colors - 1) for _ in range(num_edges)]
            
            # Run fast local search
            result_col, c_count = fast_local_search(col, target_colors, 5000, deadline - 5.0)
            
            if result_col is not None:
                if c_count < best_color_count:
                    best_color_count = c_count
                    best_result = result_col
                if c_count <= 13:
                    return best_result

    # Polish: Try to reduce best_result if it has > 13 colors
    if best_result is not None and best_color_count > 13:
        # Try to eliminate colors from best_result
        # We can use the fast local search with a constraint that we try to reduce color count?
        # Or just try to recolor edges to reduce the set of colors used.
        
        # A simple approach: Try to recolor edges to use fewer colors.
        # This is tricky. Let's just return the best found.
        pass

    if best_result is not None:
        return best_result
    
    return seed_col
# EVOLVE-BLOCK-END
