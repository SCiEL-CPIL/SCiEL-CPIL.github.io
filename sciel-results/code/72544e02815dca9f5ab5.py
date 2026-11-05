import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    n = 16
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_edges = len(edges)  # 120
    
    # Precompute edge index map
    edge_idx_map = {e: i for i, e in enumerate(edges)}
    
    # Precompute K4s and their edge indices
    k4s = []
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
    
    k4s_np = np.array(k4s, dtype=np.int32)
    num_k4s = len(k4s_np)  # 1820
    
    # Precompute which K4s contain each edge
    edge_to_k4s = [[] for _ in range(num_edges)]
    for k4_idx in range(num_k4s):
        for e_idx in k4s_np[k4_idx]:
            edge_to_k4s[e_idx].append(k4_idx)
    
    edge_to_k4s_np = [np.array(lst, dtype=np.int32) for lst in edge_to_k4s]
    
    # Vectorized distinct count computation
    def compute_k4_distincts(col):
        """Compute distinct color counts for all K4s."""
        distincts = np.zeros(num_k4s, dtype=np.int32)
        for i in range(num_k4s):
            colors = col[k4s_np[i]]
            distincts[i] = len(set(colors.tolist()))
        return distincts
    
    def count_violations(col):
        """Count K4s with < 5 distinct colors."""
        distincts = compute_k4_distincts(col)
        return int(np.sum(distincts < 5))
    
    def try_move(col, e_idx, new_color, k4_distinct_cache):
        """
        Try changing col[e_idx] to new_color.
        Returns (delta_violations, old_color, affected_k4s)
        """
        old_color = col[e_idx]
        if old_color == new_color:
            return 0, old_color, []
        
        affected_k4s = edge_to_k4s_np[e_idx]
        delta_viol = 0
        
        # Compute old distinct counts for affected K4s
        old_distincts = {}
        for k4_idx in affected_k4s:
            old_distincts[k4_idx] = k4_distinct_cache[k4_idx]
        
        # Apply the move
        col[e_idx] = new_color
        
        # Compute new distinct counts and delta
        for k4_idx in affected_k4s:
            new_colors = col[k4s_np[k4_idx]]
            new_distinct = len(set(new_colors.tolist()))
            old_distinct = old_distincts[k4_idx]
            
            old_is_valid = old_distinct >= 5
            new_is_valid = new_distinct >= 5
            
            if old_is_valid and not new_is_valid:
                delta_viol += 1
            elif not old_is_valid and new_is_valid:
                delta_viol -= 1
            
            k4_distinct_cache[k4_idx] = new_distinct
        
        return delta_viol, old_color, affected_k4s
    
    def revert_move(col, e_idx, old_color, affected_k4s, k4_distinct_cache):
        """Revert a move."""
        col[e_idx] = old_color
        for k4_idx in affected_k4s:
            colors = col[k4s_np[k4_idx]]
            k4_distinct_cache[k4_idx] = len(set(colors.tolist()))
    
    def local_search(col, max_iters=50000, time_limit=10.0, max_colors=15):
        """Local search to fix violations."""
        k4_distinct_cache = compute_k4_distincts(col)
        current_violations = int(np.sum(k4_distinct_cache < 5))
        
        if current_violations == 0:
            return col, 0, k4_distinct_cache
        
        T = 1.0
        T_min = 0.01
        cooling = 0.9995
        
        start = time.time()
        for _ in range(max_iters):
            if time.time() - start > time_limit:
                break
            
            e_idx = random.randint(0, num_edges - 1)
            new_color = random.randint(0, max_colors - 1)
            
            delta_viol, old_color, affected = try_move(col, e_idx, new_color, k4_distinct_cache)
            
            if delta_viol <= 0:
                current_violations += delta_viol
            else:
                if random.random() < np.exp(-delta_viol / T):
                    current_violations += delta_viol
                else:
                    revert_move(col, e_idx, old_color, affected, k4_distinct_cache)
            
            T *= cooling
            if T < T_min:
                T = T_min
            
            if current_violations == 0:
                break
        
        return col, current_violations, k4_distinct_cache
    
    def eliminate_color(col, target_color, k4_distinct_cache, max_color=15):
        """Try to eliminate a specific color. Returns (success, new_col, new_cache)."""
        edges_to_change = list(np.where(col == target_color)[0])
        
        # Track changes for potential revert
        changes = []
        col_copy = col.copy()
        cache_copy = k4_distinct_cache.copy()
        
        for e_idx in edges_to_change:
            old_color = col_copy[e_idx]
            found_new_color = False
            
            # Try all other colors
            for new_color in range(max_color):
                if new_color == old_color or new_color == target_color:
                    continue
                
                delta_viol, old_c, affected = try_move(col_copy, e_idx, new_color, cache_copy)
                
                if delta_viol <= 0:
                    changes.append((e_idx, old_c, new_color))
                    found_new_color = True
                    break
                else:
                    revert_move(col_copy, e_idx, old_c, affected, cache_copy)
            
            if not found_new_color:
                # Revert all changes made so far in this elimination attempt
                for e, old_c, new_c in reversed(changes):
                    affected = edge_to_k4s_np[e]
                    col_copy[e] = old_c
                    for k4_idx in affected:
                        cache_copy[k4_idx] = len(set(col_copy[k4s_np[k4_idx]].tolist()))
                return False, None, None
        
        return True, col_copy, cache_copy
    
    def try_structured_init(n_colors):
        """Try structured initializations."""
        inits = []
        
        # (i+j) % prime
        for p in [13, 17, 19, 23]:
            if p >= n_colors:
                col = np.array([(i + j) % p for i in range(n) for j in range(i + 1, n)], dtype=np.int32)
                inits.append(col)
        
        # (i*j) % prime  
        for p in [13, 17, 19, 23]:
            if p >= n_colors:
                col = np.array([(i * j) % p for i in range(n) for j in range(i + 1, n)], dtype=np.int32)
                inits.append(col)
        
        # Random
        col = np.random.randint(0, n_colors, size=num_edges).astype(np.int32)
        inits.append(col)
        
        return inits
    
    # Strategy: Try to find valid solutions with decreasing number of colors
    best_col = None
    best_colors_used = 15
    
    # First, find any valid solution
    for attempt in range(10):
        if time.time() > deadline - 20:
            break
        
        # Try different initializations
        n_colors = 15 if attempt < 5 else 14
        inits = try_structured_init(n_colors)
        
        best_this_attempt = None
        best_violations = float('inf')
        
        for col_init in inits:
            if time.time() > deadline - 10:
                break
            
            col = col_init.copy()
            
            # Run local search
            col, violations, cache = local_search(col, max_iters=30000, time_limit=15.0, max_colors=n_colors)
            
            if violations < best_violations:
                best_violations = violations
                best_this_attempt = col.copy()
                best_cache = cache.copy()
            
            if violations == 0:
                colors_in_use = len(set(col.tolist()))
                if colors_in_use < best_colors_used:
                    best_col = col.copy()
                    best_colors_used = colors_in_use
                    best_cache = cache.copy()
                break
        
        if best_violations == 0 and best_col is not None:
            # Try to eliminate colors
            k4_distinct_cache = best_cache.copy() if 'best_cache' in dir() else compute_k4_distincts(best_col)
            
            while best_colors_used > 13 and time.time() < deadline - 5:
                colors_in_use = set(best_col.tolist())
                
                if len(colors_in_use) <= 13:
                    break
                
                # Pick color with fewest edges to eliminate
                color_counts = {}
                for c in colors_in_use:
                    color_counts[c] = int(np.sum(best_col == c))
                
                # Sort by count, try to eliminate smallest
                sorted_colors = sorted(color_counts.items(), key=lambda x: x[1])
                
                eliminated = False
                for target_color, count in sorted_colors:
                    if count == 0:
                        continue
                    
                    # Try to eliminate this color
                    success, new_col, new_cache = eliminate_color(best_col, target_color, k4_distinct_cache, max_color=best_colors_used)
                    
                    if success:
                        # Check if still valid
                        new_violations = count_violations(new_col)
                        if new_violations == 0:
                            best_col = new_col
                            k4_distinct_cache = new_cache
                            best_colors_used = len(set(best_col.tolist()))
                            eliminated = True
                            break
                    else:
                        # Failed, continue to next color
                        pass
                
                if not eliminated:
                    break
        
        if best_colors_used <= 13:
            break
    
    # If we didn't find a good solution, use a fallback
    if best_col is None:
        # Generate a valid coloring using (i+j) % 17
        seed_col = [(i + j) % 17 for i in range(n) for j in range(i + 1, n)]
        return seed_col
    
    return best_col.tolist()
