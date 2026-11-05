import time
import random
import math
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Leave 15s buffer for safety
    
    n = 16
    num_edges = n * (n - 1) // 2  # 120
    
    # Precompute edges
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    edge_index = {e: idx for idx, e in enumerate(edges)}
    
    # Precompute K4s
    k4s = list(combinations(range(n), 4))
    num_k4s = len(k4s)  # 1820
    
    # For each K4, find the indices of its 6 edges
    k4_edge_indices = []
    for k4 in k4s:
        vs = list(k4)
        e_indices = []
        for i in range(4):
            for j in range(i + 1, 4):
                e_indices.append(edge_index[(vs[i], vs[j])])
        k4_edge_indices.append(e_indices)
    
    # For each edge, find which K4s it belongs to
    edge_to_k4s = [[] for _ in range(num_edges)]
    for k4_idx, e_indices in enumerate(k4_edge_indices):
        for e_idx in e_indices:
            edge_to_k4s[e_idx].append(k4_idx)
    
    # Verify seed coloring (i+j)%17
    def get_seed_colors():
        return [(i + j) % 17 for i in range(n) for j in range(i + 1, n)]
    
    def count_violations_fast(colors):
        """Count K4s with fewer than 5 distinct colors."""
        viol = 0
        for e_indices in k4_edge_indices:
            if len(set(colors[idx] for idx in e_indices)) < 5:
                viol += 1
        return viol
    
    # Incremental local search with proper state tracking
    def incremental_local_search(max_colors, time_limit, init_colors=None):
        """
        Local search with incremental violation counting.
        Returns (colors, violations).
        """
        if init_colors is not None:
            colors = init_colors[:]
        else:
            colors = [random.randint(0, max_colors - 1) for _ in range(num_edges)]
        
        # Build state: for each K4, track color counts and distinct count
        k4_counts = []
        k4_distinct = []
        total_viol = 0
        
        for e_indices in k4_edge_indices:
            counts = [0] * max_colors
            for idx in e_indices:
                counts[colors[idx]] += 1
            distinct = sum(1 for c in counts if c > 0)
            k4_counts.append(counts)
            k4_distinct.append(distinct)
            if distinct < 5:
                total_viol += 1
        
        best_viol = total_viol
        best_colors = colors[:]
        
        temp = 1.0
        min_temp = 0.005
        cool_rate = 0.998
        
        while time.time() < start_time + time_limit:
            if total_viol == 0:
                return colors, 0
            
            e_idx = random.randint(0, num_edges - 1)
            old_c = colors[e_idx]
            new_c = random.randint(0, max_colors - 1)
            
            if old_c == new_c:
                continue
            
            # Compute delta in violations
            delta_viol = 0
            affected = edge_to_k4s[e_idx]
            
            # Store old distinct values for potential revert
            old_distincts = []
            
            for k4_idx in affected:
                counts = k4_counts[k4_idx]
                old_d = k4_distinct[k4_idx]
                old_distincts.append(old_d)
                
                c_old = counts[old_c]
                c_new = counts[new_c]
                
                new_d = old_d
                if c_old == 1:  # color disappears
                    new_d -= 1
                if c_new == 0:  # color appears
                    new_d += 1
                
                was_viol = 1 if old_d < 5 else 0
                is_viol = 1 if new_d < 5 else 0
                delta_viol += is_viol - was_viol
                
                # Apply tentatively
                counts[old_c] = c_old - 1
                counts[new_c] = c_new + 1
                k4_distinct[k4_idx] = new_d
            
            # Acceptance criterion
            if delta_viol <= 0 or (delta_viol > 0 and random.random() < math.exp(-delta_viol / max(temp, 1e-6))):
                colors[e_idx] = new_c
                total_viol += delta_viol
                if total_viol < best_viol:
                    best_viol = total_viol
                    best_colors = colors[:]
            else:
                # Revert
                for k4_idx, old_d in zip(affected, old_distincts):
                    counts = k4_counts[k4_idx]
                    counts[old_c] += 1
                    counts[new_c] -= 1
                    k4_distinct[k4_idx] = old_d
            
            temp *= cool_rate
            if temp < min_temp:
                temp = min_temp
        
        return best_colors, best_viol
    
    # Try to find solutions with 13, 14, 15 colors
    # Strategy: try multiple random restarts for each color count
    
    # First, verify the seed is valid
    seed_colors = get_seed_colors()
    seed_viol = count_violations_fast(seed_colors)
    
    # Try C=13 with several restarts
    for attempt in range(5):
        if time.time() > deadline - 20:
            break
        t_left = min(20.0, deadline - 20.0 - time.time())
        if t_left <= 0:
            break
        sol, viol = incremental_local_search(13, t_left)
        if viol == 0:
            # Verify
            if count_violations_fast(sol) == 0:
                return sol
    
    # Try C=14 with several restarts
    for attempt in range(5):
        if time.time() > deadline - 15:
            break
        t_left = min(15.0, deadline - 15.0 - time.time())
        if t_left <= 0:
            break
        sol, viol = incremental_local_search(14, t_left)
        if viol == 0:
            if count_violations_fast(sol) == 0:
                return sol
    
    # Try C=15 with several restarts
    for attempt in range(5):
        if time.time() > deadline - 10:
            break
        t_left = min(10.0, deadline - 10.0 - time.time())
        if t_left <= 0:
            break
        sol, viol = incremental_local_search(15, t_left)
        if viol == 0:
            if count_violations_fast(sol) == 0:
                return sol
    
    # If we couldn't find a solution with 13-15 colors, try to reduce the seed's colors
    # Start from seed and try to reduce colors used
    if seed_viol == 0:
        # Try to find a valid coloring with fewer colors starting from seed
        # We can map colors to a smaller palette
        colors_used = sorted(set(seed_colors))
        color_map = {c: i for i, c in enumerate(colors_used)}
        mapped = [color_map[c] for c in seed_colors]
        num_mapped = len(colors_used)  # Should be 17
        
        # Try to reduce from 17 down
        for target_c in range(15, 12, -1):
            if time.time() > deadline - 5:
                break
            t_left = min(5.0, deadline - 5.0 - time.time())
            if t_left <= 0:
                break
            # Start from a random permutation of the seed mapped to target_c colors
            # But we need to ensure validity. Let's just do local search from seed.
            sol, viol = incremental_local_search(target_c, t_left, init_colors=mapped)
            if viol == 0:
                if count_violations_fast(sol) == 0:
                    return sol
    
    # Fallback: return the seed coloring (17 colors, valid)
    return seed_colors
