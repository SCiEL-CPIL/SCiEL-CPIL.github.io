import time
import random
from collections import Counter

def construct():
    start_time = time.time()
    time_limit = 110.0  # Safety margin

    n = 16
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_edges = len(edges)
    
    edge_index = {}
    for idx, (u, v) in enumerate(edges):
        edge_index[(u, v)] = idx
        edge_index[(v, u)] = idx
        
    k4_edges = []
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                for d in range(c + 1, n):
                    e_ab = edge_index[(a, b)]
                    e_ac = edge_index[(a, c)]
                    e_ad = edge_index[(a, d)]
                    e_bc = edge_index[(b, c)]
                    e_bd = edge_index[(b, d)]
                    e_cd = edge_index[(c, d)]
                    k4_edges.append((e_ab, e_ac, e_ad, e_bc, e_bd, e_cd))
                    
    num_k4s = len(k4_edges)
    
    # Precompute edge to k4 mapping
    edge_to_k4s = [[] for _ in range(num_edges)]
    for k4_idx, edge_list in enumerate(k4_edges):
        for e in edge_list:
            edge_to_k4s[e].append(k4_idx)
            
    # Initialize with seed
    colors = [(u + v) % 17 for u, v in edges]
    
    # Cache distinct counts? 
    # Since we modify colors, we need to update counts.
    # But for small backtracking, recalculating is easier and less error-prone.
    
    def get_distinct_colors(k4_idx):
        edge_list = k4_edges[k4_idx]
        # Unroll for speed?
        # c0 = colors[edge_list[0]]
        # ...
        # Using set is probably fine for 6 elements.
        return len({colors[e] for e in edge_list})
        
    def is_valid_k4(k4_idx):
        return get_distinct_colors(k4_idx) >= 5
        
    def check_validity_for_edge_change(e, new_color):
        # Check if changing edge e to new_color keeps all K4s valid
        # We assume the rest of the coloring is valid.
        # We only need to check K4s containing e.
        # But we don't know if the rest is valid in the middle of backtracking?
        # In backtracking, we maintain the invariant that all K4s not affected by the current path are valid.
        # So we only check affected K4s.
        for k4_idx in edge_to_k4s[e]:
            # Temporarily set color
            # We need to simulate
            # This is inefficient if we don't store old color.
            # Let's do it inside the backtrack function where we have control.
            pass
        return True

    # Better backtracking implementation
    def try_reduce_colors(current_colors):
        color_counts = Counter(current_colors)
        sorted_colors = sorted(color_counts.items(), key=lambda x: x[1])
        
        # Try to eliminate the least frequent color
        target_color, count = sorted_colors[0]
        
        # If count is too high, skip
        if count > 12:
            return None
            
        edges_to_recolor = [i for i, c in enumerate(current_colors) if c == target_color]
        other_colors = [c for c in color_counts if c != target_color]
        
        # Shuffle other colors for randomness
        random.shuffle(other_colors)
        
        def backtrack(idx):
            if idx == len(edges_to_recolor):
                return True
            
            e = edges_to_recolor[idx]
            old_c = current_colors[e]
            
            for new_c in other_colors:
                current_colors[e] = new_c
                
                # Check validity for affected K4s
                valid = True
                for k4_idx in edge_to_k4s[e]:
                    if get_distinct_colors(k4_idx) < 5:
                        valid = False
                        break
                        
                if valid:
                    if backtrack(idx + 1):
                        return True
                else:
                    current_colors[e] = old_c
            
            current_colors[e] = old_c
            return False
        
        if backtrack(0):
            return current_colors[:]
        
        return None

    def perturb_valid(colors_arr):
        # Try a few random swaps
        for _ in range(10):
            e1 = random.randint(0, num_edges - 1)
            e2 = random.randint(0, num_edges - 1)
            if e1 == e2:
                continue
                
            c1 = colors_arr[e1]
            c2 = colors_arr[e2]
            
            if c1 == c2:
                continue
                
            colors_arr[e1] = c2
            colors_arr[e2] = c1
            
            # Check validity
            affected = set(edge_to_k4s[e1])
            affected.update(edge_to_k4s[e2])
            
            valid = True
            for k4_idx in affected:
                if get_distinct_colors(k4_idx) < 5:
                    valid = False
                    break
            
            if valid:
                return colors_arr
            else:
                # Rollback
                colors_arr[e1] = c1
                colors_arr[e2] = c2
                
        return colors_arr

    best_colors = colors[:]
    best_num_colors = len(set(best_colors))
    
    current_colors = colors[:]
    current_num_colors = best_num_colors
    
    while time.time() - start_time < time_limit:
        # Try to reduce
        if current_num_colors > 13:
            new_colors = try_reduce_colors(current_colors)
            if new_colors is not None:
                current_colors = new_colors
                current_num_colors = len(set(current_colors))
                if current_num_colors < best_num_colors:
                    best_colors = current_colors[:]
                    best_num_colors = current_num_colors
                    
        # Perturb to escape local minima
        current_colors = perturb_valid(current_colors)
        
        # If we are stuck for a long time, restart from best?
        # Simple restart strategy: if no improvement in last X seconds, restart from seed or best.
        # We'll just keep going.
        
    return best_colors
