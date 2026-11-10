"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill
    
    n = 16
    num_edges = n * (n - 1) // 2  # 120 edges
    
    # Precompute edge index mapping
    idx_map = {}
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            idx_map[(i, j)] = idx
            idx += 1
    
    # Precompute all K4s and their edge indices
    k4_edges = []
    edge_to_k4 = [[] for _ in range(num_edges)]
    for v in itertools.combinations(range(n), 4):
        a, b, c, d = v
        edges = [(a,b), (a,c), (a,d), (b,c), (b,d), (c,d)]
        e_indices = []
        for e in edges:
            i, j = e
            if i > j:
                i, j = j, i
            ei = idx_map[(i, j)]
            e_indices.append(ei)
            edge_to_k4[ei].append(len(k4_edges))
        k4_edges.append(e_indices)
    
    num_k4s = len(k4_edges)
    
    def count_violations(colors):
        v = 0
        for e_indices in k4_edges:
            cs = set()
            for ei in e_indices:
                cs.add(colors[ei])
            if len(cs) < 5:
                v += 1
        return v
    
    # Base solution: (i+j) mod 17, uses 17 colours
    base_colors = [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
    best_colors = base_colors[:]
    best_num = len(set(best_colors))  # 17
    
    if count_violations(best_colors) > 0:
        return best_colors  # Should not happen
    
    # Local search: try to reduce number of colours
    # Strategy: iteratively pick a colour to eliminate, reassign its edges
    # to other colours while maintaining validity
    
    def try_eliminate_colour(cur_colors, target_colour, time_limit):
        """Try to eliminate target_colour by reassigning its edges."""
        if target_colour not in set(cur_colors):
            return None
        
        cur = cur_colors[:]
        # Find all edges with target_colour
        target_edges = [ei for ei in range(num_edges) if cur[ei] == target_colour]
        
        # Get available other colours
        other_colours = [c for c in set(cur) if c != target_colour]
        if not other_colours:
            return None
        
        # For each edge with target_colour, try to reassign
        success = True
        for ei in target_edges:
            orig = cur[ei]
            # Try each other colour
            found = False
            for new_c in other_colours:
                cur[ei] = new_c
                if count_violations(cur) == 0:
                    found = True
                    break
            if not found:
                cur[ei] = orig
                success = False
                break
        
        if success:
            # Verify
            if count_violations(cur) == 0:
                return cur
        return None
    
    # Iteratively try to eliminate colours
    current = best_colors[:]
    for iteration in range(50):
        if time.time() > deadline:
            break
        
        cur_cols = sorted(set(current))
        if len(cur_cols) <= best_num:
            break
        
        # Try to eliminate each colour (start from highest to reduce palette)
        eliminated = False
        for c in reversed(cur_cols):
            if time.time() > deadline:
                break
            result = try_eliminate_colour(current, c, deadline)
            if result is not None:
                current = result
                eliminated = True
                break
        
        if not eliminated:
            # Try random local search moves
            for _ in range(5000):
                if time.time() > deadline:
                    break
                
                # Pick random edge and random colour
                ei = random.randint(0, num_edges - 1)
                new_c = random.randint(0, 16)
                if new_c == current[ei]:
                    continue
                
                orig = current[ei]
                current[ei] = new_c
                viol = count_violations(current)
                
                if viol == 0:
                    # Check if we reduced palette
                    new_num = len(set(current))
                    if new_num < best_num:
                        best_colors = current[:]
                        best_num = new_num
                else:
                    current[ei] = orig
    
    # Final check
    if count_violations(best_colors) != 0:
        best_colors = base_colors[:]
        best_num = 17
    
    # Renormalize colours to 0..best_num-1
    col_map = {}
    for i, c in enumerate(sorted(set(best_colors))):
        col_map[c] = i
    best_colors = [col_map[c] for c in best_colors]
    
    return best_colors
# EVOLVE-BLOCK-END
