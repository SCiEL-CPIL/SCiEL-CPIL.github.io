"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    start_time = time.time()
    deadline = start_time + 70.0  # 70s safety margin before 90s kill
    
    # Build adjacency bitmask
    adj_mask = [0] * n
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
    
    # Precompute is_clique and is_independent for all masks
    full_mask = (1 << n) - 1
    is_clique = [False] * (1 << n)
    is_independent = [False] * (1 << n)
    is_clique[0] = True
    is_independent[0] = True
    
    for mask in range(1, 1 << n):
        lsb = mask & -mask
        v = lsb.bit_length() - 1
        rest = mask ^ lsb
        
        if is_clique[rest]:
            if (adj_mask[v] & rest) == rest:
                is_clique[mask] = True
        
        if is_independent[rest]:
            if (adj_mask[v] & rest) == 0:
                is_independent[mask] = True
    
    # Precompute valid parts with their vertex lists
    # For each vertex v, list of (mask, vertex_list) where v is in mask and mask is valid
    valid_parts_for_vertex = [[] for _ in range(n)]
    for mask in range(1, 1 << n):
        if is_clique[mask] or is_independent[mask]:
            lsb = mask & -mask
            v = lsb.bit_length() - 1
            # Extract vertices
            part_vertices = []
            m = mask
            while m:
                lsb_s = m & -m
                u = lsb_s.bit_length() - 1
                part_vertices.append(u)
                m ^= lsb_s
            valid_parts_for_vertex[v].append((mask, part_vertices))
    
    # Sort by size descending to try larger parts first
    for v in range(n):
        valid_parts_for_vertex[v].sort(key=lambda x: len(x[1]), reverse=True)
    
    # Count of valid parts per vertex for heuristic ordering
    part_counts = [len(valid_parts_for_vertex[v]) for v in range(n)]
    
    best_parts = n + 1
    best_partition = None
    infeasible_states = set()
    
    def count_bits(mask):
        c = 0
        while mask:
            mask &= mask - 1
            c += 1
        return c
    
    def dfs(remaining_mask, depth, current_parts):
        nonlocal best_parts, best_partition
        
        if time.time() > deadline:
            return
        
        if depth >= best_parts:
            return
        
        state_key = (remaining_mask, depth)
        if state_key in infeasible_states:
            return
        
        if remaining_mask == 0:
            if depth < best_parts:
                best_parts = depth
                best_partition = [list(p) for p in current_parts]
            return
        
        # Heuristic: pick the vertex with fewest valid part options among remaining vertices
        # Find all vertices in remaining_mask
        v = -1
        min_count = n + 1
        m = remaining_mask
        while m:
            lsb = m & -m
            u = lsb.bit_length() - 1
            # Count how many valid parts for u are subsets of remaining_mask
            cnt = 0
            for mask, verts in valid_parts_for_vertex[u]:
                if (mask & remaining_mask) == mask:
                    cnt += 1
                    if cnt >= min_count:
                        break
            if cnt < min_count:
                min_count = cnt
                v = u
                if cnt == 1:
                    break
            m ^= lsb
        
        # Try all valid parts that contain v and are subsets of remaining_mask
        for S, part_vertices in valid_parts_for_vertex[v]:
            if (S & remaining_mask) != S:
                continue
            
            new_remaining = remaining_mask ^ S
            
            current_parts.append(part_vertices)
            dfs(new_remaining, depth + 1, current_parts)
            current_parts.pop()
            
            if best_parts == 1:
                return
        
        infeasible_states.add(state_key)
    
    dfs(full_mask, 0, [])
    
    if best_partition is None:
        best_partition = [[i] for i in range(n)]
    
    result = []
    for part in best_partition:
        result.append([int(v) for v in part])
    
    return result
# EVOLVE-BLOCK-END
