"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    start_time = time.time()
    deadline = start_time + 80.0  # 80s safety margin before 90s kill
    
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
        
        # Check if v is connected to all vertices in rest (for clique)
        if is_clique[rest]:
            # v must be adjacent to all in rest
            if (adj_mask[v] & rest) == rest:
                is_clique[mask] = True
        
        # Check if v is not connected to any vertex in rest (for independent set)
        if is_independent[rest]:
            # v must NOT be adjacent to any in rest
            if (adj_mask[v] & rest) == 0:
                is_independent[mask] = True
    
    # Precompute valid parts: for each vertex v, list of masks S where v is in S and S is valid
    # We only need masks where v is the lowest vertex (to avoid duplicates in DFS)
    valid_parts_for_vertex = [[] for _ in range(n)]
    for mask in range(1, 1 << n):
        if is_clique[mask] or is_independent[mask]:
            lsb = mask & -mask
            v = lsb.bit_length() - 1
            valid_parts_for_vertex[v].append(mask)
    
    # Sort by size descending to try larger parts first (fewer parts overall)
    for v in range(n):
        valid_parts_for_vertex[v].sort(key=lambda x: bin(x).count('1'), reverse=True)
    
    best_parts = n + 1  # Initialize to impossible
    best_partition = None
    
    # Memoization: (remaining_mask, depth) -> whether we can complete with <= best_parts
    # Use a set to track states we've already determined are infeasible
    infeasible_states = set()
    
    def dfs(remaining_mask, depth, current_parts):
        nonlocal best_parts, best_partition
        
        # Deadline check
        if time.time() > deadline:
            return
        
        # Pruning: if current depth already >= best, no point continuing
        if depth >= best_parts:
            return
        
        # Check if this state is known infeasible
        state_key = (remaining_mask, depth)
        if state_key in infeasible_states:
            return
        
        # Base case: all vertices assigned
        if remaining_mask == 0:
            if depth < best_parts:
                best_parts = depth
                best_partition = [list(p) for p in current_parts]
            return
        
        # Pick the lowest set bit vertex
        lsb = remaining_mask & -remaining_mask
        v = lsb.bit_length() - 1
        
        # Try all valid parts that contain v and are subsets of remaining_mask
        for S in valid_parts_for_vertex[v]:
            if (S & remaining_mask) != S:
                continue  # S is not a subset of remaining
            
            new_remaining = remaining_mask ^ S
            
            # Extract vertices from S for the part
            part_vertices = []
            m = S
            while m:
                lsb_s = m & -m
                u = lsb_s.bit_length() - 1
                part_vertices.append(u)
                m ^= lsb_s
            
            current_parts.append(part_vertices)
            dfs(new_remaining, depth + 1, current_parts)
            current_parts.pop()
            
            # If we found a solution with 1 part, that's optimal
            if best_parts == 1:
                return
        
        # Mark this state as infeasible if we exhausted all options without improving
        # Actually, we should only mark infeasible if depth + min_additional > best_parts
        # For simplicity, we'll skip memoization for now since n=14 is small
    
    dfs(full_mask, 0, [])
    
    # Fallback: if no solution found (shouldn't happen), return singletons
    if best_partition is None:
        best_partition = [[i] for i in range(n)]
    
    # Ensure all elements are Python int and lists are lists
    result = []
    for part in best_partition:
        result.append([int(v) for v in part])
    
    return result
# EVOLVE-BLOCK-END
