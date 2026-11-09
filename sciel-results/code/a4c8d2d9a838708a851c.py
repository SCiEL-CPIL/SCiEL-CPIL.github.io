import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency as bitmasks for fast subset checks
    adj_mask = [0] * n
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
    
    # Precompute: for each vertex v, the mask of vertices NOT adjacent to v (excluding v itself)
    non_adj_mask = [0] * n
    for v in range(n):
        non_adj_mask[v] = 0
        for u in range(n):
            if u != v and not (adj_mask[v] & (1 << u)):
                non_adj_mask[v] |= (1 << u)
    
    best_partition = None
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin
    
    # Try k from 1 to n
    for k in range(1, n + 1):
        if time.time() > deadline:
            break
            
        # Backtracking with bitmask-based parts
        # Each part: (vertex_mask, is_clique, is_ind)
        # vertex_mask: bitmask of vertices in this part
        
        def backtrack(remaining_mask, k_limit, parts):
            if remaining_mask == 0:
                nonlocal best_partition
                # Convert parts to list of lists of ints
                result = []
                for vmask, _, _ in parts:
                    verts = []
                    for v in range(n):
                        if vmask & (1 << v):
                            verts.append(v)
                    result.append(verts)
                best_partition = result
                return True
            
            if time.time() > deadline:
                return False
            
            # Pick vertex with highest "constraint" - prefer vertices that are
            # hard to place (many adjacencies or non-adjacencies to remaining)
            # Use a heuristic: pick vertex with max degree in remaining subgraph
            best_v = -1
            best_score = -1
            temp_mask = remaining_mask
            while temp_mask:
                lsb = temp_mask & (-temp_mask)
                v = lsb.bit_length() - 1
                temp_mask ^= lsb
                # Score: degree in remaining subgraph
                deg = bin(adj_mask[v] & remaining_mask).count('1')
                # Also consider: how many parts can this vertex join?
                # Fewer options = more constrained = should place earlier
                joinable = 0
                for pmask, is_cliq, is_ind in parts:
                    if is_cliq and (adj_mask[v] & pmask) == pmask:
                        joinable += 1
                    if is_ind and (non_adj_mask[v] & pmask) == pmask:
                        joinable += 1
                # Prefer high degree, low joinable count
                score = deg * 100 - joinable
                if score > best_score:
                    best_score = score
                    best_v = v
            
            if best_v == -1:
                return False
            
            v = best_v
            v_bit = 1 << v
            new_remaining = remaining_mask & ~v_bit
            
            # Try adding v to existing parts
            for i in range(len(parts)):
                pmask, is_cliq, is_ind = parts[i]
                
                new_is_cliq = is_cliq
                new_is_ind = is_ind
                
                if new_is_cliq:
                    # v must be connected to all in part
                    if (adj_mask[v] & pmask) != pmask:
                        new_is_cliq = False
                        
                if new_is_ind:
                    # v must not be connected to any in part
                    if (non_adj_mask[v] & pmask) != pmask:
                        new_is_ind = False
                        
                if new_is_cliq or new_is_ind:
                    parts[i] = (pmask | v_bit, new_is_cliq, new_is_ind)
                    
                    if backtrack(new_remaining, k_limit, parts):
                        return True
                    
                    # Undo
                    parts[i] = (pmask, is_cliq, is_ind)
            
            # Try adding v to a new part
            if len(parts) < k_limit:
                new_part = (v_bit, True, True)
                parts.append(new_part)
                
                if backtrack(new_remaining, k_limit, parts):
                    return True
                
                parts.pop()
            
            return False

        initial_parts = []
        remaining_mask = (1 << n) - 1
        
        if backtrack(remaining_mask, k, initial_parts):
            if best_partition:
                # Ensure all elements are Python int
                validated = []
                for part in best_partition:
                    validated.append([int(v) for v in part])
                return validated
        
    # Fallback: simple valid partition (each vertex its own part)
    return [[v] for v in range(n)]
