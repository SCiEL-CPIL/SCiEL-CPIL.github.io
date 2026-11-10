"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    adj = [0] * n
    for a, b in P['edges']:
        adj[a] |= (1 << b)
        adj[b] |= (1 << a)
    
    # Precompute clique and independent set properties for all masks
    # is_clique[mask] and is_independent[mask]
    size = 1 << n
    is_clique = [False] * size
    is_independent = [False] * size
    is_clique[0] = True
    is_independent[0] = True
    
    for mask in range(1, size):
        # Find least significant set bit
        lsb = mask & -mask
        v = (lsb.bit_length() - 1)
        rest = mask ^ lsb
        
        # Check clique: v must be connected to all in rest, and rest must be clique
        is_clique[mask] = is_clique[rest] and ((adj[v] & rest) == rest)
        
        # Check independent set: v must not be connected to any in rest, and rest must be independent
        is_independent[mask] = is_independent[rest] and ((adj[v] & rest) == 0)
        
        # Early exit optimization not strictly necessary for n=14, but good practice
        # if not is_clique[mask] and not is_independent[mask]: pass # No change needed, just store
    
    is_valid = [is_clique[m] or is_independent[m] for m in range(size)]
    
    best_parts = []
    best_count = n + 1 # Upper bound
    
    # Greedy initialization to get a good initial upper bound
    greedy_parts = []
    unassigned = (1 << n) - 1
    while unassigned:
        # Pick a vertex, e.g., first available
        v = (unassigned & -unassigned).bit_length() - 1
        placed = False
        for i, part_mask in enumerate(greedy_parts):
            new_mask = part_mask | (1 << v)
            if is_valid[new_mask]:
                greedy_parts[i] = new_mask
                placed = True
                break
        if not placed:
            greedy_parts.append(1 << v)
        unassigned ^= (1 << v)
    
    best_count = len(greedy_parts)
    best_parts = greedy_parts[:]
    
    # Branch and Bound Search
    start_time = time.time()
    deadline = start_time + 50.0 # Safety margin
    
    # Heuristic for vertex selection: 
    # We want to pick a vertex that is "hard" to place.
    # A good proxy is the vertex with the highest degree in the remaining graph, 
    # or the one that has the fewest valid placements in current parts.
    # For simplicity and speed, let's just pick the lowest index vertex in unassigned.
    # A better heuristic: pick the vertex v in unassigned that minimizes the number of 
    # existing parts p such that is_valid[p | (1<<v)] is True.
    # If that number is 0, it forces a new part.
    
    def search(unassigned, current_parts, depth):
        nonlocal best_parts, best_count
        
        if time.time() > deadline:
            return False
        
        if not unassigned:
            if len(current_parts) < best_count:
                best_count = len(current_parts)
                best_parts = current_parts[:]
            return True
        
        # Pruning: if we already have more or equal parts than best, stop
        if len(current_parts) >= best_count:
            return False
        
        # Select vertex v from unassigned
        # Heuristic: choose v that has fewest valid existing parts to join
        # If no valid existing parts, it must create a new one.
        # This helps prune branches where we create too many new parts.
        
        best_v = -1
        min_valid_placements = n + 1
        
        # Iterate through bits in unassigned
        u = unassigned
        while u:
            lsb = u & -u
            v = lsb.bit_length() - 1
            valid_placements = 0
            for p in current_parts:
                if is_valid[p | lsb]:
                    valid_placements += 1
            
            if valid_placements < min_valid_placements:
                min_valid_placements = valid_placements
                best_v = v
                if min_valid_placements == 0:
                    break # Can't do better than 0
            u ^= lsb
        
        v = best_v
        v_bit = 1 << v
        new_unassigned = unassigned ^ v_bit
        
        # 1. Try adding v to existing parts
        for i, p in enumerate(current_parts):
            new_p = p | v_bit
            if is_valid[new_p]:
                current_parts[i] = new_p
                if search(new_unassigned, current_parts, depth + 1):
                    current_parts[i] = p
                    return True
                current_parts[i] = p
        
        # 2. Try creating a new part with v
        current_parts.append(v_bit)
        if search(new_unassigned, current_parts, depth + 1):
            current_parts.pop()
            return True
        current_parts.pop()
        
        return False
    
    # Start search
    # We can start with empty list of parts
    search((1 << n) - 1, [], 0)
    
    # Convert best_parts (bitmasks) to list of lists of ints
    result = []
    for mask in best_parts:
        part_vertices = []
        m = mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            part_vertices.append(v)
            m ^= lsb
        result.append(part_vertices)
    
    return result
# EVOLVE-BLOCK-END
