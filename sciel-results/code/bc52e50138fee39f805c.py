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
    size = 1 << n
    is_clique = [False] * size
    is_independent = [False] * size
    is_clique[0] = True
    is_independent[0] = True
    
    for mask in range(1, size):
        lsb = mask & -mask
        v = (lsb.bit_length() - 1)
        rest = mask ^ lsb
        
        is_clique[mask] = is_clique[rest] and ((adj[v] & rest) == rest)
        is_independent[mask] = is_independent[rest] and ((adj[v] & rest) == 0)
    
    is_valid = [is_clique[m] or is_independent[m] for m in range(size)]
    
    # Precompute popcount for faster counting
    popcount = [0] * size
    for i in range(1, size):
        popcount[i] = popcount[i >> 1] + (i & 1)
    
    best_parts = []
    best_count = n + 1
    
    # Greedy initialization: try multiple orderings to get a good upper bound
    def greedy_order(order):
        parts = []
        unassigned = (1 << n) - 1
        for v in order:
            v_bit = 1 << v
            if not (unassigned & v_bit):
                continue
            placed = False
            for i, part_mask in enumerate(parts):
                new_mask = part_mask | v_bit
                if is_valid[new_mask]:
                    parts[i] = new_mask
                    placed = True
                    break
            if not placed:
                parts.append(v_bit)
            unassigned ^= v_bit
        return parts
    
    # Try different vertex orderings for greedy
    orders = [
        list(range(n)),
        sorted(range(n), key=lambda v: -popcount[adj[v]]),
        sorted(range(n), key=lambda v: popcount[adj[v]]),
        [3, 5, 2, 0, 4, 12, 13, 8, 9, 11, 6, 10, 7, 1],
        [5, 3, 2, 4, 0, 12, 13, 8, 9, 11, 6, 10, 7, 1],
    ]
    
    best_greedy = None
    for order in orders:
        parts = greedy_order(order)
        if best_greedy is None or len(parts) < len(best_greedy):
            best_greedy = parts
    
    best_count = len(best_greedy)
    best_parts = best_greedy[:]
    
    # Branch and Bound Search with memoization
    start_time = time.time()
    deadline = start_time + 50.0
    
    failed_states = set()
    
    def search(unassigned, current_parts):
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
        
        # Memoization: if we've seen this state with >= parts before, skip
        # State: (unassigned, tuple of sorted part masks)
        # But this can be expensive, so only use when unassigned is small
        if popcount[unassigned] <= 8:
            parts_tuple = tuple(sorted(current_parts))
            state_key = (unassigned, parts_tuple)
            if state_key in failed_states:
                return False
        
        # Select vertex v from unassigned using most-constrained-first heuristic
        best_v = -1
        min_valid_placements = n + 1
        max_degree = -1
        
        u = unassigned
        while u:
            lsb = u & -u
            v = lsb.bit_length() - 1
            valid_placements = 0
            for p in current_parts:
                if is_valid[p | lsb]:
                    valid_placements += 1
            
            # Prefer vertices with fewer valid placements, then higher degree
            if valid_placements < min_valid_placements or \
               (valid_placements == min_valid_placements and popcount[adj[v] & unassigned] > max_degree):
                min_valid_placements = valid_placements
                max_degree = popcount[adj[v] & unassigned]
                best_v = v
                if min_valid_placements == 0:
                    break
            u ^= lsb
        
        v = best_v
        v_bit = 1 << v
        new_unassigned = unassigned ^ v_bit
        
        # 1. Try adding v to existing parts (in order of part size, smaller first)
        part_indices = list(range(len(current_parts)))
        part_indices.sort(key=lambda i: popcount[current_parts[i]])
        
        for i in part_indices:
            p = current_parts[i]
            new_p = p | v_bit
            if is_valid[new_p]:
                current_parts[i] = new_p
                if search(new_unassigned, current_parts):
                    current_parts[i] = p
                    return True
                current_parts[i] = p
        
        # 2. Try creating a new part with v
        current_parts.append(v_bit)
        if search(new_unassigned, current_parts):
            current_parts.pop()
            return True
        current_parts.pop()
        
        # Record failed state
        if popcount[unassigned] <= 8:
            parts_tuple = tuple(sorted(current_parts))
            state_key = (unassigned, parts_tuple)
            failed_states.add(state_key)
        
        return False
    
    # Start search
    search((1 << n) - 1, [])
    
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
