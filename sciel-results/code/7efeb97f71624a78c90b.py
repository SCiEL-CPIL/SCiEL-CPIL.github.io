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
    
    # Precompute popcount
    popcount = [0] * size
    for i in range(1, size):
        popcount[i] = popcount[i >> 1] + (i & 1)
    
    # Find max clique size and max independent set size
    max_omega = 0
    max_alpha = 0
    for mask in range(1, size):
        if is_clique[mask]:
            max_omega = max(max_omega, popcount[mask])
        if is_independent[mask]:
            max_alpha = max(max_alpha, popcount[mask])
    
    # Greedy to get initial upper bound
    def greedy(order):
        parts = []
        unassigned = (1 << n) - 1
        for v in order:
            v_bit = 1 << v
            if not (unassigned & v_bit):
                continue
            placed = False
            for i, pm in enumerate(parts):
                if is_valid[pm | v_bit]:
                    parts[i] = pm | v_bit
                    placed = True
                    break
            if not placed:
                parts.append(v_bit)
            unassigned ^= v_bit
        return parts
    
    # Try multiple orderings
    orders = [
        list(range(n)),
        sorted(range(n), key=lambda v: -popcount[adj[v]]),
        sorted(range(n), key=lambda v: popcount[adj[v]]),
        sorted(range(n), key=lambda v: -popcount[adj[v] & ((1 << n) - 1)]),
    ]
    
    best_greedy = min((greedy(o) for o in orders), key=len)
    best_count = len(best_greedy)
    best_parts = best_greedy[:]
    
    # Lower bound
    lower_bound = (n + max(max_omega, max_alpha) - 1) // max(max_omega, max_alpha)
    
    # Branch and bound search with iterative deepening
    start_time = time.time()
    deadline = start_time + 55.0
    
    failed_states = set()
    
    def search(unassigned, current_parts, part_count, max_parts):
        nonlocal best_parts, best_count
        
        if time.time() > deadline:
            return False
        
        if not unassigned:
            if part_count < best_count:
                best_count = part_count
                best_parts = current_parts[:]
            return True
        
        # Pruning: can't exceed max_parts
        if part_count >= max_parts:
            return False
        
        # Lower bound: need at least 1 more part if any remain
        if part_count + 1 > max_parts:
            return False
        
        # Memoization for small unassigned
        if popcount[unassigned] <= 8:
            key = (unassigned, tuple(sorted(current_parts)))
            if key in failed_states:
                return False
        
        # Select vertex: most constrained (fewest valid placements)
        best_v = -1
        min_placements = n + 1
        max_deg = -1
        
        u = unassigned
        while u:
            lsb = u & -u
            v = lsb.bit_length() - 1
            placements = 0
            for p in current_parts:
                if is_valid[p | lsb]:
                    placements += 1
            deg = popcount[adj[v] & unassigned]
            if placements < min_placements or (placements == min_placements and deg > max_deg):
                min_placements = placements
                max_deg = deg
                best_v = v
                if placements == 0:
                    break
            u ^= lsb
        
        v = best_v
        v_bit = 1 << v
        new_unassigned = unassigned ^ v_bit
        
        # Try adding to existing parts
        indices = sorted(range(len(current_parts)), key=lambda i: popcount[current_parts[i]])
        
        for i in indices:
            p = current_parts[i]
            if is_valid[p | v_bit]:
                current_parts[i] = p | v_bit
                if search(new_unassigned, current_parts, part_count, max_parts):
                    current_parts[i] = p
                    return True
                current_parts[i] = p
        
        # Try new part
        if part_count + 1 <= max_parts:
            current_parts.append(v_bit)
            if search(new_unassigned, current_parts, part_count + 1, max_parts):
                current_parts.pop()
                return True
            current_parts.pop()
        
        if popcount[unassigned] <= 8:
            key = (unassigned, tuple(sorted(current_parts)))
            failed_states.add(key)
        
        return False
    
    # Iterative deepening: try to find solution with k parts for k from lower_bound to best_count-1
    for k in range(lower_bound, best_count):
        if time.time() > deadline:
            break
        failed_states.clear()
        if search((1 << n) - 1, [], 0, k):
            if best_count <= k:
                break
    
    # Convert to list of lists
    result = []
    for mask in best_parts:
        part_vertices = []
        m = mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            part_vertices.append(int(v))
            m ^= lsb
        result.append(part_vertices)
    
    return result
# EVOLVE-BLOCK-END
