"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    import math
    from math import gcd
    
    N = 512
    start_time = time.time()
    deadline = start_time + 85
    
    # Build conflict graph
    adj = [0] * N
    for i in range(N):
        a = i + 1
        for j in range(i + 1, N):
            b = j + 1
            g = gcd(a, b)
            l = (a // g) * b
            if l > N:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    if time.time() > deadline:
        return [d for d in range(1, N+1) if 504 % d == 0]
    
    def popcount(x):
        return bin(x).count('1')
    
    # Greedy initial solution
    def greedy_is():
        candidates = (1 << N) - 1
        result = set()
        while candidates:
            # Pick node with min degree in candidates
            best_node = -1
            best_deg = N + 1
            c = candidates
            while c:
                lb = c & (-c)
                pos = lb.bit_length() - 1
                deg = popcount(adj[pos] & candidates)
                if deg < best_deg:
                    best_deg = deg
                    best_node = pos
                c ^= lb
            result.add(best_node)
            candidates &= ~(1 << best_node)
            candidates &= ~adj[best_node]
        return result
    
    initial = greedy_is()
    best_size = len(initial)
    best_set = set(initial)
    
    # B&B to find optimal
    # Node ordering by degree
    degrees = [popcount(adj[i]) for i in range(N)]
    order = sorted(range(N), key=lambda x: -degrees[x])
    pos_to_node = order
    node_to_pos = [0] * N
    for pos, node in enumerate(order):
        node_to_pos[node] = pos
    
    adj2 = [0] * N
    for pi in range(N):
        ni = order[pi]
        for pj in range(pi+1, N):
            nj = order[pj]
            if (adj[ni] >> nj) & 1:
                adj2[pi] |= (1 << pj)
                adj2[pj] |= (1 << pi)
    
    def bb(candidates, current_size, current_set):
        nonlocal best_size, best_set
        if time.time() > deadline:
            return
        if candidates == 0:
            if current_size > best_size:
                best_size = current_size
                best_set = set(current_set)
            return
        
        # Upper bound
        if current_size + popcount(candidates) <= best_size:
            return
        
        # Pick node with max degree in candidates
        # For speed, pick lowest set bit
        lb = candidates & (-candidates)
        pos = lb.bit_length() - 1
        
        # Include
        new_cand = candidates & ~(1 << pos) & ~adj2[pos]
        current_set.append(pos)
        bb(new_cand, current_size + 1, current_set)
        current_set.pop()
        
        # Exclude
        new_cand2 = candidates & ~(1 << pos)
        bb(new_cand2, current_size, current_set)
    
    all_nodes = (1 << N) - 1
    bb(all_nodes, 0, [])
    
    if best_size == 0:
        return [d for d in range(1, N+1) if 504 % d == 0]
    
    result = [pos_to_node[p] + 1 for p in best_set]
    return sorted(result)

# EVOLVE-BLOCK-END
