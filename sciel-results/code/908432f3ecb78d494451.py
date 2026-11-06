import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    # Precompute HC table
    hc_table = [0] * 32768
    for i in range(32768):
        adj = [[0]*6 for _ in range(6)]
        idx = 0
        for u in range(6):
            for v in range(u+1, 6):
                if (i >> idx) & 1:
                    adj[u][v] = 1
                    adj[v][u] = 1
                idx += 1
        # Check HC
        found = False
        for p in itertools.permutations(range(1, 6)):
            if adj[0][p[0]] and adj[p[0]][p[1]] and adj[p[1]][p[2]] and \
               adj[p[2]][p[3]] and adj[p[3]][p[4]] and adj[p[4]][0]:
                found = True
                break
        hc_table[i] = 1 if found else 0

    n = 17
    num_edges = n * (n - 1) // 2
    
    # Mapping
    edge_to_idx = {}
    idx_to_edge = []
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            edge_to_idx[(i,j)] = idx
            idx_to_edge.append((i,j))
            idx += 1
            
    # Precompute subset info
    subsets = list(itertools.combinations(range(n), 6))
    subset_info = []
    for S in subsets:
        # Map global vertex to local index
        v_to_local = {v: i for i, v in enumerate(S)}
        pairs = []
        bit_idx = 0
        for i in range(6):
            for j in range(i+1, 6):
                u = S[i]
                v = S[j]
                pairs.append(((u,v), bit_idx))
                bit_idx += 1
        # Store dict for fast lookup
        d = {}
        for (u,v), b in pairs:
            d[(u,v)] = b
            d[(v,u)] = b
        subset_info.append(d)
        
    def get_pattern(S, c, adj_bitsets):
        # S is tuple
        # adj_bitsets is list of lists of ints
        pattern = 0
        bit_idx = 0
        for i in range(6):
            for j in range(i+1, 6):
                u = S[i]
                v = S[j]
                if (adj_bitsets[c][u] >> v) & 1:
                    pattern |= (1 << bit_idx)
                bit_idx += 1
        return pattern

    # Initialize
    colors = [random.randint(0, 3) for _ in range(num_edges)]
    adj_bitsets = [[0]*n for _ in range(4)]
    for idx, (u,v) in enumerate(idx_to_edge):
        c = colors[idx]
        adj_bitsets[c][u] |= (1 << v)
        adj_bitsets[c][v] |= (1 << u)
        
    costs = [0] * 4
    for S in subsets:
        for c in range(4):
            p = get_pattern(S, c, adj_bitsets)
            if hc_table[p]:
                costs[c] += 1
                
    total_cost = sum(costs)
    best_cost = total_cost
    best_colors = colors[:]
    
    # Local Search
    while time.time() < deadline:
        if total_cost == 0:
            break
            
        idx = random.randint(0, num_edges - 1)
        u, v = idx_to_edge[idx]
        c_old = colors[idx]
        
        best_delta = 0
        best_c_new = c_old
        
        for c_new in range(4):
            if c_new == c_old:
                continue
                
            delta = 0
            for S, d in zip(subsets, subset_info):
                if u not in S or v not in S:
                    continue
                    
                bit_idx = d[(u,v)]
                mask = 1 << bit_idx
                
                # Get current patterns
                p_old = get_pattern(S, c_old, adj_bitsets)
                p_new = get_pattern(S, c_new, adj_bitsets)
                
                # p_old has edge (u,v) present.
                # p_new has edge (u,v) absent.
                
                # After move:
                # c_old loses edge: p_old & ~mask
                # c_new gains edge: p_new | mask
                
                old_hc_old = hc_table[p_old]
                new_hc_old = hc_table[p_old & ~mask]
                old_hc_new = hc_table[p_new]
                new_hc_new = hc_table[p_new | mask]
                
                delta += (new_hc_old - old_hc_old) + (new_hc_new - old_hc_new)
                
            if delta < best_delta:
                best_delta = delta
                best_c_new = c_new
                
        if best_delta < 0:
            # Apply move
            u, v = idx_to_edge[idx]
            adj_bitsets[c_old][u] &= ~(1 << v)
            adj_bitsets[c_old][v] &= ~(1 << u)
            adj_bitsets[best_c_new][u] |= (1 << v)
            adj_bitsets[best_c_new][v] |= (1 << u)
            
            colors[idx] = best_c_new
            total_cost += best_delta
            
            if total_cost < best_cost:
                best_cost = total_cost
                best_colors = colors[:]
                
    return {'n': n, 'colors': best_colors}
