import time
import random
import math
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    n = 16
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    m = len(all_edges) # 120
    
    # Fast Q3 check using bitmasks
    def creates_q3(u, v, adj):
        mask_u = adj[u] & ~(1 << v)
        mask_v = adj[v] & ~(1 << u)
        
        mask_a = mask_u
        while mask_a:
            lsb_a = mask_a & -mask_a
            a = lsb_a.bit_length() - 1
            mask_a ^= lsb_a
            
            mask_b = mask_u & ~lsb_a
            while mask_b:
                lsb_b = mask_b & -mask_b
                b = lsb_b.bit_length() - 1
                mask_b ^= lsb_b
                
                mask_c = mask_v & adj[a]
                mask_d = mask_v & adj[b]
                
                if not mask_c or not mask_d:
                    continue
                
                mask_c_iter = mask_c
                while mask_c_iter:
                    lsb_c = mask_c_iter & -mask_c_iter
                    c = lsb_c.bit_length() - 1
                    mask_c_iter ^= lsb_c
                    
                    if c == a or c == b:
                        continue
                        
                    mask_d_iter = mask_d & ~lsb_c
                    while mask_d_iter:
                        lsb_d = mask_d_iter & -mask_d_iter
                        d = lsb_d.bit_length() - 1
                        mask_d_iter ^= lsb_d
                        
                        if d == a or d == b or d == c:
                            continue
                        
                        mask_e = (adj[a] & adj[b]) & ~((1 << u) | (1 << v) | (1 << a) | (1 << b) | (1 << c) | (1 << d))
                        if mask_e == 0:
                            continue
                            
                        mask_f = (adj[c] & adj[d]) & ~((1 << u) | (1 << v) | (1 << a) | (1 << b) | (1 << c) | (1 << d))
                        if mask_f == 0:
                            continue
                        
                        if mask_e.bit_count() < mask_f.bit_count():
                            mask_e_iter = mask_e
                            while mask_e_iter:
                                lsb_e = mask_e_iter & -mask_e_iter
                                e = lsb_e.bit_length() - 1
                                mask_e_iter ^= lsb_e
                                if mask_f & adj[e] & ~lsb_e:
                                    return True
                        else:
                            mask_f_iter = mask_f
                            while mask_f_iter:
                                lsb_f = mask_f_iter & -mask_f_iter
                                f = lsb_f.bit_length() - 1
                                mask_f_iter ^= lsb_f
                                if mask_e & adj[f] & ~lsb_f:
                                    return True
        return False

    def greedy_construct(edge_order, adj_init=None):
        if adj_init is not None:
            adj = adj_init[:]
        else:
            adj = [0] * n
        edges = []
        for u in range(n):
            for v in range(u+1, n):
                if (adj[u] >> v) & 1:
                    edges.append((u, v))
        
        for u, v in edge_order:
            if not ((adj[u] >> v) & 1):
                if not creates_q3(u, v, adj):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edges.append((u, v))
        return adj, edges

    def local_search(adj, edges, time_limit):
        best_count = len(edges)
        best_adj = adj[:]
        best_edges = edges[:]
        
        def get_missing_edges(adj):
            missing = []
            for u in range(n):
                for v in range(u+1, n):
                    if not (adj[u] >> v) & 1:
                        missing.append((u, v))
            return missing
        
        current_adj = adj[:]
        current_edges = set(edges)
        current_count = len(current_edges)
        
        while time.time() < time_limit:
            missing = get_missing_edges(current_adj)
            if not missing:
                break
                
            random.shuffle(missing)
            
            improved = False
            for u, v in missing:
                if time.time() >= time_limit:
                    break
                if not creates_q3(u, v, current_adj):
                    current_adj[u] |= (1 << v)
                    current_adj[v] |= (1 << u)
                    current_edges.add((u, v))
                    current_count += 1
                    improved = True
                    if current_count > best_count:
                        best_count = current_count
                        best_adj = current_adj[:]
                        best_edges = list(current_edges)
                        
            if not improved:
                if current_edges:
                    u, v = random.choice(list(current_edges))
                    current_adj[u] &= ~(1 << v)
                    current_adj[v] &= ~(1 << u)
                    current_edges.remove((u, v))
                    current_count -= 1
                    
                    missing = get_missing_edges(current_adj)
                    random.shuffle(missing)
                    for u2, v2 in missing:
                        if time.time() >= time_limit:
                            break
                        if not creates_q3(u2, v2, current_adj):
                            current_adj[u2] |= (1 << v2)
                            current_adj[v2] |= (1 << u2)
                            current_edges.add((u2, v2))
                            current_count += 1
                            if current_count > best_count:
                                best_count = current_count
                                best_adj = current_adj[:]
                                best_edges = list(current_edges)
                                
        return best_adj, best_edges

    # Vertex-partitioning strategy: 4 parts of 4 vertices
    random.seed(42)
    
    # Create 4 parts
    parts = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]
    
    # Start with complete 4-partite graph (all inter-part edges)
    adj = [0] * n
    for i in range(4):
        for j in range(i+1, 4):
            for u in parts[i]:
                for v in parts[j]:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
    
    # Collect current edges
    edges = []
    for u in range(n):
        for v in range(u+1, n):
            if (adj[u] >> v) & 1:
                edges.append((u, v))
    
    # Now try to add intra-part edges
    intra_edges = []
    for part in parts:
        for idx in range(len(part)):
            for jdx in range(idx+1, len(part)):
                intra_edges.append((part[idx], part[jdx]))
    
    # Randomize intra-part edge order for diversity
    random.shuffle(intra_edges)
    
    # Greedy add intra-part edges
    adj, edges = greedy_construct(intra_edges, adj_init=adj)
    
    # Also try inter-part edges that might have been missed (shouldn't be any in complete multipartite)
    # Run local search
    time_left = deadline - time.time()
    if time_left > 5.0:
        adj, edges = local_search(adj, edges, deadline - 5.0)
    
    # Convert to list of lists
    result = [[u, v] for u, v in edges]
    return result
