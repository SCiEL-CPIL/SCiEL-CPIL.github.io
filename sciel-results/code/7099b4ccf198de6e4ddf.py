import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    n = 16
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    edge_index = {(u, v): i for i, (u, v) in enumerate(all_edges)}
    m = len(all_edges) # 120
    
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

    best_edges = []
    best_count = 0
    
    def greedy_construct():
        adj = [0] * n
        edges = []
        shuffled_edges = all_edges[:]
        random.shuffle(shuffled_edges)
        
        for u, v in shuffled_edges:
            if not creates_q3(u, v, adj):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges.append((u, v))
        return adj, edges

    def local_search_add(adj, edges):
        improved = True
        while improved:
            improved = False
            missing_edges = [e for e in all_edges if not ((adj[e[0]] >> e[1]) & 1)]
            random.shuffle(missing_edges)
            
            for u, v in missing_edges:
                if not creates_q3(u, v, adj):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edges.append((u, v))
                    improved = True
        return adj, edges

    def kick_and_fill(adj, edges, kick_size):
        # Make a copy to revert if needed
        orig_adj = adj[:]
        orig_edges = edges[:]
        
        # Remove kick_size random edges
        current_adj = adj[:]
        current_edges = edges[:]
        
        for _ in range(kick_size):
            if not current_edges:
                break
            idx = random.randint(0, len(current_edges) - 1)
            u, v = current_edges[idx]
            current_adj[u] &= ~(1 << v)
            current_adj[v] &= ~(1 << u)
            current_edges.pop(idx)
            
        # Run local search add
        current_adj, current_edges = local_search_add(current_adj, current_edges)
        
        return current_adj, current_edges

    while time.time() < deadline:
        adj, edges = greedy_construct()
        adj, edges = local_search_add(adj, edges)
        
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges[:]
            
        # Try kicks
        if time.time() < deadline - 5:
            # Try different kick sizes
            for kick_size in [2, 3, 5, 8]:
                if time.time() > deadline - 5:
                    break
                new_adj, new_edges = kick_and_fill(adj, edges, kick_size)
                if len(new_edges) > best_count:
                    best_count = len(new_edges)
                    best_edges = new_edges[:]
                    # Update the current best for further kicks?
                    # It's safer to kick from the original local optimum
                    # or the new one? Kicking from the new one might be good.
                    adj, edges = new_adj, new_edges

    return [[u, v] for u, v in best_edges]
