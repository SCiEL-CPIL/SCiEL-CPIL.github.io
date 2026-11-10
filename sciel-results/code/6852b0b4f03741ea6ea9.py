import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    n = 16
    adj = [0] * n
    current_edges = []
    
    # Start with K16
    for u in range(n):
        for v in range(u + 1, n):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            current_edges.append((u, v))
            
    best_count = len(current_edges)
    best_edges = current_edges.copy()
    
    def creates_q3(u, v, adj):
        nu = [x for x in range(n) if x != u and x != v and (adj[u] >> x) & 1]
        nv = [x for x in range(n) if x != v and x != u and (adj[v] >> x) & 1]
        
        for i in range(len(nu)):
            a = nu[i]
            for j in range(i + 1, len(nu)):
                b = nu[j]
                mask_a = adj[a]
                mask_b = adj[b]
                
                for k in range(len(nv)):
                    c = nv[k]
                    for l in range(k + 1, len(nv)):
                        d = nv[l]
                        
                        if a == c or a == d or b == c or b == d:
                            continue
                        
                        mask_c = adj[c]
                        mask_d = adj[d]
                        
                        if (mask_a >> c) & 1 and (mask_b >> d) & 1:
                            used = (1 << u) | (1 << v) | (1 << a) | (1 << b) | (1 << c) | (1 << d)
                            e_cand = mask_a & mask_b & ~used
                            f_cand = mask_c & mask_d & ~used
                            
                            e_mask = e_cand
                            while e_mask:
                                lsb = e_mask & -e_mask
                                e = lsb.bit_length() - 1
                                e_mask ^= lsb
                                f_mask = f_cand & ~(1 << e)
                                if adj[e] & f_mask:
                                    return True
                        
                        if (mask_a >> d) & 1 and (mask_b >> c) & 1:
                            used = (1 << u) | (1 << v) | (1 << a) | (1 << b) | (1 << c) | (1 << d)
                            e_cand = mask_a & mask_b & ~used
                            f_cand = mask_c & mask_d & ~used
                            
                            e_mask = e_cand
                            while e_mask:
                                lsb = e_mask & -e_mask
                                e = lsb.bit_length() - 1
                                e_mask ^= lsb
                                f_mask = f_cand & ~(1 << e)
                                if adj[e] & f_mask:
                                    return True
        return False

    def find_q3_containing(u, v, adj):
        nu = [x for x in range(n) if x != u and x != v and (adj[u] >> x) & 1]
        nv = [x for x in range(n) if x != v and x != u and (adj[v] >> x) & 1]
        
        for i in range(len(nu)):
            a = nu[i]
            for j in range(i + 1, len(nu)):
                b = nu[j]
                mask_a = adj[a]
                mask_b = adj[b]
                
                for k in range(len(nv)):
                    c = nv[k]
                    for l in range(k + 1, len(nv)):
                        d = nv[l]
                        
                        if a == c or a == d or b == c or b == d:
                            continue
                        
                        mask_c = adj[c]
                        mask_d = adj[d]
                        
                        if (mask_a >> c) & 1 and (mask_b >> d) & 1:
                            used = (1 << u) | (1 << v) | (1 << a) | (1 << b) | (1 << c) | (1 << d)
                            e_cand = mask_a & mask_b & ~used
                            f_cand = mask_c & mask_d & ~used
                            
                            e_mask = e_cand
                            while e_mask:
                                lsb = e_mask & -e_mask
                                e = lsb.bit_length() - 1
                                e_mask ^= lsb
                                f_mask = f_cand & ~(1 << e)
                                if adj[e] & f_mask:
                                    return (u, v)
                        
                        if (mask_a >> d) & 1 and (mask_b >> c) & 1:
                            used = (1 << u) | (1 << v) | (1 << a) | (1 << b) | (1 << c) | (1 << d)
                            e_cand = mask_a & mask_b & ~used
                            f_cand = mask_c & mask_d & ~used
                            
                            e_mask = e_cand
                            while e_mask:
                                lsb = e_mask & -e_mask
                                e = lsb.bit_length() - 1
                                e_mask ^= lsb
                                f_mask = f_cand & ~(1 << e)
                                if adj[e] & f_mask:
                                    return (u, v)
        return None

    while time.time() < deadline:
        # Try to find a Q3
        q3_edge = None
        attempts = 0
        max_attempts = 5
        while attempts < max_attempts and current_edges:
            e_idx = random.randint(0, len(current_edges) - 1)
            u, v = current_edges[e_idx]
            res = find_q3_containing(u, v, adj)
            if res:
                q3_edge = (u, v)
                break
            attempts += 1
            
        if q3_edge:
            u, v = q3_edge
            # Remove (u,v)
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            # Find index and remove
            for i, (eu, ev) in enumerate(current_edges):
                if (eu == u and ev == v) or (eu == v and ev == u):
                    current_edges.pop(i)
                    break
        else:
            # Try to add an edge
            missing_edges = [(u, v) for u in range(n) for v in range(u + 1, n) if not ((adj[u] >> v) & 1)]
            if missing_edges:
                random.shuffle(missing_edges)
                for u, v in missing_edges:
                    if time.time() > deadline: break
                    if not creates_q3(u, v, adj):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        current_edges.append((u, v))
                        
                        if len(current_edges) > best_count:
                            best_count = len(current_edges)
                            best_edges = current_edges.copy()
                        break
    
    return [[u, v] for u, v in best_edges]
