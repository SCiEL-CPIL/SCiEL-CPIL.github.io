import time
import random

def construct():
    n = 40
    start_time = time.time()
    deadline = start_time + 115.0  # 5s margin
    
    def recompute(adj):
        N1 = adj[:]
        N2 = [0] * n
        for i in range(n):
            mask = 0
            x = N1[i]
            while x:
                lsb = x & -x
                j = lsb.bit_length() - 1
                mask |= N1[j]
                x ^= lsb
            N2[i] = mask
        N4 = [0] * n
        for i in range(n):
            mask = 0
            x = N2[i]
            while x:
                lsb = x & -x
                j = lsb.bit_length() - 1
                mask |= N2[j]
                x ^= lsb
            N4[i] = mask
        return N1, N2, N4
    
    def is_c6_free(adj):
        N1, N2, N4 = recompute(adj)
        for u in range(n):
            x = adj[u]
            while x:
                lsb = x & -x
                v = lsb.bit_length() - 1
                if u < v:
                    if (N1[u] & N4[v]) != 0 or (N1[v] & N4[u]) != 0:
                        return False
                x ^= lsb
        return True
    
    def add_edge_safe(u, v, adj, N1, N4):
        if adj[u] & (1 << v):
            return False
        if (N1[u] & N4[v]) != 0 or (N1[v] & N4[u]) != 0:
            return False
        adj[u] |= 1 << v
        adj[v] |= 1 << u
        return True
    
    def edge_count(adj):
        count = 0
        for u in range(n):
            count += bin(adj[u]).count('1')
        return count // 2
    
    def to_edges(adj):
        edges = []
        for u in range(n):
            x = adj[u]
            while x:
                lsb = x & -x
                v = lsb.bit_length() - 1
                if u < v:
                    edges.append([u, v])
                x ^= lsb
        return edges
    
    best_edges = []
    best_count = 0
    
    def try_improve(adj):
        nonlocal best_edges, best_count
        N1, N2, N4 = recompute(adj)
        count = edge_count(adj)
        
        # Greedy add
        pairs = [(u, v) for u in range(n) for v in range(u+1, n) if not (adj[u] & (1<<v))]
        random.shuffle(pairs)
        for u, v in pairs:
            if time.time() > deadline:
                break
            if add_edge_safe(u, v, adj, N1, N4):
                N1, N2, N4 = recompute(adj)
                count += 1
        
        # Local search: try edge swaps
        improved = True
        while improved and time.time() < deadline:
            improved = False
            N1, N2, N4 = recompute(adj)
            current_count = edge_count(adj)
            # Try removing an edge and adding two
            edges_list = to_edges(adj)
            random.shuffle(edges_list)
            for e1 in edges_list:
                if time.time() > deadline:
                    break
                u1, v1 = e1[0], e1[1]
                # Try adding two new edges
                candidates = [(u, v) for u in range(n) for v in range(u+1, n) 
                             if not (adj[u] & (1<<v)) and (u,v) != (u1,v1) and (u,v) != (v1,u1)]
                random.shuffle(candidates)
                for e2 in candidates[:50]:  # limit candidates
                    u2, v2 = e2
                    for e3 in candidates[:50]:
                        u3, v3 = e3
                        if e2 == e3:
                            continue
                        # Try swap: remove (u1,v1), add (u2,v2) and (u3,v3)
                        adj[u1] ^= (1 << v1)
                        adj[v1] ^= (1 << u1)
                        if add_edge_safe(u2, v2, adj, N1, N4):
                            N1, N2, N4 = recompute(adj)
                            if add_edge_safe(u3, v3, adj, N1, N4):
                                N1, N2, N4 = recompute(adj)
                                new_count = edge_count(adj)
                                if new_count > current_count:
                                    current_count = new_count
                                    improved = True
                                else:
                                    # revert
                                    adj[u3] ^= (1 << v3)
                                    adj[v3] ^= (1 << u3)
                                    adj[u2] ^= (1 << v2)
                                    adj[v2] ^= (1 << u2)
                            else:
                                adj[u2] ^= (1 << v2)
                                adj[v2] ^= (1 << u2)
                        else:
                            adj[u1] ^= (1 << v1)
                            adj[v1] ^= (1 << u1)
                            break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            
            if current_count > best_count:
                best_count = current_count
                best_edges = to_edges(adj)
        
        if count > best_count:
            best_count = count
            best_edges = to_edges(adj)
    
    # Seed: 8xK5
    seed_adj = [0] * n
    for k in range(8):
        for a in range(5):
            for b in range(a + 1, 5):
                u = 5 * k + a
                v = 5 * k + b
                seed_adj[u] |= 1 << v
                seed_adj[v] |= 1 << u
    
    # Multiple restarts
    while time.time() < deadline:
        adj = seed_adj[:]
        try_improve(adj)
        
        if time.time() > deadline:
            break
        
        # Also try random start
        if random.random() < 0.3:
            adj = [0] * n
            # Add random edges
            for _ in range(60):
                u, v = random.sample(range(n), 2)
                adj[u] |= 1 << v
                adj[v] |= 1 << u
            try_improve(adj)
    
    return best_edges
