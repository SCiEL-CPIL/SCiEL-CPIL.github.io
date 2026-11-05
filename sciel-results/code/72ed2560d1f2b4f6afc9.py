import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    def get_k4(adj, n):
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    mask = adj[u] & adj[v]
                    # Find an edge in mask
                    m = mask
                    while m:
                        l = m & -m
                        w = l.bit_length() - 1
                        m2 = m ^ l
                        if adj[w] & m2:
                            x = (adj[w] & m2).bit_length() - 1
                            # Check if x is in m2
                            if (m2 >> x) & 1:
                                return [u, v, w, x]
                        m = m2
        return None

    def get_ind6(adj, n):
        comp_adj = [0] * n
        full_mask = (1 << n) - 1
        for i in range(n):
            comp_adj[i] = full_mask ^ (1 << i) ^ adj[i]
        
        # Find K6 in comp_adj
        # Simple recursive search returning the set
        import sys
        sys.setrecursionlimit(10000)
        
        def search(candidates, path):
            if len(path) >= 6:
                return path[:]
            if not candidates:
                return None
            # Pick vertex with max degree in candidates
            best_v = -1
            best_deg = -1
            for v in candidates:
                deg = bin(comp_adj[v] & candidates).count('1')
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
            
            new_candidates = candidates & comp_adj[best_v]
            res = search(new_candidates, path + [best_v])
            if res:
                return res
                
            new_candidates = candidates & ~(1 << best_v)
            return search(new_candidates, path)

        return search((1 << n) - 1, [])

    def is_valid(adj, n):
        return get_k4(adj, n) is None and get_ind6(adj, n) is None

    def adj_to_edges(adj, n):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (adj[i] >> j) & 1:
                    edges.append([i, j])
        return edges

    for n in range(40, 34, -1):
        if time.time() > deadline - 5:
            break
            
        for restart in range(100):
            if time.time() > deadline:
                break
                
            p = 0.5
            adj = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < p:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
                        
            # Repair loop
            steps = 0
            while steps < 1000:
                if time.time() > deadline:
                    break
                    
                k4 = get_k4(adj, n)
                if k4:
                    # Remove random edge from k4
                    e = random.choice([(k4[i], k4[j]) for i in range(4) for j in range(i+1, 4)])
                    u, v = e
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    steps += 1
                    continue
                
                ind6 = get_ind6(adj, n)
                if ind6:
                    # Add random edge within ind6
                    e = random.choice([(ind6[i], ind6[j]) for i in range(6) for j in range(i+1, 6)])
                    u, v = e
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    steps += 1
                    continue
                
                # Valid!
                return {'n': n, 'edges': adj_to_edges(adj, n)}
                
    # Fallback
    squares = {x * x % 17 for x in range(1, 17)}
    return {'n': 17, 'edges': [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]}
