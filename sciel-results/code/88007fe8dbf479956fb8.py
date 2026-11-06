import random

def construct():
    n = 40
    best_edges = []
    best_count = 0
    
    for seed in range(20):
        random.seed(seed)
        adj = [set() for _ in range(n)]
        adj_mask = [0] * n
        
        p = 0.2
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < p:
                    adj[i].add(j)
                    adj[j].add(i)
                    adj_mask[i] |= (1 << j)
                    adj_mask[j] |= (1 << i)
                    
        def find_c6():
            for u in range(n):
                for v in range(u + 1, n):
                    if u in adj[v]:
                        continue
                    paths = []
                    for a in adj[u]:
                        if a == v: continue
                        for b in adj[a]:
                            if b == u or b == a: continue
                            for c in adj[b]:
                                if c == u or c == a or c == b: continue
                                if c == v:
                                    paths.append((a, b))
                    for i in range(len(paths)):
                        a1, b1 = paths[i]
                        for j in range(i + 1, len(paths)):
                            a2, b2 = paths[j]
                            if a1 != a2 and a1 != b2 and b1 != a2 and b1 != b2:
                                return [u, a1, b1, v, b2, a2]
            return None

        while True:
            cycle = find_c6()
            if cycle is None:
                break
            edge_idx = random.randint(0, 5)
            u = cycle[edge_idx]
            v = cycle[(edge_idx + 1) % 6]
            if v in adj[u]:
                adj[u].remove(v)
                adj[v].remove(u)
                adj_mask[u] &= ~(1 << v)
                adj_mask[v] &= ~(1 << u)
        
        changed = True
        while changed:
            changed = False
            for u in range(n):
                for v in range(u + 1, n):
                    if v in adj[u]:
                        continue
                    found_path = False
                    mask = (1 << u) | (1 << v)
                    for a in adj[u]:
                        mask_a = mask | (1 << a)
                        for b in adj[a]:
                            if b == u or b == a: continue
                            mask_b = mask_a | (1 << b)
                            for c in adj[b]:
                                if c == u or c == a or c == b: continue
                                mask_c = mask_b | (1 << c)
                                for d in adj[c]:
                                    if d == u or d == a or d == b or d == c: continue
                                    mask_d = mask_c | (1 << d)
                                    common = adj_mask[d] & adj_mask[v]
                                    common &= ~mask_d
                                    common &= ~(1 << v)
                                    if common != 0:
                                        found_path = True
                                        break
                                    if found_path: break
                                if found_path: break
                            if found_path: break
                        if found_path: break
                    if not found_path:
                        adj[u].add(v)
                        adj[v].add(u)
                        adj_mask[u] |= (1 << v)
                        adj_mask[v] |= (1 << u)
                        changed = True
        
        edge_count = sum(len(s) for s in adj) // 2
        if edge_count > best_count:
            best_count = edge_count
            best_edges = []
            for i in range(n):
                for j in adj[i]:
                    if i < j:
                        best_edges.append([i, j])
                        
    return best_edges
