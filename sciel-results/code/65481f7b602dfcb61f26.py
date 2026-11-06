import random
import time

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    best_adj = None
    best_edge_count = -1
    
    def get_edge_count(adj):
        count = 0
        for i in range(n):
            count += adj[i].bit_count()
        return count // 2

    def get_edges(adj):
        edges = []
        for i in range(n):
            mask = adj[i]
            while mask:
                lsb = mask & -mask
                j = lsb.bit_length() - 1
                edges.append([i, j])
                mask ^= lsb
        return edges

    def has_violation(adj):
        for u in range(n):
            for v in range(u + 1, n):
                for w in range(v + 1, n):
                    common = adj[u] & adj[v] & adj[w]
                    if common.bit_count() >= 3:
                        return (u, v, w, common)
        return None

    def is_safe_to_add(adj, u, v):
        if adj[u] & (1 << v):
            return True
        
        new_adj_u = adj[u] | (1 << v)
        new_adj_v = adj[v] | (1 << u)
        
        C = new_adj_u & new_adj_v
        
        for w in range(n):
            if w == u or w == v:
                continue
            if (C & adj[w]).bit_count() >= 3:
                return False
        
        for x in range(n):
            if x == u or x == v:
                continue
            for y in range(x + 1, n):
                if y == u or y == v:
                    continue
                if (new_adj_u & adj[x] & adj[y]).bit_count() >= 3:
                    return False
                if (new_adj_v & adj[x] & adj[y]).bit_count() >= 3:
                    return False
        return True

    # Run multiple restarts if time permits
    while time.time() < deadline - 5:
        # Initialize
        adj = [0] * n
        p = 0.5
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < p:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        current_adj = adj
        stuck_counter = 0
        
        while time.time() < deadline - 5:
            viol = has_violation(current_adj)
            if viol:
                u, v, w, common_mask = viol
                neighbors = []
                m = common_mask
                while len(neighbors) < 3 and m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    neighbors.append(idx)
                    m ^= lsb
                
                candidates = []
                for a in [u, v, w]:
                    for b in neighbors:
                        if current_adj[a] & (1 << b):
                            candidates.append((a, b))
                
                if candidates:
                    a, b = random.choice(candidates)
                    current_adj[a] &= ~(1 << b)
                    current_adj[b] &= ~(1 << a)
                else:
                    # Should not happen if violation exists
                    break
            else:
                added = False
                for _ in range(100):
                    if time.time() > deadline - 5:
                        break
                    u = random.randint(0, n - 1)
                    v = random.randint(0, n - 1)
                    if u == v:
                        continue
                    if current_adj[u] & (1 << v):
                        continue
                    
                    if is_safe_to_add(current_adj, u, v):
                        current_adj[u] |= (1 << v)
                        current_adj[v] |= (1 << u)
                        added = True
                
                if not added:
                    stuck_counter += 1
                    if stuck_counter > 10:
                        # Kick
                        all_edges = []
                        for i in range(n):
                            for j in range(i + 1, n):
                                if current_adj[i] & (1 << j):
                                    all_edges.append((i, j))
                        if all_edges:
                            u, v = random.choice(all_edges)
                            current_adj[u] &= ~(1 << v)
                            current_adj[v] &= ~(1 << u)
                        stuck_counter = 0
                else:
                    stuck_counter = 0
                
                # Update best
                ec = get_edge_count(current_adj)
                if ec > best_edge_count:
                    best_edge_count = ec
                    best_adj = [m for m in current_adj]
        
        # End of restart

    if best_adj is None:
        return []
    return get_edges(best_adj)
