import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    N = 40
    
    def initialize_random():
        adj = [0] * N
        for i in range(N):
            for j in range(i + 1, N):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return adj

    def has_triangle(adj):
        for u in range(N):
            neighbors_u = adj[u]
            while neighbors_u:
                lsb = neighbors_u & -neighbors_u
                v = lsb.bit_length() - 1
                neighbors_u ^= lsb
                common = adj[u] & adj[v]
                if common:
                    lsb_w = common & -common
                    w = lsb_w.bit_length() - 1
                    return (u, v, w)
        return None

    def find_independent_set_10(adj):
        # Try multiple random greedy passes
        for _ in range(20):
            verts = list(range(N))
            random.shuffle(verts)
            current_set = []
            for v in verts:
                is_ok = True
                for u in current_set:
                    if adj[v] & (1 << u):
                        is_ok = False
                        break
                if is_ok:
                    current_set.append(v)
                    if len(current_set) == 10:
                        return current_set
        return None

    adj = initialize_random()
    
    while time.time() < deadline:
        tri = has_triangle(adj)
        if tri:
            u, v, w = tri
            a, b = random.choice([(u, v), (v, w), (u, w)])
            adj[a] &= ~(1 << b)
            adj[b] &= ~(1 << a)
            continue
        
        indep_set = find_independent_set_10(adj)
        if not indep_set:
            edges = []
            for i in range(N):
                for j in range(i + 1, N):
                    if adj[i] & (1 << j):
                        edges.append([i, j])
            return {'n': N, 'edges': edges}
        
        found_edge = False
        random.shuffle(indep_set)
        for i in range(len(indep_set)):
            for j in range(i + 1, len(indep_set)):
                u = indep_set[i]
                v = indep_set[j]
                if (adj[u] & adj[v]) == 0:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    found_edge = True
                    break
            if found_edge:
                break
        
        if not found_edge:
            all_edges = []
            for i in range(N):
                for j in range(i + 1, N):
                    if adj[i] & (1 << j):
                        all_edges.append((i, j))
            if all_edges:
                u, v = random.choice(all_edges)
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
            else:
                adj = initialize_random()

    # Fallback
    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            if adj[i] & (1 << j):
                edges.append([i, j])
    return {'n': N, 'edges': edges}
