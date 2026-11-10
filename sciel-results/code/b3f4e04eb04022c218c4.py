import random
import time

def construct():
    N = 20
    deadline = time.time() + 110
    
    def is_valid(adj):
        for i in range(N):
            for j in range(i + 1, N):
                for k in range(j + 1, N):
                    if (adj[i] & adj[j] & adj[k]).bit_count() > 2:
                        return False
        return True

    def get_violation(adj):
        for i in range(N):
            for j in range(i + 1, N):
                for k in range(j + 1, N):
                    common = adj[i] & adj[j] & adj[k]
                    if common.bit_count() > 2:
                        neighbors = [v for v in range(N) if (common >> v) & 1]
                        return [i, j, k] + neighbors[:3]
        return None

    def remove_edge(adj, u, v):
        adj[u] ^= (1 << v)
        adj[v] ^= (1 << u)

    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def has_edge(adj, u, v):
        return (adj[u] >> v) & 1

    def count_edges(adj):
        return sum(adj[i].bit_count() for i in range(N)) // 2

    def random_graph(p):
        adj = [0] * N
        for i in range(N):
            for j in range(i + 1, N):
                if random.random() < p:
                    add_edge(adj, i, j)
        return adj

    def is_valid_after_add(adj, u, v):
        for w in range(N):
            if w == u or w == v:
                continue
            for x in range(w + 1, N):
                if x == u or x == v:
                    continue
                if (adj[w] >> v) & 1 and (adj[x] >> v) & 1:
                    if (adj[u] & adj[w] & adj[x]).bit_count() >= 2:
                        return False
                if (adj[w] >> u) & 1 and (adj[x] >> u) & 1:
                    if (adj[v] & adj[w] & adj[x]).bit_count() >= 2:
                        return False
        return True

    def local_search(adj):
        while True:
            if is_valid(adj):
                break
            viol = get_violation(adj)
            if viol is None:
                break
            u_part = viol[:3]
            v_part = viol[3:]
            u = random.choice(u_part)
            v = random.choice(v_part)
            if has_edge(adj, u, v):
                remove_edge(adj, u, v)
            else:
                found = False
                for uu in u_part:
                    for vv in v_part:
                        if has_edge(adj, uu, vv):
                            remove_edge(adj, uu, vv)
                            found = True
                            break
                    if found:
                        break
                if not found:
                    break

        improved = True
        while improved:
            improved = False
            non_edges = []
            for i in range(N):
                for j in range(i + 1, N):
                    if not has_edge(adj, i, j):
                        non_edges.append((i, j))
            random.shuffle(non_edges)
            
            for u, v in non_edges:
                if time.time() > deadline:
                    return adj
                if is_valid_after_add(adj, u, v):
                    add_edge(adj, u, v)
                    improved = True
                    break
        return adj

    best_adj = None
    best_count = -1
    
    while time.time() < deadline:
        p = random.uniform(0.3, 0.6)
        adj = random_graph(p)
        adj = local_search(adj)
        
        cnt = count_edges(adj)
        if cnt > best_count:
            best_count = cnt
            best_adj = adj[:]
            
        if time.time() > deadline - 5:
            break

    if best_adj is None:
        return [[a, b] for a in range(2) for b in range(2, 20)]
    
    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            if has_edge(best_adj, i, j):
                edges.append([i, j])
    return edges
