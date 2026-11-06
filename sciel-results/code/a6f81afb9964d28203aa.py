import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110
    n = 40
    best_edges = None
    best_count = 0

    def makes_c6(u, v, adj):
        target = adj[u] & ~(1 << v)
        if target == 0:
            return False
        for a in range(n):
            if a == u or a == v:
                continue
            if not (adj[v] >> a) & 1:
                continue
            b_set = adj[a] & ~(1 << u) & ~(1 << v) & ~(1 << a)
            while b_set:
                b = (b_set & -b_set).bit_length() - 1
                b_set &= b_set - 1
                c_set = adj[b] & ~(1 << u) & ~(1 << v) & ~(1 << a) & ~(1 << b)
                while c_set:
                    c = (c_set & -c_set).bit_length() - 1
                    c_set &= c_set - 1
                    d_set = adj[c] & ~(1 << u) & ~(1 << v) & ~(1 << a) & ~(1 << b) & ~(1 << c) & target
                    if d_set:
                        return True
        return False

    def to_adj(edge_list):
        adj = [0] * n
        for u, v in edge_list:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def from_adj(adj):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    edges.append([u, v])
        return edges

    def edge_count(adj):
        return sum(bin(adj[u]).count('1') for u in range(n)) // 2

    def greedy_add(adj):
        changed = True
        while changed:
            changed = False
            for u in range(n):
                for v in range(u + 1, n):
                    if not ((adj[u] >> v) & 1):
                        if not makes_c6(u, v, adj):
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                            changed = True
        return adj

    def local_search(adj, time_limit):
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            for u in range(n):
                for v in range(u + 1, n):
                    if not ((adj[u] >> v) & 1):
                        if not makes_c6(u, v, adj):
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                            improved = True
                            break
                if improved:
                    break
            if improved:
                continue
            non_edges = [(u, v) for u in range(n) for v in range(u + 1, n) if not ((adj[u] >> v) & 1)]
            edges_list = [(u, v) for u in range(n) for v in range(u + 1, n) if (adj[u] >> v) & 1]
            random.shuffle(non_edges)
            random.shuffle(edges_list)
            found = False
            for a, b in edges_list[:30]:
                if time.time() > time_limit:
                    break
                adj[a] &= ~(1 << b)
                adj[b] &= ~(1 << a)
                for u, v in non_edges[:30]:
                    if not makes_c6(u, v, adj):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        found = True
                        break
                    if time.time() > time_limit:
                        break
                if not found:
                    adj[a] |= (1 << b)
                    adj[b] |= (1 << a)
            improved = found
        return adj

    # Initial: 8 disjoint K5s
    edges = []
    for k in range(8):
        for a in range(5):
            for b in range(a + 1, 5):
                edges.append([5 * k + a, 5 * k + b])
    adj = to_adj(edges)
    adj = greedy_add(adj)
    ec = edge_count(adj)
    if ec > best_count:
        best_count = ec
        best_edges = from_adj(adj)

    random.seed(42)
    while time.time() < deadline - 10:
        adj = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if random.random() < 0.3:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        adj = greedy_add(adj)
        ec = edge_count(adj)
        if ec > best_count:
            best_count = ec
            best_edges = from_adj(adj)
        adj = local_search(adj, deadline - 5)
        ec = edge_count(adj)
        if ec > best_count:
            best_count = ec
            best_edges = from_adj(adj)

    return best_edges if best_edges else edges
