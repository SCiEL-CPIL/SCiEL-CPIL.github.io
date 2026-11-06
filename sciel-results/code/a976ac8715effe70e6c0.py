"""C6-free graph on 40 vertices — maximize edges via fast local search with diverse starts."""
import time
import random


def construct():
    n = 40
    start_time = time.time()
    deadline = start_time + 115.0

    all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]

    def count_edges(adj):
        return sum(bin(adj[u]).count('1') for u in range(n)) // 2

    def get_edges(adj):
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

    def recompute_n1_n2_n4(adj):
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

    def is_safe(u, v, N1, N4):
        return (N1[u] & N4[v]) == 0 and (N1[v] & N4[u]) == 0

    def make_k5_seed():
        adj = [0] * n
        for k in range(8):
            for a in range(5):
                for b in range(a + 1, 5):
                    u = 5 * k + a
                    v = 5 * k + b
                    adj[u] |= 1 << v
                    adj[v] |= 1 << u
        return adj

    def greedy_from_adj(adj, deadline):
        N1, N2, N4 = recompute_n1_n2_n4(adj)
        candidates = [(u, v) for u, v in all_pairs if not (adj[u] & (1 << v))]
        random.shuffle(candidates)
        for u, v in candidates:
            if time.time() > deadline:
                break
            if adj[u] & (1 << v):
                continue
            if is_safe(u, v, N1, N4):
                adj[u] |= 1 << v
                adj[v] |= 1 << u
                N1, N2, N4 = recompute_n1_n2_n4(adj)
        return adj

    def local_search_readd(adj, deadline):
        N1, N2, N4 = recompute_n1_n2_n4(adj)
        best_adj = adj[:]
        best_count = count_edges(adj)
        edges = get_edges(adj)
        random.shuffle(edges)

        for u, v in edges:
            if time.time() > deadline:
                break
            test_adj = adj[:]
            test_adj[u] &= ~(1 << v)
            test_adj[v] &= ~(1 << u)
            tN1, tN2, tN4 = recompute_n1_n2_n4(test_adj)

            candidates = [(a, b) for a, b in all_pairs if not (test_adj[a] & (1 << b))]
            random.shuffle(candidates)
            for a, b in candidates:
                if time.time() > deadline:
                    break
                if test_adj[a] & (1 << b):
                    continue
                if is_safe(a, b, tN1, tN4):
                    test_adj[a] |= 1 << b
                    test_adj[b] |= 1 << a
                    tN1, tN2, tN4 = recompute_n1_n2_n4(test_adj)

            new_count = count_edges(test_adj)
            if new_count > best_count:
                best_count = new_count
                best_adj = test_adj[:]

        return best_adj, best_count

    best_adj = None
    best_count = 0
    best_edges = []

    # Strategy 1: K5 seed + greedy
    adj = make_k5_seed()
    adj = greedy_from_adj(adj, deadline)
    cnt = count_edges(adj)
    if cnt > best_count:
        best_count = cnt
        best_adj = adj
        best_edges = get_edges(adj)

    # Strategy 2: Empty + greedy
    adj = [0] * n
    adj = greedy_from_adj(adj, deadline)
    cnt = count_edges(adj)
    if cnt > best_count:
        best_count = cnt
        best_adj = adj
        best_edges = get_edges(adj)

    # Strategy 3: Multiple restarts with local search
    while time.time() < deadline - 15:
        r = random.random()
        if r < 0.4:
            adj = make_k5_seed()
        elif r < 0.7:
            adj = [0] * n
        else:
            adj = [0] * n
            for u, v in all_pairs:
                if random.random() < 0.08:
                    adj[u] |= 1 << v
                    adj[v] |= 1 << u

        adj = greedy_from_adj(adj, deadline)
        cnt = count_edges(adj)

        if cnt >= best_count:
            adj, cnt = local_search_readd(adj, deadline)
            if cnt > best_count:
                best_count = cnt
                best_adj = adj
                best_edges = get_edges(adj)

    # Final polish
    if best_adj is not None and time.time() < deadline - 3:
        best_adj, best_count = local_search_readd(best_adj, deadline)
        best_edges = get_edges(best_adj)

    if not best_edges:
        best_edges = get_edges(make_k5_seed())

    return best_edges
