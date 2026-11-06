import time
import random

def construct():
    n = 40
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin for 120s limit

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

    def is_safe(u, v, N1, N4):
        return (N1[u] & N4[v]) == 0 and (N1[v] & N4[u]) == 0

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

    def make_seed_adj():
        adj = [0] * n
        for k in range(8):
            for a in range(5):
                for b in range(a + 1, 5):
                    u = 5 * k + a
                    v = 5 * k + b
                    adj[u] |= 1 << v
                    adj[v] |= 1 << u
        return adj

    def greedy_add(adj, N1, N2, N4, order, deadline):
        for u, v in order:
            if time.time() > deadline:
                break
            if adj[u] & (1 << v):
                continue
            if is_safe(u, v, N1, N4):
                adj[u] |= 1 << v
                adj[v] |= 1 << u
                N1, N2, N4 = recompute(adj)
        return adj, N1, N2, N4

    def local_search(adj, N1, N2, N4, deadline):
        edges = get_edges(adj)
        best_adj = adj[:]
        best_N1, best_N2, best_N4 = N1[:], N2[:], N4[:]
        best_count = len(edges)

        # Try removing each edge and see if we can add more
        for idx, (u, v) in enumerate(edges):
            if time.time() > deadline:
                break

            test_adj = adj[:]
            test_adj[u] &= ~(1 << v)
            test_adj[v] &= ~(1 << u)
            tN1, tN2, tN4 = recompute(test_adj)

            candidates = []
            for a in range(n):
                for b in range(a + 1, n):
                    if not (test_adj[a] & (1 << b)):
                        candidates.append((a, b))
            random.shuffle(candidates)

            test_adj, tN1, tN2, tN4 = greedy_add(test_adj, tN1, tN2, tN4, candidates, deadline)
            new_count = len(get_edges(test_adj))

            if new_count > best_count:
                best_count = new_count
                best_adj = test_adj
                best_N1, best_N2, best_N4 = tN1, tN2, tN4

        return best_adj, best_N1, best_N2, best_N4, best_count

    best_edges = []
    best_count = 0

    all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]

    while time.time() < deadline:
        strategy = random.random()

        if strategy < 0.35:
            # Strategy 1: Greedy from seed with random order
            adj = make_seed_adj()
            N1, N2, N4 = recompute(adj)
            candidates = [(u, v) for u, v in all_pairs if not (adj[u] & (1 << v))]
            random.shuffle(candidates)
            adj, N1, N2, N4 = greedy_add(adj, N1, N2, N4, candidates, deadline)
            edges = get_edges(adj)
            if len(edges) > best_count:
                best_count = len(edges)
                best_edges = edges

        elif strategy < 0.65:
            # Strategy 2: Greedy from empty with random order
            adj = [0] * n
            N1, N2, N4 = recompute(adj)
            candidates = all_pairs[:]
            random.shuffle(candidates)
            adj, N1, N2, N4 = greedy_add(adj, N1, N2, N4, candidates, deadline)
            edges = get_edges(adj)
            if len(edges) > best_count:
                best_count = len(edges)
                best_edges = edges

        elif strategy < 0.85:
            # Strategy 3: Greedy from seed + local search
            adj = make_seed_adj()
            N1, N2, N4 = recompute(adj)
            candidates = [(u, v) for u, v in all_pairs if not (adj[u] & (1 << v))]
            random.shuffle(candidates)
            adj, N1, N2, N4 = greedy_add(adj, N1, N2, N4, candidates, deadline)
            adj, N1, N2, N4, count = local_search(adj, N1, N2, N4, deadline)
            edges = get_edges(adj)
            if count > best_count:
                best_count = count
                best_edges = edges

        else:
            # Strategy 4: Perturbed seed
            adj = make_seed_adj()
            for _ in range(random.randint(2, 4)):
                k = random.randint(0, 7)
                a, b = random.sample(range(5), 2)
                u, v = 5 * k + a, 5 * k + b
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
            N1, N2, N4 = recompute(adj)
            candidates = [(u, v) for u, v in all_pairs if not (adj[u] & (1 << v))]
            random.shuffle(candidates)
            adj, N1, N2, N4 = greedy_add(adj, N1, N2, N4, candidates, deadline)
            edges = get_edges(adj)
            if len(edges) > best_count:
                best_count = len(edges)
                best_edges = edges

    return best_edges
