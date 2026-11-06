import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 112  # safety margin under 120s (evaluator kills at 150s)

    n = 40
    full_mask = (1 << n) - 1

    def now_dead():
        return time.time() >= deadline

    # adjacency as list of int bitmasks
    adj = [0] * n

    def bit(i):
        return 1 << i

    def has_edge(u, v):
        return (adj[u] >> v) & 1

    def add_edge(u, v):
        adj[u] |= bit(v)
        adj[v] |= bit(u)

    def remove_edge(u, v):
        adj[u] &= ~bit(v)
        adj[v] &= ~bit(u)

    def edge_count():
        return sum(adj[u].bit_count() for u in range(n)) // 2

    def creates_c6(u, v):
        """Return True if adding edge (u,v) creates a C6 (assume not already present)."""
        mask_uv = bit(u) | bit(v)
        nu = adj[u] & ~bit(v)
        nv = adj[v] & ~bit(u)
        # iterate a in nu
        a = nu
        while a:
            lb = a & (-a)
            ai = lb.bit_length() - 1
            a ^= lb
            # d in nv, d != a
            d = nv & ~lb
            while d:
                ld = d & (-d)
                di = ld.bit_length() - 1
                d ^= ld
                # need path a-b-c-d length 3 in graph, b,c distinct from a,d,u,v
                # b in N(a), c in N(d), b~c
                # b candidates: N(a) minus a,d,u,v
                bmask = adj[ai] & ~lb & ~ld & ~mask_uv
                # for each b in bmask, check N(b) & N(d) has a c (c != a,d,u,v,b)
                b = bmask
                while b:
                    lb2 = b & (-b)
                    bi = lb2.bit_length() - 1
                    b ^= lb2
                    cmask = adj[bi] & ld & ~lb & ~mask_uv & ~lb2
                    if cmask:
                        return True
        return False

    def is_c6_free():
        # verify no C6 (for final check) - use creates_c6 logic on each edge? expensive.
        # Instead do a direct BFS-based C6 check.
        for u in range(n):
            for v in range(u + 1, n):
                if not has_edge(u, v):
                    continue
                # temporarily remove u-v and check creates_c6
                remove_edge(u, v)
                bad = creates_c6(u, v)
                add_edge(u, v)
                if bad:
                    return False
        return True

    # Skip initial greedy to save time and avoid buggy creates_c6.
    # Start with empty graph.
    best_edges = 0
    best_adj = [0] * n

    def snapshot():
        return adj[:]

    def restore(a):
        for i in range(n):
            adj[i] = a[i]

    # Restart loop
    attempts = 0
    while not now_dead():
        attempts += 1
        random.seed(12345 + attempts)
        adj2 = [0] * n
        # need to re-run greedy with new order using adj2
        # refactor: run greedy into a fresh adj
        def greedy(seed):
            ra = [0] * n
            def add2(u, v):
                ra[u] |= 1 << v
                ra[v] |= 1 << u
            def has2(u, v):
                return (ra[u] >> v) & 1
            def creates2(u, v):
                mask_uv = (1 << u) | (1 << v)
                nu = ra[u] & ~(1 << v)
                nv = ra[v] & ~(1 << u)
                a = nu
                while a:
                    lb = a & (-a)
                    ai = lb.bit_length() - 1
                    a ^= lb
                    d = nv & ~lb
                    while d:
                        ld = d & (-d)
                        di = ld.bit_length() - 1
                        d ^= ld
                        # Check for path a-b-c-d of length 3
                        # b in N(a), c in N(d), b~c
                        # b != a, d, u, v
                        bmask = ra[ai] & ~lb & ~ld & ~mask_uv
                        b = bmask
                        while b:
                            lb2 = b & (-b)
                            bi = lb2.bit_length() - 1
                            b ^= lb2
                            # c in N(b) intersect N(d)
                            # c != a, b, d, u, v
                            cmask = ra[bi] & ra[di] & ~lb & ~lb2 & ~ld & ~mask_uv
                            if cmask:
                                return True
                        # end d
                    # end a
                return False
            pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
            random.seed(seed)
            random.shuffle(pairs)
            for (u, v) in pairs:
                if time.time() >= deadline - 2:
                    break
                if not has2(u, v) and not creates2(u, v):
                    add2(u, v)
            return ra, sum(ra[u].bit_count() for u in range(n)) // 2

        ra, cnt = greedy(12345 + attempts)
        if cnt > best_edges:
            best_edges = cnt
            best_adj = ra[:]

        # also try local improvement on best_adj: add any non-edge that fits
        # (greedy already maximal for its order, but a different order might allow more; restarts cover that)

    # Use best_adj
    adj = best_adj[:]

    # Build edge list
    edges = []
    for u in range(n):
        m = adj[u]
        for v in range(u + 1, n):
            if (m >> v) & 1:
                edges.append([u, v])

    return edges
