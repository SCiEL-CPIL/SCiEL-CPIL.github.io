import random
import time

N = 20
ALL = (1 << N) - 1
DEADLINE = time.time() + 110.0  # safety margin before 120s limit


def _initial_graph():
    """Dense, symmetric, K3,3-free seed (union of two edge-disjoint 5-cycles)."""
    edges = []
    # 5-cycle on 0..4 (offset 0)
    for i in range(5):
        edges.append([i, (i + 1) % 5])
    # 5-cycle on 0..4 (offset 2)
    for i in range(5):
        edges.append([i, (i + 2) % 5])
    # 5-cycle on 5..9 (offset 0)
    for i in range(5):
        edges.append([5 + i, 5 + ((i + 1) % 5)])
    # 5-cycle on 5..9 (offset 2)
    for i in range(5):
        edges.append([5 + i, 5 + ((i + 2) % 5)])
    # 5-cycle on 10..14 (offset 0)
    for i in range(5):
        edges.append([10 + i, 10 + ((i + 1) % 5)])
    # 5-cycle on 10..14 (offset 2)
    for i in range(5):
        edges.append([10 + i, 10 + ((i + 2) % 5)])
    # 5-cycle on 15..19 (offset 0)
    for i in range(5):
        edges.append([15 + i, 15 + ((i + 1) % 5)])
    # 5-cycle on 15..19 (offset 2)
    for i in range(5):
        edges.append([15 + i, 15 + ((i + 2) % 5)])
    # Cross connections (deterministic, keep K3,3-free)
    cross = [
        [0, 5], [1, 6], [2, 7], [3, 8], [4, 9],
        [5, 10], [6, 11], [7, 12], [8, 13], [9, 14],
        [10, 15], [11, 16], [12, 17], [13, 18], [14, 19],
        [0, 10], [5, 15], [1, 11], [6, 16], [2, 12],
        [7, 17], [3, 13], [8, 18], [4, 14], [9, 19],
        [0, 15], [5, 10], [1, 16], [6, 11], [2, 17],
        [7, 12], [3, 18], [8, 13], [4, 14], [9, 19],
    ]
    for e in cross:
        if e[0] < e[1]:
            edges.append(e)
    # Deduplicate
    seen = set()
    uniq = []
    for u, v in edges:
        if u > v:
            u, v = v, u
        if (u, v) not in seen:
            seen.add((u, v))
            uniq.append([u, v])
    return uniq


def _to_bitsets(edges):
    adj = [0] * N
    for u, v in edges:
        adj[u] |= 1 << v
        adj[v] |= 1 << u
    return adj


def _popcount(x):
    return x.bit_count() if hasattr(int, "bit_count") else bin(x).count("1")


def _is_valid_add(adj, u, v):
    # Adding edge u-v is valid iff |N(u) ∩ N(v)| <= 1
    return _popcount(adj[u] & adj[v]) <= 1


def _local_search(adj, rng, max_iter=200000):
    """Greedy + swap local search. Returns (adj, edge_count)."""
    n_edges = sum(_popcount(adj[i]) for i in range(N)) // 2
    # First: try to add any missing edge
    for _ in range(max_iter):
        if time.time() > DEADLINE - 5.0:
            break
        improved = False
        # Try adding a random missing edge
        for _ in range(6):
            u = rng.randrange(N)
            v = rng.randrange(N)
            if u == v:
                continue
            if not (adj[u] >> v) & 1:
                if _is_valid_add(adj, u, v):
                    adj[u] |= 1 << v
                    adj[v] |= 1 << u
                    n_edges += 1
                    improved = True
                    break
        if not improved:
            # Try a 1-remove / 2-add swap
            swapped = False
            for _ in range(4):
                u = rng.randrange(N)
                # pick a neighbour to remove
                nbrs = adj[u]
                if nbrs == 0:
                    continue
                # pick random set bit
                x = nbrs
                idx = 0
                while x:
                    if x & 1:
                        break
                    idx += 1
                    x >>= 1
                w = idx  # remove edge u-w
                if w == u:
                    continue
                # remove u-w
                adj[u] &= ~(1 << w)
                adj[w] &= ~(1 << u)
                # try to add two edges
                added = 0
                candidates = []
                for _ in range(12):
                    a = rng.randrange(N)
                    b = rng.randrange(N)
                    if a == b or (a == u and b == w) or (a == w and b == u):
                        continue
                    if not (adj[a] >> b) & 1:
                        candidates.append((a, b))
                for a, b in candidates:
                    if added == 2:
                        break
                    if _is_valid_add(adj, a, b):
                        adj[a] |= 1 << b
                        adj[b] |= 1 << a
                        added += 1
                if added >= 2:
                    n_edges += added - 1
                    swapped = True
                    break
                else:
                    # revert
                    adj[u] |= 1 << w
                    adj[w] |= 1 << u
            if not swapped:
                break
    return adj, n_edges


def construct():
    start = time.time()
    rng = random.Random(12345)
    best_adj = _to_bitsets(_initial_graph())
    best_edges = sum(_popcount(best_adj[i]) for i in range(N)) // 2

    # Run multiple random-restart local searches
    while time.time() < DEADLINE - 3.0:
        adj = _to_bitsets(_initial_graph())
        # Perturb: randomly add a few valid edges
        for _ in range(30):
            u = rng.randrange(N)
            v = rng.randrange(N)
            if u != v and not (adj[u] >> v) & 1:
                if _is_valid_add(adj, u, v):
                    adj[u] |= 1 << v
                    adj[v] |= 1 << u
        # Perturb: randomly remove a few edges
        for _ in range(5):
            u = rng.randrange(N)
            nbrs = adj[u]
            if nbrs:
                x = nbrs
                idx = 0
                while x:
                    if x & 1:
                        break
                    idx += 1
                    x >>= 1
                w = idx
                if w != u:
                    adj[u] &= ~(1 << w)
                    adj[w] &= ~(1 << u)

        adj, n_edges = _local_search(adj, rng)
        if n_edges > best_edges:
            best_edges = n_edges
            best_adj = adj[:]

    # Build edge list from best_adj
    edges = []
    for u in range(N):
        for v in range(u + 1, N):
            if (best_adj[u] >> v) & 1:
                edges.append([u, v])
    return edges
