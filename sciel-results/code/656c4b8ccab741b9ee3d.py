import time
import itertools
import random

def construct():
    start_time = time.time()
    deadline = start_time + 128.0  # safety margin for 150s limit

    n = 16
    FULL = (1 << n) - 1

    # Q3 structure: 3-cube with 8 vertices (0..7), edge if Hamming distance 1
    Q3_EDGES = []
    for u in range(8):
        for v in range(u + 1, 8):
            if bin(u ^ v).count('1') == 1:
                Q3_EDGES.append((u, v))

    # All 8-vertex subsets of 0..15, as bitmasks
    all_8_subsets = [0] * 0
    all_8_subsets = [sum(1 << v for v in S) for S in itertools.combinations(range(n), 8)]

    def _count_edges(adj):
        return sum(bin(adj[u]).count('1') for u in range(n)) // 2

    def _get_edge_list(adj):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    edges.append([u, v])
        return edges

    def find_one_q3(adj):
        """Return a Q3 (list of 12 (u,v) edges) if present, else None.
        Uses precomputed 8-subset masks for fast degree filtering."""
        for S in all_8_subsets:
            # Filter: every vertex in S must have degree >= 3 within S
            ok = True
            temp = S
            while temp:
                lb = temp & (-temp)
                v = lb.bit_length() - 1
                if (adj[v] & S).bit_count() < 3:
                    ok = False
                    break
                temp ^= lb
            if not ok:
                continue

            # Backtracking isomorphism check: assign Q3 vertex idx to graph vertex
            mapping = [-1] * 8
            used = 0
            # Precompute for each idx which Q3 neighbors have index < idx
            assigned_nb = []
            for idx in range(8):
                nbrs = []
                for nb in range(8):
                    if nb < idx and (Q3_EDGES and any((a == nb and b == idx) or (a == idx and b == nb) for a, b in Q3_EDGES)):
                        nbrs.append(nb)
                assigned_nb.append(nbrs)

            def dfs(idx):
                if time.time() > deadline - 0.3:
                    return False
                if idx == 8:
                    return True
                for v in S:
                    if (used >> v) & 1:
                        continue
                    valid = True
                    for nb in assigned_nb[idx]:
                        if not (adj[mapping[nb]] >> v) & 1:
                            valid = False
                            break
                    if valid:
                        mapping[idx] = v
                        used |= (1 << v)
                        if dfs(idx + 1):
                            return True
                        used &= ~(1 << v)
                        mapping[idx] = -1
                return False

            if dfs(0):
                edges = []
                for u, v in Q3_EDGES:
                    edges.append((mapping[u], mapping[v]))
                return edges
        return None

    def _make_random_graph(p, seed):
        rng = random.Random(seed)
        adj = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj

    def _make_complete_bipartite(a, b):
        adj = [0] * n
        for u in range(a):
            for v in range(a, a + b):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        return adj

    def _make_half_graph(a, b):
        # Complete bipartite plus all edges within each part (two cliques + complete join)
        adj = [0] * n
        for u in range(a):
            for v in range(a, a + b):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        for u in range(a):
            for v in range(u + 1, a):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        for u in range(a, a + b):
            for v in range(u + 1, a + b):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        return adj

    # ---- Build a dense Q3-free graph via greedy removal from dense starts ----
    def remove_q3s(adj, rng, max_iter=300):
        m = _count_edges(adj)
        for _ in range(max_iter):
            if time.time() > deadline - 4.0:
                break
            q3 = find_one_q3(adj)
            if q3 is None:
                break
            # remove a random edge from the found Q3
            u, v = rng.choice(q3)
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            m -= 1
        return adj, m

    def try_add_edges(adj, rng):
        """Greedy add any edge that keeps graph Q3-free."""
        m = _count_edges(adj)
        changed = True
        while changed and time.time() < deadline - 3.0:
            changed = False
            cand = []
            for u in range(n):
                for v in range(u + 1, n):
                    if not (adj[u] >> v) & 1:
                        cand.append((u, v))
            rng.shuffle(cand)
            for u, v in cand:
                if time.time() > deadline - 3.0:
                    break
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                m += 1
                if find_one_q3(adj) is None:
                    changed = True
                else:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    m -= 1
        return adj, m

    def swap_search(adj, rng, max_iter=4000):
        """Try edge swaps (remove e, add f) that keep Q3-free and increase density."""
        m = _count_edges(adj)
        for _ in range(max_iter):
            if time.time() > deadline - 3.0:
                break
            present = []
            absent = []
            for u in range(n):
                for v in range(u + 1, n):
                    if (adj[u] >> v) & 1:
                        present.append((u, v))
                    else:
                        absent.append((u, v))
            if not present or not absent:
                break
            u1, v1 = rng.choice(present)
            u2, v2 = rng.choice(absent)
            if (u1, v1) == (u2, v2):
                continue
            # remove (u1,v1)
            adj[u1] &= ~(1 << v1)
            adj[v1] &= ~(1 << u1)
            # add (u2,v2)
            adj[u2] |= (1 << v2)
            adj[v2] |= (1 << u2)
            if find_one_q3(adj) is None:
                m  # same count
            else:
                # revert
                adj[u2] &= ~(1 << v2)
                adj[v2] &= ~(1 << u2)
                adj[u1] |= (1 << v1)
                adj[v1] |= (1 << u1)
        return adj, m

    # ---- Main: try multiple dense starting points ----
    best_m = 0
    best_edges = []

    starts = []
    for p in (0.85, 0.90, 0.93, 0.95, 0.97):
        for seed in range(6):
            starts.append(("rand", _make_random_graph(p, seed * 1000 + int(p * 100))))
    starts.append(("K3,13", _make_complete_bipartite(3, 13)))
    starts.append(("half 4,12", _make_half_graph(4, 12)))
    starts.append(("half 5,11", _make_half_graph(5, 11)))

    rng = random.Random(12345)
    for name, adj in starts:
        if time.time() > deadline - 6.0:
            break
        adj, m = remove_q3s(adj, rng, max_iter=400)
        if find_one_q3(adj) is None and m > best_m:
            best_m = m
            best_edges = _get_edge_list(adj)
        # refine: add edges then swap
        if time.time() < deadline - 4.0:
            adj, m2 = try_add_edges(adj, rng)
            if find_one_q3(adj) is None and m2 > best_m:
                best_m = m2
                best_edges = _get_edge_list(adj)
        if time.time() < deadline - 3.0:
            adj, m3 = swap_search(adj, rng, max_iter=1500)
            if find_one_q3(adj) is None and m3 > best_m:
                best_m = m3
                best_edges = _get_edge_list(adj)

    # Final safety: ensure returned graph is Q3-free
    if best_edges:
        adj = [0] * n
        for u, v in best_edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        if find_one_q3(adj) is not None:
            # fallback to a known Q3-free graph
            adj = _make_complete_bipartite(3, 13)
            best_edges = _get_edge_list(adj)

    return best_edges
