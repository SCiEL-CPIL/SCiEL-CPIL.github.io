"""
C6-free graph construction on 40 vertices.
Bitset adjacency + conflict-aware greedy fill + 1-edge-swap local search.
Multiple random restarts, keep best, explicit deadline guard.
"""
import time
import random

def construct():
    n = 40
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    def time_ok():
        return time.time() < deadline

    # ---------- Bitset adjacency ----------
    # adj[i] is an int bitmask of neighbors of vertex i

    def popcount(x):
        return x.bit_count()

    def make_adj(edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def edges_from_adj(adj):
        out = []
        for u in range(n):
            mask = adj[u]
            while mask:
                bit = mask & (-mask)
                v = bit.bit_length() - 1
                if v > u:
                    out.append([u, v])
                mask ^= bit
        return out

    # ---------- Fast C6 check ----------
    def would_create_c6(adj, u, v):
        """Does adding edge (u,v) create a C6?
        A C6 using (u,v): u-a-b-c-d-v, all 6 distinct.
        Need a path a-b-c-d of length 3 in G with a in N(u), d in N(v).
        a != d, and b,c distinct from a,d,u,v.
        """
        Nu = adj[u] & ~(1 << v)
        Nv = adj[v] & ~(1 << u)

        # Enumerate a in N(u), d in N(v)
        # For each (a,d) with a!=d, check if there's a length-2 path a-b-c-d
        # i.e., exists b in N(a)\{d,u,v}, c in N(d)\{a,u,v} with b~c and b!=c.
        # Equivalently: (N(a) & N(d)) minus {u,v,a,d} ... no, that's common neighbors.
        # a-b-c-d: b adjacent to a, c adjacent to d, b adjacent to c.
        # So for b in N(a), c in N(d), need b~c.
        # Let S = N(a) & ~(1<<u) & ~(1<<v) & ~(1<<d)  (candidates for b)
        # Let T = N(d) & ~(1<<u) & ~(1<<v) & ~(1<<a)  (candidates for c)
        # Need exists b in S, c in T with b~c and b!=c.
        # i.e., exists b in S such that (N(b) & T) minus {b} is non-empty.

        # Choose smaller of Nu, Nv to iterate over for 'a' side
        if popcount(Nu) <= popcount(Nv):
            mask_a = Nu
            mask_d_full = Nv
        else:
            # swap roles: a from Nv, d from Nu
            mask_a = Nv
            mask_d_full = Nu
            # but then the path is v-d-c-b-a-u? Let me just keep a in Nu, d in Nv
            # and iterate the smaller as the outer loop variable
            # Actually the roles of a and d are symmetric in the length-3 path check
            # (a-b-c-d vs d-c-b-a). So I can iterate a over the smaller set.
            mask_a, mask_d_full = Nv, Nu

        while mask_a:
            bit_a = mask_a & (-mask_a)
            a = bit_a.bit_length() - 1
            mask_a ^= bit_a

            Na = adj[a] & ~(1 << u) & ~(1 << v)
            mask_d = mask_d_full & ~(1 << a)
            while mask_d:
                bit_d = mask_d & (-mask_d)
                d = bit_d.bit_length() - 1
                mask_d ^= bit_d

                # S = candidates for b: N(a) minus {u,v,d}
                S = Na & ~(1 << d)
                if S == 0:
                    continue
                # T = candidates for c: N(d) minus {u,v,a}
                T = adj[d] & ~(1 << u) & ~(1 << v) & ~(1 << a)
                if T == 0:
                    continue
                # Need exists b in S such that (N(b) & T) has a vertex != b
                mask_b = S
                while mask_b:
                    bit_b = mask_b & (-mask_b)
                    b = bit_b.bit_length() - 1
                    mask_b ^= bit_b
                    if adj[b] & T & ~(1 << b):
                        return True
        return False

    # ---------- Conflict score for greedy ordering ----------
    def conflict_score(adj, u, v):
        """Approximate number of length-3 paths between u and v (potential C6s
        that edge (u,v) would close). Lower is better (less conflict)."""
        Nu = adj[u] & ~(1 << v)
        Nv = adj[v] & ~(1 << u)
        if popcount(Nu) > popcount(Nv):
            Nu, Nv = Nv, Nu
        score = 0
        mask_a = Nu
        while mask_a:
            bit_a = mask_a & (-mask_a)
            a = bit_a.bit_length() - 1
            mask_a ^= bit_a
            Na = adj[a] & ~(1 << u) & ~(1 << v)
            # For each d in Nv, count common neighbors of a and d (as b~c candidates)
            # Approximate: count of d in Nv such that (N(a) & N(d)) is non-empty
            mask_d = Nv & ~(1 << a)
            while mask_d:
                bit_d = mask_d & (-mask_d)
                d = bit_d.bit_length() - 1
                mask_d ^= bit_d
                if Na & adj[d] & ~(1 << a) & ~(1 << d):
                    score += 1
        return score

    # ---------- Greedy fill with conflict-aware ordering ----------
    def greedy_fill(adj, edge_set, edges, max_passes=5):
        for _ in range(max_passes):
            if not time_ok():
                break
            # Build candidate list of non-edges
            candidates = []
            for u in range(n):
                for v in range(u + 1, n):
                    if not (adj[u] & (1 << v)):
                        candidates.append((u, v))
            if not candidates:
                break
            # Score and sort by conflict (ascending)
            scored = []
            for u, v in candidates:
                s = conflict_score(adj, u, v)
                scored.append((s, u, v))
            scored.sort()
            added_any = False
            for s, u, v in scored:
                if not time_ok():
                    break
                if adj[u] & (1 << v):
                    continue
                if not would_create_c6(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edge_set.add((u, v))
                    edges.append([u, v])
                    added_any = True
            if not added_any:
                break

    # ---------- 1-edge-swap local search ----------
    def local_swap(adj, edge_set, edges, iters=200):
        for _ in range(iters):
            if not time_ok():
                break
            cur = len(edges)
            if cur == 0:
                break
            improved = False

            # First try a pure add
            for _ in range(25):
                if not time_ok():
                    break
                u = random.randrange(n)
                v = random.randrange(n)
                if u == v:
                    continue
                if u > v:
                    u, v = v, u
                if (u, v) in edge_set:
                    continue
                if not would_create_c6(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edge_set.add((u, v))
                    edges.append([u, v])
                    improved = True
                    break
            if improved:
                continue

            # Stuck: try a 1-swap
            idx = random.randrange(cur)
            ru, rv = edges[idx]
            key = (min(ru, rv), max(ru, rv))
            adj[ru] &= ~(1 << rv)
            adj[rv] &= ~(1 << ru)
            edge_set.discard(key)
            edges.pop(idx)
            swapped = False
            for _ in range(35):
                if not time_ok():
                    break
                u = random.randrange(n)
                v = random.randrange(n)
                if u == v:
                    continue
                if u > v:
                    u, v = v, u
                if (u, v) in edge_set:
                    continue
                if not would_create_c6(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edge_set.add((u, v))
                    edges.append([u, v])
                    swapped = True
                    break
            if not swapped:
                adj[ru] |= (1 << rv)
                adj[rv] |= (1 << ru)
                edge_set.add(key)
                edges.append([ru, rv])

    # ---------- Single run ----------
    def run(seed):
        random.seed(seed)
        adj = [0] * n
        edge_set = set()
        edges = []
        # Start with 8 disjoint K5s
        for k in range(8):
            for a in range(5):
                for b in range(a + 1, 5):
                    u, v = 5 * k + a, 5 * k + b
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edge_set.add((u, v))
                    edges.append([u, v])
        # Greedy fill
        greedy_fill(adj, edge_set, edges)
        # Local search
        local_swap(adj, edge_set, edges)
        return edges

    # ---------- Multi-restart, keep best ----------
    best_edges = None
    best_count = 0
    seeds = [42, 1337, 7, 99, 2024, 55, 123, 888, 314, 271]
    for i, seed in enumerate(seeds):
        if not time_ok():
            break
        edges = run(seed)
        cnt = len(edges)
        if cnt > best_count:
            best_count = cnt
            best_edges = edges
        # If we hit the known best, stop early
        if best_count >= 105:
            break

    if best_edges is None:
        # Fallback: 8 disjoint K5s
        best_edges = []
        for k in range(8):
            for a in range(5):
                for b in range(a + 1, 5):
                    best_edges.append([5 * k + a, 5 * k + b])

    return best_edges
