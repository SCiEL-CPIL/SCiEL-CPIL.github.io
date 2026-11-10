import random
import time
import itertools

N = 20

def _k33_free(adj):
    # adj[u] is a bitmask of neighbors of u
    # K3,3 exists iff there exist two disjoint 3-sets A={a1,a2,a3}, B={b1,b2,b3}
    # such that every ai is adjacent to every bj.
    # Equivalent: exists a vertex u and three neighbors x,y,z of u such that
    # x,y,z have a common neighbor w (w != u) -- because then {u,w} x {x,y,z} is K2,3
    # and combined... actually K3,3 needs 3+3. Let me use: exists 3-set S and 3-set T
    # disjoint, all cross-edges present.
    # Fast check: for each 3-subset T, let C(T) = intersection of adj over T (common neighbors of T).
    # If |C(T)| >= 3 and there exist 3 vertices in C(T) that are pairwise non-adjacent to each other
    # and disjoint from T, then K3,3. But simpler: K3,3 iff exists 3-set T with |C(T)| >= 3
    # where 3 of C(T) are mutually non-adjacent? No.
    # Cleanest: exists two disjoint 3-sets with full bipartite adjacency.
    # For 20 vertices, iterate 3-sets T (1140) and check if any 3-subset of C(T) is
    # independent in the induced graph on C(T) and disjoint from T.
    # Actually we need the 3-set A to be subset of C(T) and A must be an independent set
    # (no edges within A) so that A and T are disjoint parts of K3,3. Wait, K3,3 requires
    # NO edges within A and NO edges within T either? No! K3,3 as a subgraph only requires
    # the 9 cross edges; the 6 internal edges may or may not exist. So we just need
    # two disjoint 3-sets A, T with all 9 cross edges present.
    # So: exists 3-set T, and 3-set A subset of C(T), A disjoint from T.
    # That's it. A can have internal edges, T can have internal edges.
    triples = list(itertools.combinations(range(N), 3))
    for T in triples:
        c = adj[T[0]] & adj[T[1]] & adj[T[2]]
        # remove T vertices from c
        for t in T:
            c &= ~(1 << t)
        # need at least 3 bits set in c
        if (c & (c - 1) & (c - 2) and bin(c).count('1') >= 3):
            # pick any 3 from c -> they form A, all adjacent to all of T
            return False
    return True

def construct():
    start = time.time()
    deadline = start + 120.0
    safety = 8.0  # return 8s before kill

    rng = random.Random(12345)

    # Greedy + local search using bitmasks.
    # State: adj list of 20 ints (bitmasks).
    def can_add(adj, u, v):
        # Adding edge u-v: check it doesn't create K3,3.
        # A K3,3 using new edge (u,v) must have u and v in opposite parts,
        # and 2 more neighbors of u in one part, 2 more neighbors of v in other part,
        # with those 2+2 fully cross-adjacent plus the new edge connections.
        # Simpler: just test full K3,3 on the candidate graph for speed via targeted check.
        # We'll do: for the new edge, any K3,3 containing it has u in part X, v in part Y,
        # X = {u, x1, x2}, Y = {v, y1, y2}, where x1,x2 in N(u), y1,y2 in N(v),
        # and all cross edges among {x1,x2}x{v,y1,y2} and {u,x1,x2}x{y1,y2} present.
        # So x1,x2 must be in N(u) AND N(v) AND N(y1) AND N(y2) for some y1,y2 in N(v).
        # And y1,y2 must be in N(v) AND N(u) AND N(x1) AND N(x2).
        # Equivalent: x1,x2 in C = N(u) & N(v) & (common neighbors), y1,y2 in same.
        # Let S = N(u) & N(v) (common neighbors of u and v). Then we need 2 vertices x1,x2 in S
        # and 2 vertices y1,y2 in S such that... wait they must be cross-adjacent.
        # This is getting complex. Use the global check but it's O(1140 * small).
        # For a candidate, just run a fast K3,3 test.
        return True  # placeholder; we use global validation

    # We'll maintain adj and validate with a fast full check.
    def is_k33_free(adj):
        # Fast: for each 3-set T, common neighbors c; need |c \ T| >= 3.
        # Precompute triples once.
        for (a, b, c) in _triples:
            m = adj[a] & adj[b] & adj[c]
            m &= ~(1 << a) & ~(1 << b) & ~(1 << c)
            # popcount >= 3 ?
            mm = m
            cnt = 0
            while mm:
                mm &= mm - 1
                cnt += 1
                if cnt >= 3:
                    return False
        return True

    _triples = list(itertools.combinations(range(N), 3))

    # Greedy construction: start empty, add edges in random order, keep if K3,3-free.
    def greedy():
        adj = [0] * N
        edges = list(itertools.combinations(range(N), 2))
        rng.shuffle(edges)
        for (u, v) in edges:
            adj[u] |= 1 << v
            adj[v] |= 1 << u
            if not is_k33_free(adj):
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
        return adj

    def edge_count(adj):
        return sum(bin(x).count('1') for x in adj) // 2

    # Local search: try flipping edges (add or remove) to increase count while staying K3,3-free.
    def local_search(adj, max_iters):
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            # try adding random missing edges
            for _ in range(500):
                u = rng.randrange(N)
                v = rng.randrange(N)
                if u == v:
                    continue
                if adj[u] & (1 << v):
                    continue
                adj[u] |= 1 << v
                adj[v] |= 1 << u
                if is_k33_free(adj):
                    improved = True
                else:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
            # try removing a random edge and re-adding others (repair)
            for _ in range(200):
                u = rng.randrange(N)
                v = rng.randrange(N)
                if u == v or not (adj[u] & (1 << v)):
                    continue
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                # now try to add a few edges
                for _ in range(10):
                    a = rng.randrange(N)
                    b = rng.randrange(N)
                    if a == b or (adj[a] & (1 << b)):
                        continue
                    adj[a] |= 1 << b
                    adj[b] |= 1 << a
                    if not is_k33_free(adj):
                        adj[a] &= ~(1 << b)
                        adj[b] &= ~(1 << a)
                # keep the removal only if we can't easily restore; we just leave it
                improved = True
        return adj

    best = None
    best_count = -1

    def consider(adj):
        nonlocal best, best_count
        if not is_k33_free(adj):
            return
        c = edge_count(adj)
        if c > best_count:
            best_count = c
            best = [adj[i] for i in range(N)]

    # Phase 1: several greedy starts
    while time.time() < deadline - safety:
        adj = greedy()
        consider(adj)
        if time.time() > deadline - safety:
            break

    # Phase 2: local search from best
    if best is not None and time.time() < deadline - safety:
        adj = [best[i] for i in range(N)]
        adj = local_search(adj, max_iters=200)
        consider(adj)

    # Phase 3: more local search rounds from best
    while time.time() < deadline - safety:
        adj = [best[i] for i in range(N)]
        adj = local_search(adj, max_iters=100)
        consider(adj)

    if best is None:
        # fallback: K2,18
        return [[a, b] for a in range(2) for b in range(2, 20)]

    # Convert bitmask to edge list
    edges = []
    for u in range(N):
        m = best[u]
        while m:
            lsb = m & (-m)
            v = lsb.bit_length() - 1
            if u < v:
                edges.append([u, v])
            m &= m - 1
    return edges
