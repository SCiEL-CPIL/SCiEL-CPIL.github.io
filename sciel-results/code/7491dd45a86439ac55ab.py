import random
import time
import math

N = 18
ALLMASK = (1 << N) - 1

def is_k4_free(adj):
    """Fast K4-free check using bitmasks."""
    for a in range(N):
        for b in range(a + 1, N):
            if not (adj[a] >> b) & 1:
                continue
            common_ab = adj[a] & adj[b]
            common_ab &= ~((1 << (b + 1)) - 1)
            while common_ab:
                lsb = common_ab & -common_ab
                c = lsb.bit_length() - 1
                common_abc = adj[a] & adj[b] & adj[c]
                common_abc &= ~((1 << (c + 1)) - 1)
                if common_abc:
                    return False
                common_ab ^= lsb
    return True

def k4_violations(adj):
    """Return list of K4 vertex-tuples (a<b<c<d) that form a clique."""
    out = []
    for a in range(N):
        for b in range(a + 1, N):
            if not (adj[a] >> b) & 1:
                continue
            common_ab = adj[a] & adj[b]
            common_ab &= ~((1 << (b + 1)) - 1)
            while common_ab:
                lsb = common_ab & -common_ab
                c = lsb.bit_length() - 1
                common_abc = adj[a] & adj[b] & adj[c]
                common_abc &= ~((1 << (c + 1)) - 1)
                while common_abc:
                    lsb2 = common_abc & -common_abc
                    d = lsb2.bit_length() - 1
                    out.append((a, b, c, d))
                    common_abc ^= lsb2
                common_ab ^= lsb
    return out

def largest_triangle_free(adj, deadline, limit=None):
    """
    Exact largest triangle-free induced subgraph via branch-and-bound on the
    independent set of the line graph of the triangle edges.
    Returns the maximum size. limit: if a TF subgraph of size >= limit exists,
    may stop early and return >= limit (used for cheap 'is L >= limit' tests).
    """
    # Enumerate all triangles
    tri_edges = []
    for u in range(N):
        nbrs = adj[u] & ~((1 << (u + 1)) - 1)
        while nbrs:
            lsb = nbrs & -nbrs
            v = lsb.bit_length() - 1
            nbrs ^= lsb
            common = adj[u] & adj[v] & ~((1 << (v + 1)) - 1)
            while common:
                lsb2 = common & -common
                w = lsb2.bit_length() - 1
                common ^= lsb2
                tri_edges.append((u, v, w))
    m = len(tri_edges)
    if m == 0:
        return N

    # Line graph: two triangles conflict if they share a vertex
    conf = [0] * m
    for i in range(m):
        u, v, w = tri_edges[i]
        mask_i = (1 << u) | (1 << v) | (1 << w)
        for j in range(i + 1, m):
            u2, v2, w2 = tri_edges[j]
            if mask_i & ((1 << u2) | (1 << v2) | (1 << w2)):
                conf[i] |= (1 << j)
                conf[j] |= (1 << i)

    # Degree ordering for better B&B
    order = sorted(range(m), key=lambda x: conf[x].bit_count())
    pos = [0] * m
    for idx, t in enumerate(order):
        pos[t] = idx
    # Build position-indexed conflict masks
    pconf = [0] * m
    for i in range(m):
        cm = conf[i]
        pm = 0
        while cm:
            lsb = cm & -cm
            j = lsb.bit_length() - 1
            pm |= (1 << pos[j])
            cm ^= lsb
        pconf[i] = pm
    pconf_pos = [pconf[order[i]] for i in range(m)]
    # vertex mask of each triangle in position order
    tri_vertex = [(1 << u) | (1 << v) | (1 << w) for (u, v, w) in (tri_edges[t] for t in order)]

    best = 0
    if limit is not None and limit <= 0:
        return 0

    # Independent set in line graph (position space)
    def branch(avail, chosen, chosen_vertices, depth):
        nonlocal best
        if time.time() > deadline:
            return
        # Upper bound: chosen + popcount(avail)
        ub = chosen + avail.bit_count()
        if ub <= best:
            return
        if avail == 0:
            if chosen > best:
                best = chosen
                if limit is not None and best >= limit:
                    return
            return
        # Pick vertex with max degree in avail
        # choose the first available vertex (heuristic: highest conflict count)
        # find a vertex in avail with max conflicts within avail
        av = avail
        best_v = -1
        best_deg = -1
        while av:
            lsb = av & -av
            v = lsb.bit_length() - 1
            deg = (pconf_pos[v] & avail).bit_count()
            if deg > best_deg:
                best_deg = deg
                best_v = v
            av ^= lsb
        v = best_v
        # Branch include
        new_avail = avail & ~pconf_pos[v] & ~(1 << v)
        branch(new_avail, chosen + 1, chosen_vertices | tri_vertex[v], depth + 1)
        # Branch exclude
        branch(avail & ~(1 << v), chosen, chosen_vertices, depth + 1)

    branch(ALLMASK, 0, 0, 0)
    return best

def tf_upper_bound(adj):
    """
    Fast upper bound on largest triangle-free subgraph.
    For every triangle, at least one of its 3 vertices must be excluded.
    Greedy set cover to find a small hitting set H; then L <= N - |H|.
    """
    tri_edges = []
    for u in range(N):
        nbrs = adj[u] & ~((1 << (u + 1)) - 1)
        while nbrs:
            lsb = nbrs & -nbrs
            v = lsb.bit_length() - 1
            nbrs ^= lsb
            common = adj[u] & adj[v] & ~((1 << (v + 1)) - 1)
            while common:
                lsb2 = common & -common
                w = lsb2.bit_length() - 1
                common ^= lsb2
                tri_edges.append((u, v, w))
    if not tri_edges:
        return N
    # Greedy hitting set
    remaining = list(tri_edges)
    H = 0
    while remaining:
        # count vertex frequencies
        freq = [0] * N
        for (u, v, w) in remaining:
            freq[u] += 1
            freq[v] += 1
            freq[w] += 1
        # pick vertex covering most
        best_v = -1
        best_c = -1
        for x in range(N):
            if (H >> x) & 1:
                continue
            if freq[x] > best_c:
                best_c = freq[x]
                best_v = x
        if best_v < 0:
            break
        H |= (1 << best_v)
        new_remaining = []
        for (u, v, w) in remaining:
            if (H & ((1 << u) | (1 << v) | (1 << w))) == 0:
                new_remaining.append((u, v, w))
        remaining = new_remaining
    return N - H.bit_count()

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin under 150s

    def adj_to_edges(adj):
        edges = []
        for i in range(N):
            for j in range(i + 1, N):
                if adj[i] & (1 << j):
                    edges.append([i, j])
        return edges

    best_l = N + 1
    best_adj = None

    def update_best(adj, l):
        nonlocal best_l, best_adj
        if l < best_l:
            best_l = l
            best_adj = adj[:]

    random.seed(42)

    candidates = []

    # 4-partite with parts 4,4,5,5
    parts = [list(range(0, 4)), list(range(4, 8)), list(range(8, 13)), list(range(13, 18))]

    for trial in range(20):
        adj = [0] * N
        for i in range(4):
            for j in range(i + 1, 4):
                for u in parts[i]:
                    for v in parts[j]:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
        # Repair K4s
        for _ in range(2000):
            if is_k4_free(adj):
                break
            k4s = k4_violations(adj)
            if not k4s:
                break
            # Pick a random K4, remove a random edge
            a, b, c, d = random.choice(k4s)
            e = random.choice([(a, b), (a, c), (a, d), (b, c), (b, d), (c, d)])
            adj[e[0]] &= ~(1 << e[1])
            adj[e[1]] &= ~(1 << e[0])
        if is_k4_free(adj):
            candidates.append(adj)

    # Turan T(18,3)
    adj_turan = [0] * N
    parts3 = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    for i in range(3):
        for j in range(i + 1, 3):
            for u in parts3[i]:
                for v in parts3[j]:
                    adj_turan[u] |= (1 << v)
                    adj_turan[v] |= (1 << u)
    candidates.append(adj_turan)

    # Random dense repaired
    for _ in range(8):
        adj = [0] * N
        for i in range(N):
            for j in range(i + 1, N):
                if random.random() < 0.65:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        for _ in range(2000):
            if is_k4_free(adj):
                break
            k4s = k4_violations(adj)
            if not k4s:
                break
            a, b, c, d = random.choice(k4s)
            e = random.choice([(a, b), (a, c), (a, d), (b, c), (b, d), (c, d)])
            adj[e[0]] &= ~(1 << e[1])
            adj[e[1]] &= ~(1 << e[0])
        if is_k4_free(adj):
            candidates.append(adj)

    # Evaluate candidates
    for adj in candidates:
        if time.time() > deadline - 20:
            break
        # Use fast upper bound first
        ub = tf_upper_bound(adj)
        if ub <= best_l:
            # can't beat best, but need exact if ub < best? ub is upper bound, so L <= ub <= best, no improvement
            continue
        l = largest_triangle_free(adj, deadline)
        update_best(adj, l)

    # Local search (simulated annealing) with fast incremental checks
    if best_adj is not None and time.time() < deadline - 15:
        current_adj = best_adj[:]
        current_l = best_l
        T = 1.0

        while time.time() < deadline - 2:
            i, j = random.sample(range(N), 2)
            current_adj[i] ^= (1 << j)
            current_adj[j] ^= (1 << i)

            if not is_k4_free(current_adj):
                current_adj[i] ^= (1 << j)
                current_adj[j] ^= (1 << i)
                continue

            # Fast upper bound
            ub = tf_upper_bound(current_adj)
            if ub < current_l:
                # Definitely improved; compute exact
                new_l = largest_triangle_free(current_adj, deadline)
                current_l = new_l
                update_best(current_adj, new_l)
                T *= 0.995
                continue
            if ub <= best_l:
                # Can't beat global best; revert
                current_adj[i] ^= (1 << j)
                current_adj[j] ^= (1 << i)
                continue

            # Need exact to compare
            new_l = largest_triangle_free(current_adj, deadline)

            if new_l < current_l:
                current_l = new_l
                update_best(current_adj, new_l)
            elif new_l == current_l:
                pass
            else:
                delta = new_l - current_l
                if random.random() < math.exp(-delta / T):
                    pass
                else:
                    current_adj[i] ^= (1 << j)
                    current_adj[j] ^= (1 << i)
            T *= 0.998

    if best_adj is None:
        return adj_to_edges(candidates[-1])

    return adj_to_edges(best_adj)
