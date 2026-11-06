"""Suite task: dense triangle-free 4-chromatic graph on 16 vertices."""
import time

N = 16
MAX_EDGES = N * (N - 1) // 2  # 120
# bit index for edge (u,v), u<v
def _eidx(u, v):
    if u > v:
        u, v = v, u
    return u * N - (u + 1) // 2 + (v - u - 1)

# Precompute all edges and adjacency-bitset helpers
_ALL_EDGES = []
for u in range(N):
    for v in range(u + 1, N):
        _ALL_EDGES.append((u, v))
EDGE_INDEX = {}
for i, (u, v) in enumerate(_ALL_EDGES):
    EDGE_INDEX[(u, v)] = i

# For each vertex, the set of neighbors that form a triangle if added
# tri_mask[u] bit i set means: adding edge i creates a triangle involving u
# Actually: for candidate edge (u,v), triangle exists if common neighbor w.
# We precompute for each pair (u,v) the list of w adjacent to both -> but adjacency changes.
# Use bitsets: adj[u] is int bitmask of neighbors.
# triangle for (u,v): adj[u] & adj[v] != 0

# Precompute edge endpoint arrays
_EU = [e[0] for e in _ALL_EDGES]
_EV = [e[1] for e in _ALL_EDGES]


def _is_3_colorable(adj):
    """adj: list of int bitmasks. Return True if 3-colorable."""
    # assign colors 0,1,2
    color = [-1] * N
    # Try to color greedily with backtracking on smallest uncolored degree
    # Use recursion
    def backtrack(pos):
        # find uncolored vertex with fewest available colors
        best_v = -1
        best_opts = 8  # 3 bits
        for v in range(N):
            if color[v] != -1:
                continue
            forbidden = 0
            nb = adj[v]
            # compute forbidden colors from neighbors
            f0 = f1 = f2 = False
            w = nb
            while w:
                lsb = w & (-w)
                u = lsb.bit_length() - 1
                c = color[u]
                if c == 0:
                    f0 = True
                elif c == 1:
                    f1 = True
                else:
                    f2 = True
                w -= lsb
            opts = 0
            if not f0:
                opts |= 1
            if not f1:
                opts |= 2
            if not f2:
                opts |= 4
            if opts == 0:
                return False
            if opts < best_opts:
                best_opts = opts
                best_v = v
                if opts == 1:
                    break
        if best_v == -1:
            return True
        opts = best_opts
        v = best_v
        while opts:
            lsb = opts & (-opts)
            c = (lsb.bit_length() - 1)  # 0,1,2
            color[v] = c
            if backtrack(pos + 1):
                return True
            color[v] = -1
            opts -= lsb
        return False
    return backtrack(0)


def _greedy_fill(order, adj, deadline):
    """Try to add edges in given order if triangle-free. Return (adj, count)."""
    cnt = 0
    for i in order:
        u, v = _EU[i], _EV[i]
        if (adj[u] >> v) & 1:
            continue
        if (adj[u] & adj[v]) != 0:
            continue
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        cnt += 1
    return adj, cnt


def _random_order():
    import random
    idx = list(range(MAX_EDGES))
    random.shuffle(idx)
    return idx


def construct():
    start = time.time()
    deadline = start + 145.0  # safety margin before 150s kill
    best_adj = None
    best_cnt = -1

    import random

    # We need a 4-chromatic triangle-free graph. The greedy fill gives a maximal
    # triangle-free graph; check if it's not 3-colorable.
    # To bias toward 4-chromatic, we can seed with a known 4-chromatic core (Grötzsch)
    # and then greedily add more edges.

    # Grötzsch graph (Mycielski of C5) on vertices 0..10
    def make_grotzsch():
        adj = [0] * N
        for i in range(5):
            j = (i + 1) % 5
            adj[i] |= (1 << j)
            adj[j] |= (1 << i)
        for i in range(5):
            j = (i + 1) % 5
            adj[i] |= (1 << (5 + j))
            adj[5 + j] |= (1 << i)
            adj[j] |= (1 << (5 + i))
            adj[5 + i] |= (1 << j)
        for i in range(5):
            adj[5 + i] |= (1 << 10)
            adj[10] |= (1 << (5 + i))
        return adj

    def adj_to_edges(adj):
        edges = []
        for u in range(N):
            nb = adj[u]
            while nb:
                lsb = nb & (-nb)
                v = lsb.bit_length() - 1
                if v > u:
                    edges.append([u, v])
                nb -= lsb
        return edges

    # First try a few seeded restarts with Grötzsch core
    base_grotzsch = make_grotzsch()

    def try_seed(base_adj):
        nonlocal best_adj, best_cnt
        # copy
        adj = list(base_adj)
        # random order of remaining edges
        present = set()
        for u in range(N):
            nb = adj[u]
            while nb:
                lsb = nb & (-nb)
                v = lsb.bit_length() - 1
                if v > u:
                    present.add((u, v))
                nb -= lsb
        order = [i for i in range(MAX_EDGES) if _ALL_EDGES[i] not in present]
        random.shuffle(order)
        adj, cnt = _greedy_fill(order, adj, deadline)
        if cnt > best_cnt and not _is_3_colorable(adj):
            best_cnt = cnt
            best_adj = list(adj)

    # Do a number of seeded restarts
    restarts_done = 0
    while time.time() < deadline - 2.0:
        try_seed(base_grotzsch)
        restarts_done += 1
        # periodically do pure random restarts too
        if restarts_done % 5 == 0:
            adj = [0] * N
            order = _random_order()
            adj, cnt = _greedy_fill(order, adj, deadline)
            if cnt > best_cnt and not _is_3_colorable(adj):
                best_cnt = cnt
                best_adj = list(adj)

    # Also try local improvement: start from best, try to add edges via random shuffles
    if best_adj is None:
        # fallback: return Grötzsch (known 4-chromatic, 11 edges)
        return adj_to_edges(base_grotzsch)

    # Local search: try to increase edge count while keeping triangle-free and 4-chromatic
    # by randomly removing an edge then re-filling
    while time.time() < deadline - 1.0:
        adj = list(best_adj)
        # pick a random edge to remove, then refill
        edges = adj_to_edges(adj)
        if not edges:
            break
        random.shuffle(edges)
        improved = False
        for (u, v) in edges[:5]:
            adj2 = list(adj)
            adj2[u] &= ~(1 << v)
            adj2[v] &= ~(1 << u)
            present = set()
            for uu in range(N):
                nb = adj2[uu]
                while nb:
                    lsb = nb & (-nb)
                    vv = lsb.bit_length() - 1
                    if vv > uu:
                        present.add((uu, vv))
                    nb -= lsb
            order = [i for i in range(MAX_EDGES) if _ALL_EDGES[i] not in present]
            random.shuffle(order)
            adj2, cnt2 = _greedy_fill(order, adj2, deadline)
            if cnt2 > best_cnt and not _is_3_colorable(adj2):
                best_cnt = cnt2
                best_adj = list(adj2)
                improved = True
                break
        if not improved and time.time() > deadline - 3.0:
            break

    return adj_to_edges(best_adj)
