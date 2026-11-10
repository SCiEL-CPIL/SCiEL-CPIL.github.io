import time
import random
from itertools import combinations

_N = 16
_TIME_LIMIT = 115.0


def build_adj(edges, n=_N):
    adj = [0] * n
    for u, v in edges:
        adj[u] |= 1 << v
        adj[v] |= 1 << u
    return adj


def is_triangle_free(adj, n=_N):
    for u in range(n):
        mask = adj[u]
        while mask:
            lsb = mask & -mask
            v = lsb.bit_length() - 1
            mask ^= lsb
            common = adj[v] & adj[u]
            if common & ~(1 << u) & ~(1 << v):
                return False
    return True


def is_not_3_colorable(adj, n=_N):
    """Exact backtracking 3-colouring. Returns True iff the graph is NOT 3-colourable."""
    memo = set()

    def dfs(uncolored_mask, colors):
        if uncolored_mask == 0:
            return False
        key = (uncolored_mask, tuple(colors))
        if key in memo:
            return True
        # pick the vertex with the most already-coloured neighbours
        best_v = -1
        best_score = -1
        m = uncolored_mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            m ^= lsb
            score = 0
            nb = adj[v]
            while nb:
                lsb2 = nb & -nb
                w = lsb2.bit_length() - 1
                nb ^= lsb2
                if colors[w] != -1:
                    score += 1
            if score > best_score:
                best_score = score
                best_v = v
        v = best_v
        forbidden = 0
        nb = adj[v]
        while nb:
            lsb2 = nb & -nb
            w = lsb2.bit_length() - 1
            nb ^= lsb2
            if colors[w] != -1:
                forbidden |= 1 << colors[w]
        ok = True
        for c in range(3):
            if not (forbidden & (1 << c)):
                colors[v] = c
                if not dfs(uncolored_mask & ~(1 << v), colors):
                    ok = False
                    break
                colors[v] = -1
        if ok:
            memo.add(key)
        return ok

    return dfs((1 << n) - 1, [-1] * n)


def greedy_3_colorable(adj, n=_N, max_trials=6):
    for _ in range(max_trials):
        order = list(range(n))
        random.shuffle(order)
        colors = [-1] * n
        for node in order:
            used = 0
            nb = adj[node]
            while nb:
                lsb = nb & -nb
                w = lsb.bit_length() - 1
                nb ^= lsb
                if colors[w] != -1:
                    used |= 1 << colors[w]
            placed = False
            for c in range(3):
                if not (used & (1 << c)):
                    colors[node] = c
                    placed = True
                    break
            if not placed:
                break
        else:
            return True
    return False


def is_not_3_colorable_fast(adj, n=_N):
    if greedy_3_colorable(adj, n, max_trials=4):
        return False
    return is_not_3_colorable(adj, n)


def build_groetzsch_edges():
    edges = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 0]]
    for u, v in [[0, 1], [1, 2], [2, 3], [3, 4], [4, 0]]:
        edges.append([5 + u, v])
        edges.append([5 + v, u])
    for u in range(5):
        edges.append([10, 5 + u])
    return edges


def build_clebsch_edges():
    """
    Build the Clebsch graph: complement of the 4x4 rook graph.
    Vertices 0..15 mapped to (i,j) with i,j in {0,1,2,3}.
    Two vertices are adjacent in Clebsch iff they share NO coordinate.
    This gives 16 vertices, 40 edges, triangle-free, 5-chromatic.
    """
    edges = []
    for a in range(16):
        i1, j1 = a // 4, a % 4
        for b in range(a + 1, 16):
            i2, j2 = b // 4, b % 4
            if i1 != i2 and j1 != j2:
                edges.append([a, b])
    return edges


def generate_blowup_distributions(num_extra, num_original):
    dists = []

    def rec(rem, idx, current):
        if idx == num_original - 1:
            current.append(rem)
            dists.append(tuple(current))
            current.pop()
            return
        for i in range(rem + 1):
            current.append(i)
            rec(rem - i, idx + 1, current)
            current.pop()

    rec(num_extra, 0, [])
    return dists


def _trial(dist, start_time):
    n = _N
    num_original = 11
    g_adj = build_adj(build_groetzsch_edges(), num_original)

    sizes = [1 + d for d in dist]
    orig_map = [0] * n
    idx = 0
    for i in range(num_original):
        for _ in range(sizes[i]):
            orig_map[idx] = i
            idx += 1

    # precompute vertex lists per original vertex
    verts = [[] for _ in range(num_original)]
    for v in range(n):
        verts[orig_map[v]].append(v)

    adj = [0] * n
    edges = []

    # add original edges (blow-up: complete between parts)
    for i in range(num_original):
        for j in range(i + 1, num_original):
            if g_adj[i] & (1 << j):
                for u in verts[i]:
                    for v in verts[j]:
                        adj[u] |= 1 << v
                        adj[v] |= 1 << u
                        edges.append([u, v])

    # saturate: add every non-edge whose endpoints have no common neighbour
    for u in range(n):
        for v in range(u + 1, n):
            if not (adj[u] & (1 << v)):
                if (adj[u] & adj[v]) == 0:
                    adj[u] |= 1 << v
                    adj[v] |= 1 << u
                    edges.append([u, v])

    if not is_triangle_free(adj, n):
        return None, 0

    # greedily add remaining triangle-free edges that keep non-3-colourability
    candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if not (adj[u] & (1 << v)):
                if (adj[u] & adj[v]) == 0:
                    candidates.append((u, v))

    random.shuffle(candidates)
    for u, v in candidates:
        if time.time() - start_time > _TIME_LIMIT:
            break
        adj[u] |= 1 << v
        adj[v] |= 1 << u
        edges.append([u, v])
        if is_not_3_colorable_fast(adj, n):
            continue
        # revert
        adj[u] ^= 1 << v
        adj[v] ^= 1 << u
        edges.pop()

    if is_not_3_colorable_fast(adj, n):
        return edges, len(edges)
    return None, 0


def _clebsch_polish(start_time):
    """
    Start from the Clebsch graph and try to add edges while keeping
    triangle-free and non-3-colorable.
    """
    n = _N
    edges = build_clebsch_edges()
    adj = build_adj(edges, n)

    if not is_triangle_free(adj, n):
        return None, 0
    if not is_not_3_colorable_fast(adj, n):
        return None, 0

    best_edges = list(edges)
    best_count = len(edges)

    improved = True
    while improved and time.time() < start_time + _TIME_LIMIT:
        improved = False
        candidates = []
        for u in range(n):
            for v in range(u + 1, n):
                if not (adj[u] & (1 << v)):
                    if (adj[u] & adj[v]) == 0:
                        candidates.append((u, v))
        random.shuffle(candidates)
        for u, v in candidates:
            if time.time() > start_time + _TIME_LIMIT:
                break
            adj[u] |= 1 << v
            adj[v] |= 1 << u
            edges.append([u, v])
            if is_not_3_colorable_fast(adj, n):
                best_count = len(edges)
                best_edges = list(edges)
                improved = True
                break
            else:
                adj[u] ^= 1 << v
                adj[v] ^= 1 << u
                edges.pop()

    if is_not_3_colorable_fast(adj, n):
        return best_edges, best_count
    return None, 0


def construct():
    start_time = time.time()
    deadline = start_time + _TIME_LIMIT

    # Try Clebsch-based polish first (40 edges base, likely high ceiling)
    clebsch_edges, clebsch_count = _clebsch_polish(start_time)

    best_edges = clebsch_edges if clebsch_edges else build_groetzsch_edges()
    best_count = clebsch_count if clebsch_edges else len(build_groetzsch_edges())

    # Also try Grötzsch blow-up approach
    num_extra = _N - 11
    dists = generate_blowup_distributions(num_extra, 11)
    random.shuffle(dists)

    for dist in dists:
        if time.time() > deadline:
            break
        edges, cnt = _trial(dist, start_time)
        if edges is not None and cnt > best_count:
            best_count = cnt
            best_edges = edges

    # final polish: try adding triangle-free edges to the best graph found
    if time.time() < deadline:
        adj = build_adj(best_edges, _N)
        cur = list(best_edges)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            candidates = []
            for u in range(_N):
                for v in range(u + 1, _N):
                    if not (adj[u] & (1 << v)):
                        if (adj[u] & adj[v]) == 0:
                            candidates.append((u, v))
            random.shuffle(candidates)
            for u, v in candidates:
                if time.time() > deadline:
                    break
                adj[u] |= 1 << v
                adj[v] |= 1 << u
                cur.append([u, v])
                if is_not_3_colorable_fast(adj, _N):
                    best_count = len(cur)
                    best_edges = list(cur)
                    improved = True
                else:
                    adj[u] ^= 1 << v
                    adj[v] ^= 1 << u
                    cur.pop()

    return best_edges
