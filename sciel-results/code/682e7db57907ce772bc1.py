"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

N = P['n']

# ---- static graph data (built once at import) ----
_adj_list = [[] for _ in range(N)]
_adj_mask = [0] * N
for _u, _v in P['edges']:
    _adj_list[_u].append(_v)
    _adj_list[_v].append(_u)
    _adj_mask[_u] |= 1 << _v
    _adj_mask[_v] |= 1 << _u

# vertex ordering for backtracking: descending degree, then index (deterministic)
_ORDER = sorted(range(N), key=lambda v: (-len(_adj_list[v]), v))
_INV_ORDER = [0] * N
for _i, _v in enumerate(_ORDER):
    _INV_ORDER[_v] = _i

# precompute, for each color-pair (a,b), the edge_mask[u] = edges of u that stay inside a∪b
_pair_edge_mask = [[None] * N for _ in range(N)]
_pair_vertex_count = [[0] * N for _ in range(N)]
for _a in range(N):
    for _b in range(_a + 1, N):
        em = [0] * N
        vc = 0
        for _u in range(N):
            m = _adj_mask[_u] & ((1 << _a) | (1 << _b))
            em[_u] = m
            vc += ((1 << _u) & ((1 << _a) | (1 << _b))) >> _u
        _pair_edge_mask[_a][_b] = em
        _pair_edge_mask[_b][_a] = em
        _pair_vertex_count[_a][_b] = vc
        _pair_vertex_count[_b][_a] = vc


def _components_and_edges(mask, edge_mask):
    """Return (num_components, num_edges) of the subgraph induced by 'mask'
    where edge_mask[u] holds the in-pair adjacency bits of u."""
    edges = 0
    for u in range(N):
        if (mask >> u) & 1:
            edges += bin(edge_mask[u] & mask).count('1')
    edges //= 2
    comps = 0
    remaining = mask
    while remaining:
        comps += 1
        # BFS from lowest set bit
        lsb = remaining & (-remaining)
        stack = lsb
        remaining ^= lsb
        while stack:
            l = stack & (-stack)
            stack ^= l
            u = l.bit_length() - 1
            nb = edge_mask[u] & mask & ~remaining
            remaining ^= nb
            stack |= nb
    return comps, edges


def _is_forest(color_mask, edge_mask, vertex_count):
    """Bicolored induced subgraph is a forest iff edges == vertices - components."""
    comps, edges = _components_and_edges(color_mask, edge_mask, vertex_count)
    return edges == vertex_count - comps


def _valid_full(color_mask, k):
    for a in range(k):
        for b in range(a + 1, k):
            if not _is_forest(color_mask[a] | color_mask[b],
                              _pair_edge_mask[a][b],
                              _pair_vertex_count[a][b]):
                return False
    return True


def _backtrack_4color(order):
    """Find a valid 4-coloring (colors 0..3) via backtracking on 'order'."""
    color = [-1] * N
    color_mask = [0, 0, 0, 0]

    def bt(idx):
        if idx == N:
            return True
        v = order[idx]
        for c in range(4):
            if _adj_mask[v] & color_mask[c]:
                continue
            # assign
            color[v] = c
            color_mask[c] |= 1 << v
            ok = True
            for c2 in range(4):
                if c2 == c:
                    continue
                cm = color_mask[c] | color_mask[c2]
                if cm == 0:
                    continue
                if not _is_forest(cm, _pair_edge_mask[c][c2], _pair_vertex_count[c][c2]):
                    ok = False
                    break
            if ok and bt(idx + 1):
                return True
            color[v] = -1
            color_mask[c] ^= 1 << v
        return False

    if bt(0):
        return color
    return None


def _reduce(color, deadline):
    """Reduce distinct colors by single-vertex moves that eliminate a color class."""
    color = color[:]
    k = max(color) + 1
    color_mask = [0] * N
    for i in range(N):
        color_mask[color[i]] |= 1 << i

    # helper: check only pairs involving color c (all others already valid)
    def ok_for(c):
        for c2 in range(k):
            if c2 == c:
                continue
            cm = color_mask[c] | color_mask[c2]
            if cm == 0:
                continue
            if not _is_forest(cm, _pair_edge_mask[c][c2], _pair_vertex_count[c][c2]):
                return False
        return True

    def move(v, newc):
        oldc = color[v]
        if newc == oldc:
            return
        color[v] = newc
        color_mask[oldc] ^= 1 << v
        color_mask[newc] |= 1 << v
        if not ok_for(oldc) or not ok_for(newc):
            color[v] = oldc
            color_mask[oldc] |= 1 << v
            color_mask[newc] ^= 1 << v
            return False
        return True

    changed = True
    while changed and time.time() < deadline:
        changed = False
        # try to eliminate the highest color class
        for c in range(k - 1, -1, -1):
            if color_mask[c] == 0:
                continue
            verts = [v for v in range(N) if color[v] == c]
            for v in verts:
                for nc in range(c):
                    if _adj_mask[v] & color_mask[nc]:
                        continue
                    if move(v, nc):
                        if color_mask[c] == 0:
                            k = c  # color c eliminated
                        changed = True
                        break
                if changed:
                    break
            if changed:
                break
    return color, k


def construct():
    start = time.time()
    deadline = start + 80.0

    # candidate orderings for the 4-color backtracking
    orderings = [
        _ORDER,
        list(range(N)),
        [10 * r + c for r in range(5) for c in range(10)],
        [10 * c + r for c in range(10) for r in range(5)],
    ]

    best = None
    best_k = N + 1

    # 1) deterministic 4-color search
    for order in orderings:
        if time.time() > deadline - 5.0:
            break
        col = _backtrack_4color(order)
        if col is not None:
            red, k = _reduce(col, deadline)
            if k < best_k:
                best_k = k
                best = red[:]
            if best_k <= 3:
                break

    # 2) randomized 4-color attempts
    if best_k > 4 and time.time() < deadline - 5.0:
        for seed in range(300):
            if time.time() > deadline - 5.0:
                break
            random.seed(seed)
            order = list(range(N))
            random.shuffle(order)
            col = _backtrack_4color(order)
            if col is not None:
                red, k = _reduce(col, deadline)
                if k < best_k:
                    best_k = k
                    best = red[:]
                if best_k <= 3:
                    break

    # 3) fallback: unique coloring (always valid)
    if best is None:
        best = list(range(N))
        best_k = N

    return [int(c) for c in best]
# EVOLVE-BLOCK-END
