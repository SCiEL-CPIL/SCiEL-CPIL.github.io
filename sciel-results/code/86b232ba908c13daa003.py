"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict, deque

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

N = P['n']
EDGES = P['edges']
ADJ = [[] for _ in range(N)]
for u, v in EDGES:
    ADJ[u].append(v)
    ADJ[v].append(u)


def _check_full(coloring):
    """Return (is_valid, num_distinct_colors)."""
    # properness
    for u, v in EDGES:
        if coloring[u] == coloring[v]:
            return False, 0
    # bicolored forest check via union-find over color pairs
    pair_edges = defaultdict(list)
    for u, v in EDGES:
        c1, c2 = coloring[u], coloring[v]
        if c1 < c2:
            pair_edges[(c1, c2)].append((u, v))
        elif c2 < c1:
            pair_edges[(c2, c1)].append((u, v))
    for (c1, c2), elist in pair_edges.items():
        parent = {}
        def find(x):
            if x not in parent:
                parent[x] = x
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(a, b):
            ra, rb = find(a), find(b)
            if ra == rb:
                return False
            parent[ra] = rb
            return True
        for a, b in elist:
            if not union(a, b):
                return False, 0
    return True, len(set(coloring))


def _surrogate_cycles(coloring):
    """Count total bicolored cycles across all color pairs (smooth-ish surrogate).
    0 means the bicolored constraint is satisfied (forest)."""
    pair_edges = defaultdict(list)
    for u, v in EDGES:
        c1, c2 = coloring[u], coloring[v]
        if c1 < c2:
            pair_edges[(c1, c2)].append((u, v))
        elif c2 < c1:
            pair_edges[(c2, c1)].append((u, v))
    total = 0
    for (c1, c2), elist in pair_edges.items():
        parent = {}
        def find(x):
            if x not in parent:
                parent[x] = x
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for a, b in elist:
            ra, rb = find(a), find(b)
            if ra == rb:
                total += 1  # this edge closes a cycle
            else:
                parent[ra] = rb
    return total


def _improper_count(coloring):
    """Count edges whose endpoints share a color."""
    c = 0
    for u, v in EDGES:
        if coloring[u] == coloring[v]:
            c += 1
    return c


def _structural_init(k, seed):
    """Grid-aware initial coloring: use (i mod k) pattern with random perturbation."""
    rng = random.Random(seed)
    # grid coordinate: v = 7*i + 3*j  (i in 0..6, j in 0..6, exclude (6,6),(6,7),(7,6))
    def coord(v):
        i = v // 7
        j = (v - 7 * i) // 3
        return i, j
    coloring = [0] * N
    for v in range(N):
        i, j = coord(v)
        coloring[v] = (i + j) % k
    # random perturbation to escape symmetric traps
    for v in range(N):
        if rng.random() < 0.15:
            coloring[v] = rng.randrange(k)
    return coloring


def _greedy_local_search(k, seed, deadline):
    """Local search minimizing (improper, cycles) lexicographically, then colors."""
    rng = random.Random(seed)
    coloring = _structural_init(k, seed)
    # first fix properness greedily
    order = list(range(N))
    rng.shuffle(order)
    for v in order:
        used = set(coloring[u] for u in ADJ[v])
        avail = [c for c in range(k) if c not in used]
        if avail:
            coloring[v] = rng.choice(avail)
        else:
            coloring[v] = rng.randrange(k)

    cur_imp = _improper_count(coloring)
    cur_cyc = _surrogate_cycles(coloring)
    cur_colors = len(set(coloring))
    best = coloring[:]
    best_imp, best_cyc, best_colors = cur_imp, cur_cyc, cur_colors

    # If already valid, record
    if cur_imp == 0 and cur_cyc == 0:
        best = coloring[:]
        best_imp, best_cyc, best_colors = 0, 0, cur_colors

    # Local search: single-vertex reassignment minimizing (imp, cyc, colors)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # randomize vertex order each pass
        verts = list(range(N))
        rng.shuffle(verts)
        for v in verts:
            old = coloring[v]
            # candidate colors: 0..k-1 (and maybe try to reduce color set)
            for c in range(k):
                if c == old:
                    continue
                coloring[v] = c
                imp = _improper_count(coloring)
                cyc = _surrogate_cycles(coloring)
                cols = len(set(coloring))
                # accept if lexicographically better or equal-imp/cyc but fewer colors
                if (imp, cyc, cols) < (cur_imp, cur_cyc, cur_colors):
                    cur_imp, cur_cyc, cur_colors = imp, cyc, cols
                    best = coloring[:]
                    best_imp, best_cyc, best_colors = imp, cyc, cols
                    improved = True
                    break
                coloring[v] = old
            # if improved, v now has a new color; continue
            if not improved and coloring[v] != old:
                coloring[v] = old
    return best, best_colors


def construct():
    start = time.time()
    deadline = start + 82.0  # safety margin before 90s

    best_coloring = list(range(N))  # fallback: 50 colors
    best_count = N

    # Try increasing k; for each k, run multiple local-search restarts
    for k in range(3, 11):
        if time.time() > deadline - 2.0:
            break
        # number of restarts scales with remaining time
        remaining = deadline - time.time()
        n_restarts = max(1, int(remaining * 3))
        for attempt in range(n_restarts):
            if time.time() > deadline - 1.5:
                break
            coloring, cnt = _greedy_local_search(k, seed=attempt * 7919 + 13, deadline=deadline - 1.5)
            if cnt < best_count:
                best_count = cnt
                best_coloring = coloring[:]
        # if we reached k colors exactly and valid, we can stop trying larger k
        if best_count <= k and best_count == 3:
            break

    # Final polish: try to reduce distinct colors from best_coloring via local search
    if best_count > 3 and time.time() < deadline - 3.0:
        k = best_count
        for attempt in range(30):
            if time.time() > deadline - 2.0:
                break
            coloring, cnt = _greedy_local_search(k, seed=100000 + attempt * 31, deadline=deadline - 2.0)
            if cnt < best_count:
                best_count = cnt
                best_coloring = coloring[:]

    # Final validation
    valid, cnt = _check_full(best_coloring)
    if not valid:
        # fallback to all-distinct
        return [int(i) for i in range(N)]

    result = [int(c) for c in best_coloring]
    for c in result:
        assert isinstance(c, int) and not isinstance(c, bool)
        assert 0 <= c <= 49
    return result
# EVOLVE-BLOCK-END
