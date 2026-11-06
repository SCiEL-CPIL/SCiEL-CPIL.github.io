import time
import random
from itertools import combinations

N = 20
C = 20
DEADLINE = 148.0
START = time.time()

def elapsed():
    return time.time() - START

def make_adj(col_edges):
    adj = [[-1]*N for _ in range(N)]
    for u, v, c in col_edges:
        adj[u][v] = c
        adj[v][u] = c
    return adj

def is_rainbow(col_edges):
    adj = make_adj(col_edges)
    # Check all 4-cycles
    for a in range(N):
        for b in range(a+1, N):
            if adj[a][b] < 0:
                continue
            for c in range(b+1, N):
                if adj[b][c] < 0:
                    continue
                for d in range(c+1, N):
                    # cycle a-b-c-d-a
                    if adj[c][d] >= 0 and adj[d][a] >= 0:
                        cols = {adj[a][b], adj[b][c], adj[c][d], adj[d][a]}
                        if len(cols) != 4:
                            return False
                    # cycle a-b-d-c-a
                    if adj[b][d] >= 0 and adj[d][c] >= 0:
                        cols = {adj[a][b], adj[b][d], adj[d][c], adj[c][a]}
                        if len(cols) != 4:
                            return False
                    # cycle a-c-b-d-a
                    if adj[a][c] >= 0 and adj[b][d] >= 0:
                        cols = {adj[a][c], adj[c][b], adj[b][d], adj[d][a]}
                        if len(cols) != 4:
                            return False
    return True

def count_edges(col_edges):
    return len(col_edges)

def _try_add(adj, adj_list, u, v, c):
    if u == v:
        return False
    if adj[u][v] >= 0:
        return False
    for x in adj_list[u]:
        if x == v:
            continue
        ux_col = adj[u][x]
        for y in adj_list[x]:
            if y == u or y == x:
                continue
            if adj[y][v] < 0:
                continue
            xy_col = adj[x][y]
            yv_col = adj[y][v]
            if c == ux_col or c == xy_col or c == yv_col or ux_col == xy_col or ux_col == yv_col or xy_col == yv_col:
                return False
    return True

def _add_edge(col_edges, adj, adj_list, u, v, c):
    col_edges.append([u, v, c])
    adj[u][v] = c
    adj[v][u] = c
    adj_list[u].append(v)
    adj_list[v].append(u)

def greedy_construct(seed_val):
    random.seed(seed_val)
    col_edges = []
    adj = [[-1]*N for _ in range(N)]
    adj_list = [[] for _ in range(N)]
    all_pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
    random.shuffle(all_pairs)
    for u, v in all_pairs:
        if elapsed() > DEADLINE - 8:
            break
        if adj[u][v] >= 0:
            continue
        colours = list(range(C))
        random.shuffle(colours)
        for c in colours:
            if _try_add(adj, adj_list, u, v, c):
                _add_edge(col_edges, adj, adj_list, u, v, c)
                break
    return col_edges

def local_improve(col_edges):
    adj = make_adj(col_edges)
    adj_list = [[] for _ in range(N)]
    for u, v, c in col_edges:
        adj_list[u].append(v)
        adj_list[v].append(u)
    all_pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
    for attempt in range(8):
        random.shuffle(all_pairs)
        improved = False
        for u, v in all_pairs:
            if elapsed() > DEADLINE - 8:
                break
            if adj[u][v] >= 0:
                continue
            colours = list(range(C))
            random.shuffle(colours)
            for c in colours:
                if _try_add(adj, adj_list, u, v, c):
                    _add_edge(col_edges, adj, adj_list, u, v, c)
                    improved = True
                    break
        if not improved:
            break
    return col_edges

def color_assignment_backtracking(base_edges, seed_val):
    """
    Structural approach: fix a dense set of edges (base_edges as (u,v) pairs),
    then assign colors via backtracking with forward-checking.
    Returns list of [u,v,c] for successfully colored edges (subset of base_edges).
    """
    random.seed(seed_val)
    m = len(base_edges)
    if m == 0:
        return []
    # order edges: those with most constraints first (heuristic: random for diversity)
    order = list(range(m))
    random.shuffle(order)
    color_of = [None]*m  # color per base edge index
    # adjacency of base edges by vertex for fast 4-cycle detection
    # Precompute for each vertex the list of base-edge-indices incident to it
    vert_edges = [[] for _ in range(N)]
    for idx, (u, v) in enumerate(base_edges):
        vert_edges[u].append(idx)
        vert_edges[v].append(idx)
    # edge index by (u,v)
    edge_index = {}
    for idx, (u, v) in enumerate(base_edges):
        edge_index[(min(u,v), max(u,v))] = idx

    def get_ei(u, v):
        return edge_index.get((min(u,v), max(u,v)))

    # For a candidate color assignment to edge idx with color c, check all 4-cycles
    # that include edge idx. A 4-cycle a-b-c-d contains edge (a,b) if {a,b} is an edge.
    # We need: for every 4-cycle containing base edge idx, the other 3 edges (if present & colored)
    # must have colors != c and mutually distinct from c and each other... but they may be uncolored.
    # We only enforce: no two colored edges in the same 4-cycle share a color.
    # So when coloring edge idx with c, for every 4-cycle containing idx, if any other edge in it
    # is already colored with c -> conflict.
    # Precompute for each base edge idx the list of 4-cycles (as tuples of 4 edge-indices) containing it.
    # This is expensive; instead check on the fly.

    # On-the-fly conflict check for edge idx with color c:
    # Find all 4-cycles containing base edge (u,v)=base_edges[idx].
    # A 4-cycle containing edge (u,v) has form u-v-x-y-u or u-v-y-x-u etc.
    # i.e., pick x,y such that edges (v,x),(x,y),(y,u) exist in base_edges.
    # For each such cycle, check the colors of the other 3 edges.
    def check_conflict(idx, c):
        u, v = base_edges[idx]
        # neighbors of v in base graph
        for x in vert_edges[v]:
            if x == idx:
                continue
            xu, xv = base_edges[x]
            if xv == v:
                xa = xu
            else:
                xa = xv
            if xa == u:
                continue
            # now need edge (xa, y) and (y, u)
            for y in vert_edges[xa]:
                if y == x:
                    continue
                ya, yb = base_edges[y]
                if ya == xa:
                    yc = yb
                else:
                    yc = ya
                # edge (yc, u) must exist and be not idx, not x, not y
                ei_yu = get_ei(yc, u)
                if ei_yu is None:
                    continue
                if ei_yu in (idx, x, y):
                    continue
                # cycle edges: idx (u-v), x (v-xa), y (xa-yc), ei_yu (yc-u)
                # check colors
                for eidx in (x, y, ei_yu):
                    if color_of[eidx] == c:
                        return True
        return False

    # Also need: within a 4-cycle, the 3 other edges must be mutually distinct colors too,
    # but that's already enforced when they were colored (each pair checked then).
    # Actually no: when we colored edge A and edge B (both in cycle) they were checked against each other.
    # So only need to check new edge vs existing. Good.

    def backtrack(pos):
        if elapsed() > DEADLINE - 10:
            return 0
        if pos == len(order):
            return 1
        idx = order[pos]
        # try colors
        colours = list(range(C))
        random.shuffle(colours)
        best = 0
        for c in colours:
            if check_conflict(idx, c):
                continue
            color_of[idx] = c
            got = backtrack(pos+1)
            if got > best:
                best = got
            color_of[idx] = None
            if best >= 0:  # we just want to maximize count; but backtracking is exponential
                # continue to try to fill more? No - this is decision: color or skip.
                pass
        # option: skip this edge
        got = backtrack(pos+1)
        if got > best:
            best = got
        return best

    # This full backtracking is too slow for m~190. Use a greedy-with-repair instead.
    # Simpler structural method: assign colors greedily in random order, skipping conflicts,
    # then try to fill gaps. This is essentially greedy_construct but on a fixed edge set.
    # To be structurally different: use color-major assignment.
    return None  # fallback to other methods

def color_major_construct(seed_val):
    """
    Structurally different: assign edges to colors one color at a time.
    For each color, greedily add a C4-free-ish set of edges (no two edges sharing a 4-cycle
    with same color). This is heuristic but decouples color classes.
    """
    random.seed(seed_val)
    col_edges = []
    adj = [[-1]*N for _ in range(N)]
    adj_list = [[] for _ in range(N)]
    all_pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
    for c in range(C):
        if elapsed() > DEADLINE - 8:
            break
        pairs = list(all_pairs)
        random.shuffle(pairs)
        for u, v in pairs:
            if adj[u][v] >= 0:
                continue
            if _try_add(adj, adj_list, u, v, c):
                _add_edge(col_edges, adj, adj_list, u, v, c)
    return col_edges

def round_robin_construct(seed_val):
    """
    Alternate: round-robin over colors, each round add edges.
    """
    random.seed(seed_val)
    col_edges = []
    adj = [[-1]*N for _ in range(N)]
    adj_list = [[] for _ in range(N)]
    all_pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
    random.shuffle(all_pairs)
    # cycle through colors repeatedly
    ci = 0
    for u, v in all_pairs:
        if elapsed() > DEADLINE - 8:
            break
        if adj[u][v] >= 0:
            continue
        # try current color, then others
        start = ci
        for k in range(C):
            c = (start + k) % C
            if _try_add(adj, adj_list, u, v, c):
                _add_edge(col_edges, adj, adj_list, u, v, c)
                break
        ci = (ci + 1) % C
    return col_edges

def construct():
    best = []
    best_count = 0

    # color-major (structural)
    for s in range(10):
        if elapsed() > DEADLINE - 20:
            break
        r = color_major_construct(1000 + s)
        cc = count_edges(r)
        if cc > best_count:
            best = r; best_count = cc

    # round robin (structural)
    for s in range(10):
        if elapsed() > DEADLINE - 20:
            break
        r = round_robin_construct(2000 + s)
        cc = count_edges(r)
        if cc > best_count:
            best = r; best_count = cc

    # greedy baseline
    for s in range(30):
        if elapsed() > DEADLINE - 15:
            break
        r = greedy_construct(3000 + s)
        cc = count_edges(r)
        if cc > best_count:
            best = r; best_count = cc

    # local improve on best
    if best and elapsed() < DEADLINE - 12:
        imp = local_improve([e[:] for e in best])
        cc = count_edges(imp)
        if cc > best_count:
            best = imp; best_count = cc

    if best and is_rainbow(best):
        return best

    star = [[0, v, v] for v in range(1, N)]
    return star
