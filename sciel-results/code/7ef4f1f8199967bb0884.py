"""Rewritten program: solve min-color bicolored-forest constraint via backtracking."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

# Precompute adjacency and edge list
_EDGES = P['edges']
_N = P['n']
_ADJ = defaultdict(list)
for u, v in _EDGES:
    _ADJ[u].append(v)
    _ADJ[v].append(u)
_EDGES_BY_U = defaultdict(list)
for u, v in _EDGES:
    _EDGES_BY_U[u].append(v)
    _EDGES_BY_U[v].append(u)


def _bicolored_cycle_check(coloring, k):
    """Check that for every pair of colors (a,b) with a<b<k, the subgraph
    induced by vertices with color in {a,b} is a forest (no cycles)."""
    # For each pair, build edges among vertices of those two colors and check acyclicity
    from collections import defaultdict as dd
    for a in range(k):
        for b in range(a + 1, k):
            # Collect vertices with color a or b
            verts = [i for i in range(_N) if coloring[i] == a or coloring[i] == b]
            if len(verts) < 3:
                continue
            # Build adjacency within this set
            vset = set(verts)
            adj = dd(list)
            ecount = 0
            for u, v in _EDGES:
                if u in vset and v in vset:
                    adj[u].append(v)
                    adj[v].append(u)
                    ecount += 1
            # Count connected components
            visited = set()
            comps = 0
            for start in verts:
                if start not in visited:
                    comps += 1
                    stack = [start]
                    visited.add(start)
                    while stack:
                        x = stack.pop()
                        for y in adj[x]:
                            if y not in visited:
                                visited.add(y)
                                stack.append(y)
            # Forest iff ecount == len(verts) - comps
            if ecount != len(verts) - comps:
                return False
    return True


def _count_bic_edges(coloring, k):
    """Count edges within each color pair (a,b) for a<b<k.
    Returns dict[(a,b)] -> count, and also total per pair."""
    from collections import defaultdict as dd
    cnt = dd(int)
    for u, v in _EDGES:
        cu, cv = coloring[u], coloring[v]
        if cu == cv:
            continue
        a, b = (cu, cv) if cu < cv else (cv, cu)
        if a < k and b < k:
            cnt[(a, b)] += 1
    return cnt


def _pair_edges_from_coloring(coloring, k):
    """Return dict (a,b) -> list of edges (u,v) for a<b<k where both endpoints
    have color in {a,b} (i.e., one endpoint color a, other color b)."""
    from collections import defaultdict as dd
    pair_edges = dd(list)
    for u, v in _EDGES:
        cu, cv = coloring[u], coloring[v]
        if cu == cv:
            continue
        a, b = (cu, cv) if cu < cv else (cv, cu)
        if a < k and b < k:
            pair_edges[(a, b)].append((u, v))
    return pair_edges


def _has_cycle_in_pair_edges(verts_of_pair, pair_edges_list):
    """Given the set of vertices that have color in the pair, and the list of
    edges among them, check if there's a cycle. Use union-find."""
    if not pair_edges_list:
        return False
    n = len(verts_of_pair)
    if n == 0:
        return False
    # Map vertex -> index
    idx = {v: i for i, v in enumerate(verts_of_pair)}
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx == ry:
            return False  # cycle
        parent[rx] = ry
        return True

    for u, v in pair_edges_list:
        if not union(idx[u], idx[v]):
            return True
    return False


def _solve_backtracking(k, deadline):
    """Backtracking search for a valid coloring with exactly k colors.
    Returns coloring list or None."""
    coloring = [-1] * _N
    # Order vertices by degree descending for better pruning
    order = sorted(range(_N), key=lambda i: -len(_ADJ[i]))
    # Precompute for each vertex the list of (neighbor, neighbor_index_in_order)
    # We'll use a simple approach: assign in order, check constraints incrementally.

    # For incremental checking, we maintain:
    # - coloring so far
    # - For each pair (a,b), the edges among assigned vertices with those colors
    # This is complex; instead, we do full check at each step (50 vertices, small).

    # Better: maintain per-pair edge count and per-pair vertex set, and check
    # cycle only when needed. For simplicity with 50 vertices, do full check.

    def _check_partial(coloring, assigned_count):
        """Check if partial coloring (first assigned_count vertices in order) is valid.
        Only check pairs where both colors have been used and there are enough vertices."""
        # For each pair (a,b), collect edges where both endpoints are assigned and
        # have colors a or b
        used_colors = set(coloring[i] for i in range(assigned_count) if coloring[i] >= 0)
        for a in used_colors:
            for b in used_colors:
                if a >= b:
                    continue
                # Collect assigned vertices with color a or b
                verts = [i for i in range(assigned_count) if coloring[i] == a or coloring[i] == b]
                if len(verts) < 3:
                    continue
                vset = set(verts)
                # Count edges among these vertices
                ecount = 0
                for u, v in _EDGES:
                    if u in vset and v in vset:
                        ecount += 1
                # Count components
                visited = set()
                comps = 0
                adj = defaultdict(list)
                for u, v in _EDGES:
                    if u in vset and v in vset:
                        adj[u].append(v)
                        adj[v].append(u)
                for start in verts:
                    if start not in visited:
                        comps += 1
                        stack = [start]
                        visited.add(start)
                        while stack:
                            x = stack.pop()
                            for y in adj[x]:
                                if y not in visited:
                                    visited.add(y)
                                    stack.append(y)
                if ecount != len(verts) - comps:
                    return False
        return True

    def backtrack(pos):
        if time.time() > deadline:
            return False
        if pos == _N:
            return True
        v = order[pos]
        # Try colors 0..k-1
        for c in range(k):
            # Check adjacency: no neighbor with same color already assigned
            ok = True
            for nb in _ADJ[v]:
                if nb in assigned_set and coloring[nb] == c:
                    ok = False
                    break
            if not ok:
                continue
            coloring[v] = c
            assigned_set.add(v)
            # Check partial validity
            if _check_partial(coloring, pos + 1):
                if backtrack(pos + 1):
                    return True
            coloring[v] = -1
            assigned_set.discard(v)
        return False

    assigned_set = set()
    # Use iterative deepening or just try
    if backtrack(0):
        return coloring
    return None


def _greedy_coloring(k):
    """Simple greedy coloring with k colors, ordered by degree descending."""
    coloring = [-1] * _N
    order = sorted(range(_N), key=lambda i: -len(_ADJ[i]))
    for v in order:
        used = set(coloring[nb] for nb in _ADJ[v] if coloring[nb] >= 0)
        for c in range(k):
            if c not in used:
                coloring[v] = c
                break
        else:
            return None
    return coloring


def _local_search(coloring, k, deadline):
    """Try to improve/fix coloring via local search. Returns improved coloring or None."""
    # For now, just return the coloring if valid
    if _bicolored_cycle_check(coloring, k):
        return coloring
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 75.0  # Safety margin before 90s limit

    # Try k from small to large
    for k in range(3, 11):
        if time.time() > deadline:
            break
        # First try greedy
        coloring = _greedy_coloring(k)
        if coloring is not None:
            if _bicolored_cycle_check(coloring, k):
                return [int(c) for c in coloring]
        # Try backtracking
        coloring = _solve_backtracking(k, deadline)
        if coloring is not None:
            return [int(c) for c in coloring]

    # Fallback: return the seed solution (50 colors, always valid)
    return list(range(_N))
# EVOLVE-BLOCK-END
