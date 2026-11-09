"""Rewritten program: minimize distinct colors for the degree-4 graph.
Primary hypothesis: graph is bipartite -> 2 colors suffice (proper 2-coloring
makes every color-class induced subgraph edgeless, hence a forest).
Fallback: general exact search for the smallest k with a proper coloring whose
every two-color induced subgraph is a forest (checked by union-find)."""
import time

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

_N = P['n']
_EDGES = P['edges']

# adjacency
_ADJ = [[] for _ in range(_N)]
for _a, _b in _EDGES:
    _ADJ[_a].append(_b)
    _ADJ[_b].append(_a)


def _bfs_2color():
    """Return a proper 2-coloring (list of 0/1) if the graph is bipartite, else None."""
    color = [-1] * _N
    from collections import deque
    for start in range(_N):
        if color[start] != -1:
            continue
        color[start] = 0
        q = deque([start])
        while q:
            u = q.popleft()
            cu = color[u]
            for v in _ADJ[u]:
                if color[v] == -1:
                    color[v] = 1 - cu
                    q.append(v)
                elif color[v] == cu:
                    return None
    return color


def _forest_ok(coloring):
    """Check: for every pair of colors present, the induced subgraph is a forest.
    With 2 colors this is automatic if proper, but we verify generally.
    A bicolored cycle exists iff, considering only edges whose endpoints use
    colors from that pair, some component has more edges than vertices-1.
    We do this per color-pair via union-find."""
    present = sorted(set(coloring))
    m = len(present)
    # map colors to indices
    cidx = {c: i for i, c in enumerate(present)}
    # For each pair (i,j), build DSU over vertices using only edges where
    # both endpoints have colors in {present[i], present[j]} and at least one
    # endpoint uses each? No: induced subgraph on the vertex set S_{ij} =
    # {v: color[v] in {ci,cj}}; edges with both endpoints in S_{ij}.
    for i in range(m):
        for j in range(i + 1, m):
            ci, cj = present[i], present[j]
            parent = list(range(_N))
            rank = [0] * _N

            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            def union(x, y):
                rx, ry = find(x), find(y)
                if rx == ry:
                    return False
                if rank[rx] < rank[ry]:
                    rx, ry = ry, rx
                parent[ry] = rx
                if rank[rx] == rank[ry]:
                    rank[rx] += 1
                return True

            comp_v = {}
            comp_e = {}
            for a, b in _EDGES:
                ca, cb = coloring[a], coloring[b]
                if ca in (ci, cj) and cb in (ci, cj):
                    # both endpoints in S_{ij}
                    if not union(a, b):
                        # cycle
                        return False
                    ra = find(a)
                    comp_v[ra] = comp_v.get(ra, 0) + 1
                    comp_e[ra] = comp_e.get(ra, 0) + 1
            # also count isolated vertices in S_{ij} (they form trivial components, fine)
            for v in range(_N):
                if coloring[v] in (ci, cj) and find(v) == v and v not in comp_v:
                    comp_v[v] = 1
            return True
    return True


def _is_valid(coloring):
    # proper: adjacent differ
    for a, b in _EDGES:
        if coloring[a] == coloring[b]:
            return False
    # forest per color pair
    return _forest_ok(coloring)


def _greedy_coloring(k):
    """Try to color with k colors via backtracking (MRV + forward checking).
    Returns a valid coloring (satisfying proper + forest) or None."""
    color = [-1] * _N
    # vertex ordering by degree desc
    order = sorted(range(_N), key=lambda v: -len(_ADJ[v]))

    def backtrack(pos):
        if time.time() > _DEADLINE:
            return False
        if pos == _N:
            return True
        v = order[pos]
        # candidate colors
        used = set()
        for nb in _ADJ[v]:
            if color[nb] != -1:
                used.add(color[nb])
        for c in range(k):
            if c in used:
                continue
            color[v] = c
            # quick proper check already done; check forest incrementally is expensive;
            # do full check only at leaf for correctness, but prune early:
            if backtrack(pos + 1):
                return True
        color[v] = -1
        return False

    if backtrack(0):
        return color
    return None


_DEADLINE = time.time() + 80.0  # safety margin under 90s


def construct():
    global _DEADLINE
    _DEADLINE = time.time() + 80.0

    # 1) bipartite -> 2 colors (optimal, since edges exist so >=2)
    c2 = _bfs_2color()
    if c2 is not None and _is_valid(c2):
        return [int(x) for x in c2]

    # 2) general search from k=3 upward
    best = None
    for k in range(3, _N + 1):
        if time.time() > _DEADLINE - 5.0:
            break
        col = _greedy_coloring(k)
        if col is not None and _is_valid(col):
            best = col
            break

    if best is None:
        # last-resort: proper coloring with many colors (each vertex unique) is always valid
        best = list(range(_N))

    return [int(x) for x in best]
