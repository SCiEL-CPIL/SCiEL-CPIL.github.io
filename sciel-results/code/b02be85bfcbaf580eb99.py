"""Program for the suite task: few colors without any bicolored cycle."""
# EVOLVE-BLOCK-START
import time

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}


def _has_bicolored_cycle(coloring, edges):
    """Return True if any pair of colors induces a cycle (using union-find)."""
    n = len(coloring)
    # Map each color to its vertices
    from collections import defaultdict
    color_to_verts = defaultdict(list)
    for v in range(n):
        color_to_verts[coloring[v]].append(v)

    colors = list(color_to_verts.keys())
    for i in range(len(colors)):
        for j in range(i + 1, len(colors)):
            c1, c2 = colors[i], colors[j]
            allowed = set(color_to_verts[c1]) | set(color_to_verts[c2])
            # union-find over allowed vertices
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
                    return True  # cycle
                parent[ra] = rb
                return False
            for (u, v) in edges:
                if u in allowed and v in allowed:
                    if union(u, v):
                        return True
    return False


def _is_valid(coloring, edges):
    """Check proper coloring and no bicolored cycle."""
    n = len(coloring)
    for (u, v) in edges:
        if coloring[u] == coloring[v]:
            return False
    if _has_bicolored_cycle(coloring, edges):
        return False
    return True


def construct():
    start = time.time()
    deadline = start + 60.0

    n = P['n']
    edges = P['edges']

    # 5x10 grid: vertex v -> row = v // 10, col = v % 10
    # Candidate 3-coloring: color = (row + col) % 3
    coloring = [0] * n
    for v in range(n):
        row = v // 10
        col = v % 10
        coloring[v] = (row + col) % 3

    if _is_valid(coloring, edges):
        return [int(c) for c in coloring]

    # Fallback: try other linear forms (a*row + b*col) % 3 with small coefficients
    for a in range(3):
        for b in range(3):
            if a == 0 and b == 0:
                continue
            cand = [0] * n
            for v in range(n):
                row = v // 10
                col = v % 10
                cand[v] = (a * row + b * col) % 3
            if _is_valid(cand, edges):
                return [int(c) for c in cand]

    # Safety: return a guaranteed-valid but higher-color solution
    # (should not be reached)
    return list(range(n))
# EVOLVE-BLOCK-END
