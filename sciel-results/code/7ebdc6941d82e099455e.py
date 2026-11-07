"""Solve: few colors, no bicolored cycle (each color-pair subgraph is a forest)."""
# EVOLVE-BLOCK-START
import time

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

N = P['n']
EDGES = P['edges']

# Build adjacency
ADJ = [[] for _ in range(N)]
for a, b in EDGES:
    ADJ[a].append(b)
    ADJ[b].append(a)

# Precompute degree ordering (highest degree first)
ORDER = list(range(N))
ORDER.sort(key=lambda v: -len(ADJ[v]))

# For each vertex in ORDER, which neighbors come BEFORE it in the ordering
# (i.e., already assigned when we assign v)
pos = [0] * N
for i, v in enumerate(ORDER):
    pos[v] = i
PREV_NEI = []  # for each vertex in ORDER, list of neighbors with smaller pos
for v in ORDER:
    PREV_NEI.append([u for u in ADJ[v] if pos[u] < pos[v]])


def _solve(K, deadline):
    """Backtracking with forward checking. Returns list of N ints (color per vertex) or None."""
    color = [-1] * N
    # pair[a][b] = [V_count, E_count] for a < b
    # V_count = #vertices colored a or b; E_count = #edges between them
    Vcnt = {}
    Ecnt = {}

    def Vget(a, b):
        if a > b:
            a, b = b, a
        return Vcnt.get((a, b), 0)

    def Eget(a, b):
        if a > b:
            a, b = b, a
        return Ecnt.get((a, b), 0)

    def setV(a, b, val):
        if a > b:
            a, b = b, a
        Vcnt[(a, b)] = val

    def setE(a, b, val):
        if a > b:
            a, b = b, a
        Ecnt[(a, b)] = val

    # When we assign vertex v (in ORDER) color c, we add:
    #  - 1 to Vcnt for each pair (c, b) where b is a neighbor already colored
    #    (actually for each pair (c, x) where the neighbor has color x, plus we
    #     also need to count v itself in Vcnt for pairs (c, x) even if no neighbor
    #     has color x? No: Vcnt(a,b) counts vertices colored a or b. v is colored c.
    #     So v increments Vcnt(c, x) for every x in 0..K-1? No!
    #     v is colored c. It belongs to the subgraph induced by colors {c, x} for
    #     every x != c. So Vcnt(c, x) += 1 for all x in 0..K-1, x != c.
    #     But that's K-1 updates per assignment. With K=4, that's 3 updates. Fine.
    #  - for each neighbor u of v already colored b: Ecnt(c, b) += 1
    #
    # Wait, but Vcnt should count ALL vertices with color a or b, not just those
    # connected by edges. Let me reconsider.
    #
    # Actually, the forest condition: the subgraph induced by colors {a,b} must be
    # a forest. This subgraph has V_{a,b} vertices (all vertices colored a or b)
    # and E_{a,b} edges (edges between vertices colored a or b). Forest iff E <= V-1.
    #
    # So when we assign v color c:
    #   - Vcnt(c, x) += 1 for all x in 0..K-1, x != c  (v is now in the {c,x} subgraph)
    #   - Ecnt(c, b) += 1 for each already-colored neighbor u with color b
    #
    # The constraint to check before assigning: after assignment, for all x != c:
    #   Ecnt(c, x) + (new edges to color-x neighbors) <= Vcnt(c, x) + 1 - 1
    #   i.e., Ecnt(c, x) + k_x <= Vcnt(c, x)
    #   where k_x = number of already-colored neighbors of v with color x.
    #
    # Since Vcnt(c,x) currently does NOT include v, after adding v it becomes Vcnt(c,x)+1.
    # Forest condition: (Ecnt(c,x) + k_x) <= (Vcnt(c,x) + 1) - 1 = Vcnt(c,x).
    # So we need: Ecnt(c, x) + k_x <= Vcnt(c, x) for all x != c.

    def backtrack(idx):
        if time.time() > deadline:
            return None
        if idx == N:
            return list(color)

        v = ORDER[idx]
        # Try colors in order 0..K-1
        for c in range(K):
            # Check adjacency: no neighbor with same color
            ok = True
            for u in PREV_NEI[v]:
                if color[u] == c:
                    ok = False
                    break
            if not ok:
                continue

            # Count neighbors of v (already colored) by color
            k_by_color = [0] * K
            for u in PREV_NEI[v]:
                k_by_color[color[u]] += 1

            # Check forest constraint: for all x != c, Ecnt(c,x) + k_by_color[x] <= Vcnt(c,x)
            feasible = True
            for x in range(K):
                if x == c:
                    continue
                if Eget(c, x) + k_by_color[x] > Vget(c, x):
                    feasible = False
                    break
            if not feasible:
                continue

            # Assign
            color[v] = c
            # Update Vcnt: v adds to Vcnt(c, x) for all x != c
            for x in range(K):
                if x == c:
                    continue
                setV(c, x, Vget(c, x) + 1)
            # Update Ecnt: for each neighbor u with color b, Ecnt(c, b) += 1
            for x in range(K):
                if k_by_color[x] > 0:
                    setE(c, x, Eget(c, x) + k_by_color[x])

            res = backtrack(idx + 1)
            if res is not None:
                return res

            # Undo
            for x in range(K):
                if x == c:
                    continue
                setV(c, x, Vget(c, x) - 1)
            for x in range(K):
                if k_by_color[x] > 0:
                    setE(c, x, Eget(c, x) - k_by_color[x])
            color[v] = -1

        return None

    return backtrack(0)


def construct():
    start = time.time()
    deadline = start + 80.0  # safety margin before 90s kill

    # Try K = 3 first, then 4, then 5, ...
    # But K=3 is likely infeasible (chromatic number of this graph is probably 4)
    # Let's try from K=3 upward, but with a time budget per K.
    # Actually, let's just try K=3 with a short budget, then K=4 with the rest.

    # Try K=3 with a 5-second budget
    budget3 = min(5.0, deadline - (time.time() - start))
    if budget3 > 1.0:
        res = _solve(3, start + budget3)
        if res is not None:
            return [int(x) for x in res]

    # Try K=4 with remaining budget
    res = _solve(4, deadline)
    if res is not None:
        return [int(x) for x in res]

    # Fallback: K=5
    res = _solve(5, deadline)
    if res is not None:
        return [int(x) for x in res]

    # Last resort: K=6
    res = _solve(6, deadline)
    if res is not None:
        return [int(x) for x in res]

    # Absolute fallback: use all distinct colors (always valid, score = 50)
    return [int(i) for i in range(N)]
# EVOLVE-BLOCK-END
