"""Rewritten solver for few-colors-without-bicolored-cycles on the pinned degree-4 graph."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

N = P['n']
ADJ = [[] for _ in range(N)]
for u, v in P['edges']:
    ADJ[u].append(v)
    ADJ[v].append(u)


def _check_bicolored_forest(coloring, ca, cb):
    """Return True iff the subgraph induced by vertices colored ca or cb is a forest."""
    verts = [i for i in range(N) if coloring[i] == ca or coloring[i] == cb]
    if len(verts) <= 1:
        return True
    parent = {v: v for v in verts}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for u in verts:
        for v in ADJ[u]:
            if v > u and (coloring[v] == ca or coloring[v] == cb):
                ru, rv = find(u), find(v)
                if ru == rv:
                    return False
                parent[ru] = rv
    return True


def _is_valid(coloring, k):
    """Check proper coloring and no bicolored cycle for any pair of used colors."""
    for u, v in P['edges']:
        if coloring[u] == coloring[v]:
            return False
    used = sorted(set(coloring))
    for i in range(len(used)):
        for j in range(i + 1, len(used)):
            if not _check_bicolored_forest(coloring, used[i], used[j]):
                return False
    return True


def _count_cycles(coloring, k):
    """Count total bicolored cycles (number of redundant edges) across all color pairs."""
    total = 0
    used = sorted(set(coloring))
    for i in range(len(used)):
        for j in range(i + 1, len(used)):
            ca, cb = used[i], used[j]
            verts = [v for v in range(N) if coloring[v] == ca or coloring[v] == cb]
            if len(verts) <= 1:
                continue
            parent = {v: v for v in verts}
            rank = {v: 0 for v in verts}

            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            for u in verts:
                for v in ADJ[u]:
                    if v > u and (coloring[v] == ca or coloring[v] == cb):
                        ru, rv = find(u), find(v)
                        if ru == rv:
                            total += 1
                        else:
                            if rank[ru] < rank[rv]:
                                ru, rv = rv, ru
                            parent[rv] = ru
                            if rank[ru] == rank[rv]:
                                rank[ru] += 1
    return total


def _backtracking(k, deadline):
    """Backtracking search for a valid k-coloring with forward checking."""
    coloring = [-1] * N
    # Order: degree descending, then index
    order = sorted(range(N), key=lambda v: (-len(ADJ[v]), v))

    def backtrack(idx):
        if time.time() > deadline:
            return None
        if idx == N:
            return coloring[:]

        v = order[idx]
        forbidden = set()
        for nb in ADJ[v]:
            if coloring[nb] != -1:
                forbidden.add(coloring[nb])

        for c in range(k):
            if c in forbidden:
                continue
            coloring[v] = c
            # Forward check: for each assigned neighbor with different color,
            # verify the bicolored subgraph is still a forest
            ok = True
            for nb in ADJ[v]:
                if coloring[nb] != -1 and coloring[nb] != c:
                    if not _check_bicolored_forest(coloring, c, coloring[nb]):
                        ok = False
                        break
            if not ok:
                coloring[v] = -1
                continue
            result = backtrack(idx + 1)
            if result is not None:
                return result
            coloring[v] = -1
        return None

    return backtrack(0)


def _local_search(k, deadline, max_restarts=30, max_steps=300):
    """Local search minimizing bicolored cycles with proper-coloring constraint."""
    best_coloring = None
    best_cycles = float('inf')

    for attempt in range(max_restarts):
        if time.time() > deadline:
            break

        # Initialize with a proper coloring
        coloring = [-1] * N
        for v in range(N):
            forbidden = set(coloring[nb] for nb in ADJ[v] if coloring[nb] != -1)
            available = [c for c in range(k) if c not in forbidden]
            if not available:
                available = list(range(k))
            coloring[v] = random.choice(available)

        # Fix any adjacent-equal violations
        changed = True
        while changed:
            changed = False
            for u, v in P['edges']:
                if coloring[u] == coloring[v]:
                    forbidden = set(coloring[nb] for nb in ADJ[v])
                    available = [c for c in range(k) if c not in forbidden]
                    if available:
                        coloring[v] = random.choice(available)
                        changed = True

        cycles = _count_cycles(coloring, k)
        if cycles < best_cycles:
            best_cycles = cycles
            best_coloring = coloring[:]
        if cycles == 0:
            break

        # Hill-climbing
        for step in range(max_steps):
            if time.time() > deadline:
                break
            if cycles == 0:
                break

            v = random.randint(0, N - 1)
            old_c = coloring[v]
            best_new_c = old_c
            best_new_cycles = cycles

            for c in range(k):
                if c == old_c:
                    continue
                if any(coloring[nb] == c for nb in ADJ[v]):
                    continue
                coloring[v] = c
                nc = _count_cycles(coloring, k)
                if nc < best_new_cycles:
                    best_new_cycles = nc
                    best_new_c = c

            coloring[v] = best_new_c
            cycles = best_new_cycles

            if cycles < best_cycles:
                best_cycles = cycles
                best_coloring = coloring[:]

    if best_coloring is not None and best_cycles == 0:
        return best_coloring
    return None


def construct():
    start = time.time()
    deadline = start + 75  # safety margin before 90s kill

    # Try k from 3 upward
    for k in range(3, 50):
        if time.time() > deadline:
            break

        # Backtracking first (more reliable for small k)
        result = _backtracking(k, deadline)
        if result is not None:
            if _is_valid(result, k):
                return [int(c) for c in result]

        # Local search
        result = _local_search(k, deadline)
        if result is not None:
            if _is_valid(result, k):
                return [int(c) for c in result]

    # Fallback: all distinct colors
    return [int(i) for i in range(N)]
# EVOLVE-BLOCK-END
