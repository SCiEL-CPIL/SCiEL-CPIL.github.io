"""Rewritten solver for the bicolored-forest coloring task."""
import time
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

N = P['n']
EDGES = P['edges']
ADJ = [[] for _ in range(N)]
for u, v in EDGES:
    ADJ[u].append(v)
    ADJ[v].append(u)


def _check_valid(colors):
    """Check proper coloring and no bicolored cycles (union-find per color pair)."""
    # Proper coloring
    for u, v in EDGES:
        if colors[u] == colors[v]:
            return False
    # For each color pair, check induced subgraph is a forest
    # Group edges by the frozenset of their endpoint colors
    from collections import defaultdict
    pair_edges = defaultdict(list)
    for u, v in EDGES:
        cu, cv = colors[u], colors[v]
        if cu == cv:
            return False
        key = (cu, cv) if cu < cv else (cv, cu)
        pair_edges[key].append((u, v))
    for key, edges in pair_edges.items():
        # Union-find on the vertices involved in this color pair
        parent = {}
        def find(x):
            if x not in parent:
                parent[x] = x
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
        for u, v in edges:
            if not union(u, v):
                return False
    return True


def _count_colors(colors):
    return len(set(colors))


def _greedy_coloring(order, max_colors):
    """Assign colors greedily in given order using up to max_colors colors."""
    colors = [-1] * N
    for v in order:
        used = set()
        for nb in ADJ[v]:
            if colors[nb] != -1:
                used.add(colors[nb])
        for c in range(max_colors):
            if c not in used:
                colors[v] = c
                break
        else:
            return None
    return colors


def _local_search(colors, max_colors, deadline, max_iters=50000):
    """Improve coloring by reassigning vertices to reduce distinct colors."""
    colors = list(colors)
    best = colors[:]
    best_count = _count_colors(best)
    for it in range(max_iters):
        if time.time() > deadline:
            break
        # Try to reassign a random vertex to a color that reduces or maintains count
        v = it % N
        old_c = colors[v]
        used = set()
        for nb in ADJ[v]:
            used.add(colors[nb])
        # Find a valid color for v
        for c in range(max_colors):
            if c in used:
                continue
            colors[v] = c
            if _check_valid(colors):
                cnt = _count_colors(colors)
                if cnt < best_count:
                    best = colors[:]
                    best_count = cnt
                break
        else:
            colors[v] = old_c
    return best, best_count


def construct():
    start = time.time()
    deadline = start + 75  # 75s safety margin within 90s limit

    # The graph is a 50-vertex 4-regular graph. Let's try to find a coloring
    # with few colors where each color pair induces a forest.
    #
    # Key insight: With k colors, each pair of colors can have at most
    # (number of vertices with those two colors) - 1 edges (forest constraint).
    # Total edges = 75. We need to distribute 75 edges across color pairs
    # such that each pair's edge count <= (vertex count in that pair) - 1.
    #
    # Let's try small k first.
    # k=2: only 1 pair, max edges = 49. We have 75 edges. Not enough.
    # k=3: 3 pairs. If balanced, each color ~17 vertices. Each pair can have
    #   up to 17+17-1=33 edges. Total capacity = 3*33=99 >= 75. Possible!
    # k=4: 6 pairs, even more room.
    #
    # Let's try k=3 first with a smart approach.

    # Try k=3 with multiple random orderings
    import random
    rng = random.Random(42)

    best_result = None
    best_count = N  # trivial upper bound

    # First try deterministic greedy with different orderings
    orderings = [
        list(range(N)),
        list(reversed(range(N))),
        sorted(range(N), key=lambda v: -len(ADJ[v])),
        list(rng.sample(range(N), N)),
    ]
    for _ in range(20):
        orderings.append(list(rng.sample(range(N), N)))

    for order in orderings:
        if time.time() > deadline:
            break
        for k in [3, 4, 5]:
            colors = _greedy_coloring(order, k)
            if colors is None:
                continue
            if _check_valid(colors):
                cnt = _count_colors(colors)
                if cnt < best_count:
                    best_count = cnt
                    best_result = colors[:]
                    if best_count == 3:
                        return [int(c) for c in best_result]

    # Local search to improve
    if best_result is not None:
        improved, icnt = _local_search(best_result, max(best_count, 5), deadline, max_iters=100000)
        if icnt < best_count:
            best_count = icnt
            best_result = improved
            if best_count == 3:
                return [int(c) for c in best_result]

    # If k=3 didn't work, try more systematic search for k=3
    # Use backtracking with the forest constraint checked incrementally
    if best_count > 3 and time.time() < deadline:
        # Try a more structured approach: assign colors 0,1,2 to vertices
        # using backtracking with pruning
        def backtrack(vidx, colors):
            if time.time() > deadline:
                return None
            if vidx == N:
                return colors[:]
            v = vidx
            for c in range(3):
                # Check proper coloring
                conflict = False
                for nb in ADJ[v]:
                    if colors[nb] == c:
                        conflict = True
                        break
                if conflict:
                    continue
                colors[v] = c
                # Check bicolored forest constraint incrementally
                # For each neighbor with a different color, check if adding
                # this edge creates a cycle in that color pair
                valid = True
                for nb in ADJ[v]:
                    cn = colors[nb]
                    if cn == c:
                        continue
                    # Check if edge (v, nb) creates a cycle in color pair (c, cn)
                    # We need to check if v and nb are already connected in the
                    # subgraph induced by colors {c, cn}
                    # This is expensive to check incrementally, so we do a full check
                    # only at the end or periodically
                    pass
                # Do a full validity check (expensive but necessary for correctness)
                if _check_valid(colors):
                    result = backtrack(vidx + 1, colors)
                    if result is not None:
                        return result
                colors[v] = -1
            return None

        # Use a vertex ordering that helps backtracking
        # Order by degree descending, then by some heuristic
        order = list(range(N))
        # Try backtracking with this order
        # Actually, let's use a different strategy: assign in order but use
        # the constraint more aggressively

        # Simpler: try many random assignments and check
        for attempt in range(50000):
            if time.time() > deadline:
                break
            colors = [rng.randint(0, 2) for _ in range(N)]
            # Fix proper coloring: for each edge, ensure different colors
            valid = True
            for u, v in EDGES:
                if colors[u] == colors[v]:
                    colors[v] = (colors[v] + 1) % 3
                    if colors[v] == colors[u]:
                        colors[v] = (colors[v] + 1) % 3
            if _check_valid(colors):
                cnt = _count_colors(colors)
                if cnt < best_count:
                    best_count = cnt
                    best_result = colors[:]
                    if best_count == 3:
                        return [int(c) for c in best_result]

    # Fallback: return the best we found, or a trivial valid coloring
    if best_result is not None:
        return [int(c) for c in best_result]

    # Last resort: use k=N (always valid)
    return [int(i) for i in range(N)]
