"""Solve: fewest colors for degree-4 graph with no bicolored cycles."""

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

import time
import random
import sys

DEADLINE = time.time() + 85  # safety margin before 90s kill


def _is_valid(colors, edges):
    """Check proper coloring + no bicolored cycles via union-find."""
    n = len(colors)
    # Check proper coloring
    for u, v in edges:
        if colors[u] == colors[v]:
            return False
    # Check no bicolored cycles
    # Group edges by color pair
    from collections import defaultdict
    pair_edges = defaultdict(list)
    for u, v in edges:
        cu, cv = colors[u], colors[v]
        if cu > cv:
            cu, cv = cv, cu
        pair_edges[(cu, cv)].append((u, v))
    # For each color pair, check if the subgraph is a forest
    for (cu, cv), pel in pair_edges.items():
        parent = list(range(n))
        rank = [0] * n

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra == rb:
                return False  # cycle detected
            if rank[ra] < rank[rb]:
                ra, rb = rb, ra
            parent[rb] = ra
            if rank[ra] == rank[rb]:
                rank[ra] += 1
            return True

        for u, v in pel:
            if not union(u, v):
                return False
    return True


def _count_distinct(colors):
    return len(set(colors))


def _greedy_solve(max_colors, edges, n):
    """Try to color with at most max_colors colors using backtracking with MRV."""
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    colors = [-1] * n

    def mrv_order():
        # Return uncolored vertex with fewest available colors
        best = -1
        best_avail = n + 1
        for i in range(n):
            if colors[i] == -1:
                used = set(colors[j] for j in adj[i] if colors[j] != -1)
                avail = max_colors - len(used)
                if avail < best_avail:
                    best_avail = avail
                    best = i
                    if avail <= 0:
                        break
        return best, best_avail

    sys.setrecursionlimit(10000)

    def backtrack(idx):
        if time.time() > DEADLINE:
            return False
        # Find next uncolored vertex (MRV)
        if idx == n:
            # All colored, verify no bicolored cycles
            if _is_valid(colors, edges):
                return True
            return False
        # Skip already colored
        while idx < n and colors[idx] != -1:
            idx += 1
        if idx >= n:
            return _is_valid(colors, edges)

        v = idx
        used = set(colors[j] for j in adj[v] if colors[j] != -1)
        available = [c for c in range(max_colors) if c not in used]

        # Shuffle for diversity
        random.shuffle(available)

        for c in available:
            colors[v] = c
            if backtrack(idx + 1):
                return True
            colors[v] = -1
        return False

    return backtrack(0)


def _local_search(initial, edges, n, max_iters=50000):
    """Try to reduce number of distinct colors via local search."""
    colors = list(initial)
    current_k = _count_distinct(colors)

    for _ in range(max_iters):
        if time.time() > DEADLINE:
            break
        if not _is_valid(colors, edges):
            break
        improved = False
        # Try to recolor a vertex to an existing color to reduce distinct count
        # Only useful if we can eliminate a color entirely
        used_colors = list(set(colors))
        random.shuffle(used_colors)

        for target_c in used_colors:
            if _count_distinct(colors) <= 2:
                break
            # Find a vertex with a color that appears only once (or few times)
            # Try to eliminate one color
            from collections import Counter
            cnt = Counter(colors)
            # Pick a color with smallest count to try to eliminate
            min_c = min(cnt.keys(), key=lambda c: cnt[c])
            if min_c == target_c:
                break
            # Try to recolor all vertices of min_c to other colors
            verts_of_min = [v for v in range(n) if colors[v] == min_c]
            random.shuffle(verts_of_min)
            save = list(colors)
            success = True
            for v in verts_of_min:
                used = set(colors[j] for j in range(n) if j != v and colors[j] != -1 and colors[j] != min_c)
                # Can use any color except neighbors' colors and min_c (we're eliminating min_c)
                # Actually we can use min_c for others, but not for this vertex
                avail = [c for c in range(current_k) if c != min_c and c not in used]
                if not avail:
                    success = False
                    break
                colors[v] = avail[0]
            if success:
                if _is_valid(colors, edges):
                    current_k = _count_distinct(colors)
                    improved = True
                    break
                else:
                    colors = save
            else:
                colors = save

        if not improved:
            # Try random recoloring to escape
            v = random.randint(0, n - 1)
            used = set(colors[j] for j in range(n) if j != v)
            avail = [c for c in range(current_k) if c != colors[v] and c not in used]
            if avail:
                save = list(colors)
                colors[v] = random.choice(avail)
                if not _is_valid(colors, edges):
                    colors = save
            if _count_distinct(colors) < current_k:
                current_k = _count_distinct(colors)

        if current_k <= 3:
            break

    return colors, _count_distinct(colors)


def construct():
    n = P['n']
    edges = P['edges']
    random.seed(42)

    best = None
    best_k = n + 1

    # Start with a simple greedy coloring
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    # Try increasing number of colors
    for max_c in range(3, 10):
        if time.time() > DEADLINE:
            break

        found = False
        # Multiple restarts with different orderings
        for attempt in range(30):
            if time.time() > DEADLINE:
                break
            order = list(range(n))
            random.shuffle(order)

            colors = [-1] * n
            valid = True
            for v in order:
                used = set(colors[j] for j in adj[v] if colors[j] != -1)
                avail = [c for c in range(max_c) if c not in used]
                if not avail:
                    valid = False
                    break
                # Choose color that minimizes future conflicts
                random.shuffle(avail)
                colors[v] = avail[0]

            if valid and _is_valid(colors, edges):
                k = _count_distinct(colors)
                if k < best_k:
                    best_k = k
                    best = list(colors)
                found = True
                break

        if found and best_k <= max_c:
            # Try local search to improve
            if best is not None:
                improved_colors, improved_k = _local_search(best, edges, n, max_iters=20000)
                if improved_k < best_k:
                    best_k = improved_k
                    best = improved_colors
                break

        # Also try backtracking for small max_c
        if max_c <= 5:
            if _greedy_solve(max_c, edges, n):
                k = _count_distinct(colors)
                if k < best_k:
                    best_k = k
                    best = list(colors)
                break

    # If we have a solution, try to reduce further
    if best is not None:
        # Try to see if we can do even fewer colors
        for target_k in range(best_k, 2, -1):
            if time.time() > DEADLINE:
                break
            for attempt in range(20):
                if time.time() > DEADLINE:
                    break
                random_order = list(range(n))
                random.shuffle(random_order)
                colors = [-1] * n
                valid = True
                for v in random_order:
                    used = set(colors[j] for j in adj[v] if colors[j] != -1)
                    avail = [c for c in range(target_k) if c not in used]
                    if not avail:
                        valid = False
                        break
                    random.shuffle(avail)
                    colors[v] = avail[0]
                if valid and _is_valid(colors, edges):
                    k = _count_distinct(colors)
                    if k < best_k:
                        best_k = k
                        best = list(colors)
                    break
            if best_k <= target_k:
                break

    if best is None:
        # Fallback: use all distinct colors
        return list(range(n))

    # Ensure we return proper Python ints in a list
    result = [int(c) for c in best]
    return result
