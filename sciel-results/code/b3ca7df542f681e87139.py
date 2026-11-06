import time
import random
from collections import defaultdict


def construct():
    t0 = time.time()
    DEADLINE = t0 + 140.0

    def gen_instance(n, d, seed):
        rng = random.Random(seed)
        degree = [0] * n
        edges = set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def build_conflict(edges):
        m = len(edges)
        v2e = defaultdict(list)
        for i, (u, v) in enumerate(edges):
            v2e[u].append(i)
            v2e[v].append(i)

        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        conflict = [0] * m

        # Type 1: share a vertex
        for u, lst in v2e.items():
            for a in range(len(lst)):
                for b in range(a + 1, len(lst)):
                    conflict[lst[a]] |= 1 << lst[b]
                    conflict[lst[b]] |= 1 << lst[a]

        # Type 2: connected by an edge (disjoint edges with an edge between them)
        for i in range(m):
            u, v = edges[i]
            # For each neighbor w of u, for each edge j incident to w (j != i, j not sharing v with i)
            # and j not already conflicting with i
            for w in adj[u]:
                if w == v:
                    continue
                for j in v2e[w]:
                    if j == i:
                        continue
                    if (conflict[i] >> j) & 1:
                        continue
                    # j shares vertex w with i's endpoint u, but we need j to be DISJOINT from i
                    x, y = edges[j]
                    if x == u or x == v or y == u or y == v:
                        continue
                    # j is disjoint from i, and w is adjacent to u (endpoint of i)
                    # So they are connected by edge (u, w)
                    conflict[i] |= 1 << j
                    conflict[j] |= 1 << i
            # Similarly for neighbors of v
            for w in adj[v]:
                if w == u:
                    continue
                for j in v2e[w]:
                    if j == i:
                        continue
                    if (conflict[i] >> j) & 1:
                        continue
                    x, y = edges[j]
                    if x == u or x == v or y == u or y == v:
                        continue
                    conflict[i] |= 1 << j
                    conflict[j] |= 1 << i

        return conflict

    def greedy_color(conflict, m):
        deg = [bin(conflict[i]).count('1') for i in range(m)]
        order = sorted(range(m), key=lambda i: -deg[i])
        colors = [-1] * m
        color_masks = []  # color_masks[c] = bitmask of vertices with color c
        for v in order:
            mask = conflict[v]
            # Find smallest color not used by any neighbor
            c = 0
            while True:
                # Check if any neighbor has color c
                # We need to check: does (conflict[v] & color_masks[c]) have any set bits?
                if c < len(color_masks):
                    if mask & color_masks[c]:
                        c += 1
                        continue
                # Color c is available
                colors[v] = c
                if c == len(color_masks):
                    color_masks.append(0)
                color_masks[c] |= 1 << v
                break
        return colors

    def local_search(colors, conflict, m, deadline):
        """Iteratively recolor vertices to reduce color count."""
        if m == 0:
            return colors, 0
        max_c = max(colors)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Build color masks for O(1) checks
            color_masks = defaultdict(int)
            for v in range(m):
                color_masks[colors[v]] |= 1 << v

            for v in sorted(range(m), key=lambda x: -colors[x]):
                cur = colors[v]
                if cur == 0:
                    continue
                mask = conflict[v]
                for c in range(cur):
                    if c not in color_masks or not (mask & color_masks[c]):
                        # Recolor v from cur to c
                        color_masks[cur] &= ~(1 << v)
                        if color_masks[cur] == 0:
                            del color_masks[cur]
                        color_masks[c] |= 1 << v
                        colors[v] = c
                        improved = True
                        break
        max_c = max(colors) if m > 0 else 0
        return colors, max_c + 1

    def try_color_k(conflict, m, k, deadline):
        """
        Try to color with k colors using DSATUR backtracking.
        Returns coloring list or None if infeasible/timed out.
        """
        deg = [bin(conflict[i]).count('1') for i in range(m)]
        colors = [-1] * m
        sat = [0] * m  # bitmask of forbidden colors
        uncolored = (1 << m) - 1
        result = [None]

        # Precompute neighbor lists for faster iteration
        neighbors = []
        for i in range(m):
            nb = []
            mask = conflict[i]
            for j in range(m):
                if (mask >> j) & 1:
                    nb.append(j)
            neighbors.append(nb)

        def dfs():
            if time.time() > deadline:
                result[0] = 'timeout'
                return False
            if uncolored == 0:
                return True

            # Select vertex: max saturation, then max degree
            best_v = -1
            best_sat = -1
            best_deg = -1
            tmp = uncolored
            while tmp:
                lb = tmp & (-tmp)
                v = lb.bit_length() - 1
                tmp ^= lb
                s = bin(sat[v]).count('1')
                if s > best_sat or (s == best_sat and deg[v] > best_deg):
                    best_sat = s
                    best_deg = deg[v]
                    best_v = v

            v = best_v
            for c in range(k):
                bit = 1 << c
                if sat[v] & bit:
                    continue
                # Assign
                colors[v] = c
                uncolored &= ~(1 << v)
                changed = []
                for u in neighbors[v]:
                    if (uncolored >> u) & 1:
                        if not (sat[u] & bit):
                            sat[u] |= bit
                            changed.append(u)

                if dfs():
                    return True

                # Undo
                for u in changed:
                    sat[u] &= ~bit
                uncolored |= (1 << v)
                colors[v] = -1

            return False

        if dfs():
            if result[0] == 'timeout':
                return None
            return colors
        return None

    def solve_instance(edges, deadline):
        m = len(edges)
        if m == 0:
            return []

        conflict = build_conflict(edges)

        # Phase 1: Greedy + local search
        colors = greedy_color(conflict, m)
        colors, k_upper = local_search(colors, conflict, m, deadline - 5)

        # Phase 2: Try to improve with exact search
        # Allocate remaining time for exact search
        time_left = deadline - time.time()
        if time_left > 2.0:
            # Try k = k_upper - 1, k_upper - 2, ...
            # Give each attempt a fraction of remaining time
            for k in range(k_upper - 1, 0, -1):
                if time.time() > deadline - 1.0:
                    break
                # Time budget per k: proportional to remaining
                time_budget = min(time_left * 0.3, deadline - time.time() - 1.0)
                if time_budget < 0.5:
                    break
                coloring = try_color_k(conflict, m, k, time.time() + time_budget)
                if coloring is not None:
                    colors = coloring
                    k_upper = k
                    time_left = deadline - time.time()
                    if time_left < 2.0:
                        break

        return colors

    results = []
    specs = [(40, 4, 101), (60, 5, 102), (80, 6, 103)]

    for n, d, seed in specs:
        if time.time() > DEADLINE - 10:
            # Emergency fallback
            edges = gen_instance(n, d, seed)
            m = len(edges)
            if m == 0:
                results.append([])
                continue
            conflict = build_conflict(edges)
            colors = greedy_color(conflict, m)
            results.append(colors)
            continue

        edges = gen_instance(n, d, seed)
        # Allocate time per instance: first gets less, later ones get more
        # Actually, give equal time but with safety margin
        inst_deadline = min(DEADLINE - 5, time.time() + 40.0)
        colors = solve_instance(edges, inst_deadline)
        results.append(colors)

    return results
