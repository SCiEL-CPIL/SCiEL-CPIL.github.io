"""Rewritten program: deterministic greedy with dynamic ordering + local search refinement."""
import time
from collections import deque

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}


def construct():
    n = P['n']
    edges = P['edges']
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    def count_distinct_colors(color):
        return len(set(c for c in color if c >= 0))

    def has_bicolored_cycle(color, c1, c2):
        """Check if subgraph induced by colors c1 and c2 contains a cycle."""
        # Build subgraph with only vertices colored c1 or c2
        in_sub = set()
        for i in range(n):
            if color[i] == c1 or color[i] == c2:
                in_sub.add(i)
        if len(in_sub) < 3:
            return False
        # BFS/DFS to detect cycle in this subgraph
        visited = set()
        for start in in_sub:
            if start in visited:
                continue
            # BFS
            visited.add(start)
            queue = deque([start])
            parent = {start: -1}
            while queue:
                u = queue.popleft()
                for w in adj[u]:
                    if w not in in_sub:
                        continue
                    if w not in visited:
                        visited.add(w)
                        parent[w] = u
                        queue.append(w)
                    elif parent.get(u, -1) != w and parent.get(w, -1) != u:
                        # Back edge found -> cycle
                        return True
        return False

    def is_valid(color):
        """Check proper coloring and no bicolored cycles."""
        # Proper coloring check
        for u, v in edges:
            if color[u] == color[v]:
                return False
        # Bicolored cycle check
        colors_used = sorted(set(color))
        for i in range(len(colors_used)):
            for j in range(i + 1, len(colors_used)):
                if has_bicolored_cycle(color, colors_used[i], colors_used[j]):
                    return False
        return True

    def greedy_coloring(order, k):
        """Greedy coloring with given vertex order and k colors."""
        color = [-1] * n
        for v in order:
            # Find available colors
            used_by_neighbors = set()
            for u in adj[v]:
                if color[u] >= 0:
                    used_by_neighbors.add(color[u])
            # Try colors in order
            for c in range(k):
                if c in used_by_neighbors:
                    continue
                # Check bicolored cycle constraint
                color[v] = c
                valid = True
                for u in adj[v]:
                    if color[u] >= 0 and color[u] != c:
                        if has_bicolored_cycle(color, c, color[u]):
                            valid = False
                            break
                if valid:
                    break
                color[v] = -1
            if color[v] == -1:
                return None
        return color

    def local_search(color, k, max_iters=50000):
        """Try to reduce number of colors via local search."""
        current = list(color)
        current_k = count_distinct_colors(current)
        best = list(current)
        best_k = current_k
        start = time.time()
        deadline = start + 75

        for iteration in range(max_iters):
            if time.time() > deadline:
                break
            # Try to reassign a vertex to a lower color
            improved = False
            # Random vertex order for diversity
            import random
            verts = list(range(n))
            random.shuffle(verts)
            for v in verts:
                if time.time() > deadline:
                    break
                old_c = current[v]
                # Try colors from 0 up to old_c - 1
                for new_c in range(old_c):
                    if new_c == old_c:
                        continue
                    # Check proper coloring
                    ok = True
                    for u in adj[v]:
                        if current[u] == new_c:
                            ok = False
                            break
                    if not ok:
                        continue
                    # Check bicolored cycle
                    current[v] = new_c
                    valid = True
                    for u in adj[v]:
                        if current[u] >= 0 and current[u] != new_c:
                            if has_bicolored_cycle(current, new_c, current[u]):
                                valid = False
                                break
                    if valid:
                        new_k = count_distinct_colors(current)
                        if new_k < best_k:
                            best = list(current)
                            best_k = new_k
                            improved = True
                        # Keep the change if it doesn't increase color count
                        if new_k <= current_k:
                            current_k = new_k
                            improved = True
                            break
                        else:
                            current[v] = old_c
                    else:
                        current[v] = old_c
            if not improved:
                # Try some random restarts within the same k
                pass

        return best, best_k

    # Strategy: try k=3 first, then k=4, etc.
    # Use multiple orderings to find valid colorings

    best_result = None
    best_k = n  # worst case: all different colors

    start_time = time.time()
    deadline = start_time + 80

    # Try k from 3 upward
    for k in range(3, 8):
        if time.time() > deadline:
            break

        # Try different vertex orderings
        orderings = []
        # Degree-based ordering (highest degree first)
        orderings.append(sorted(range(n), key=lambda v: -len(adj[v])))
        # Natural order
        orderings.append(list(range(n)))
        # Reverse natural
        orderings.append(list(range(n - 1, -1, -1)))
        # Random orderings
        import random
        for _ in range(20):
            if time.time() > deadline:
                break
            orderings.append(random.sample(range(n), n))

        for order in orderings:
            if time.time() > deadline:
                break
            result = greedy_coloring(order, k)
            if result is not None and is_valid(result):
                # Try to reduce further with local search
                refined, refined_k = local_search(result, k, max_iters=10000)
                if refined_k < best_k:
                    best_k = refined_k
                    best_result = refined
                elif best_result is None:
                    best_k = count_distinct_colors(result)
                    best_result = result

    if best_result is not None:
        # Relabel to 0..k-1
        uc = sorted(set(best_result))
        mp = {c: i for i, c in enumerate(uc)}
        return [int(mp[c]) for c in best_result]

    # Fallback: use all distinct colors
    return [int(i) for i in range(n)]
