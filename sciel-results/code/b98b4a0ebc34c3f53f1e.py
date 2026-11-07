"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}


def construct():
    n = P['n']
    edges = P['edges']
    deadline = time.time() + 75.0

    # Build adjacency
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    # Helper: check if a full assignment is valid
    def is_valid(color, k):
        # Check adjacent vertices have different colors
        for u, v in edges:
            if color[u] == color[v]:
                return False
        # Check bicolored forest constraint
        for c in range(k):
            for d in range(c + 1, k):
                # Build subgraph induced by colors c and d
                # Use union-find to check for cycles
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
                        return False
                    if rank[ra] < rank[rb]:
                        ra, rb = rb, ra
                    parent[rb] = ra
                    if rank[ra] == rank[rb]:
                        rank[ra] += 1
                    return True

                for u, v in edges:
                    if (color[u] == c or color[u] == d) and (color[v] == c or color[v] == d):
                        if not union(u, v):
                            return False
        return True

    # Helper: greedy coloring with given k
    def greedy_color(k, order=None):
        if order is None:
            order = list(range(n))
        color = [-1] * n
        for v in order:
            used = set(color[u] for u in adj[v])
            c = 0
            while c in used:
                c += 1
            if c >= k:
                return None
            color[v] = c
        # Check validity
        if is_valid(color, k):
            return color
        return None

    # Try increasing k
    for k in range(2, 51):
        if time.time() > deadline:
            break

        # Try greedy with different orderings
        orderings = [
            list(range(n)),
            sorted(range(n), key=lambda x: -len(adj[x])),
            sorted(range(n), key=lambda x: len(adj[x])),
        ]
        for order in orderings:
            result = greedy_color(k, order)
            if result is not None:
                used = sorted(set(result))
                remap = {old: new for new, old in enumerate(used)}
                return [int(remap[c]) for c in result]

        # Try backtracking for small k
        if k <= 6:
            def backtrack(idx, color):
                if time.time() > deadline:
                    return False
                if idx == n:
                    return True
                v = idx
                for c in range(k):
                    # Check adjacent constraint
                    valid = True
                    for u in adj[v]:
                        if color[u] == c:
                            valid = False
                            break
                    if not valid:
                        continue
                    color[v] = c
                    # Check bicolored forest constraint incrementally
                    # For each color d != c, check if adding v creates a cycle in {c,d} subgraph
                    forest_ok = True
                    for d in range(k):
                        if d == c:
                            continue
                        # Check neighbors of v that have color c or d
                        nbrs = [u for u in adj[v] if color[u] == c or color[u] == d]
                        if len(nbrs) >= 2:
                            # Use BFS to check if any two nbrs are connected in {c,d} subgraph
                            # We need to check connectivity among nbrs excluding v
                            visited = set()
                            queue = [nbrs[0]]
                            visited.add(nbrs[0])
                            while queue:
                                cur = queue.pop(0)
                                for w in adj[cur]:
                                    if w == v:
                                        continue
                                    if color[w] == c or color[w] == d:
                                        if w not in visited:
                                            visited.add(w)
                                            queue.append(w)
                            for nbr in nbrs[1:]:
                                if nbr in visited:
                                    forest_ok = False
                                    break
                        if not forest_ok:
                            break
                    if forest_ok:
                        if backtrack(idx + 1, color):
                            return True
                    color[v] = -1
                return False

            color = [-1] * n
            if backtrack(0, color):
                used = sorted(set(color))
                remap = {old: new for new, old in enumerate(used)}
                return [int(remap[c]) for c in color]

    # Fallback: use all distinct colors
    return list(range(n))
# EVOLVE-BLOCK-END
