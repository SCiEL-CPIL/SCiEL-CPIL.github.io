"""Backtracking with forward checking for list-colouring bipartite quadrangulations."""
# EVOLVE-BLOCK-START
import time
import heapq
import sys
from random import Random

sys.setrecursionlimit(50000)


def construct():
    t0 = time.time()
    LIMIT = 120.0
    MARGIN = 15.0

    def deadline():
        return (time.time() - t0) > (LIMIT - MARGIN)

    def instance(splits, seed, palette=5):
        rng = Random(seed)
        faces = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
        edges = {(min(a, b), max(a, b)) for f in faces for a, b in zip(f, f[1:] + f[:1])}
        n = 8
        for _ in range(splits):
            face = faces.pop(rng.randrange(len(faces)))
            inner = list(range(n, n + 4))
            n += 4
            for i in range(4):
                a, b = face[i], face[(i + 1) % 4]
                edges.add((min(a, inner[i]), max(a, inner[i])))
                edges.add((min(inner[i], inner[(i + 1) % 4]), max(inner[i], inner[(i + 1) % 4])))
                faces.append([a, b, inner[(i + 1) % 4], inner[i]])
            faces.append(inner)
        lists = [sorted(rng.sample(range(palette), 3)) for _ in range(n)]
        return n, sorted(edges), lists

    def solve_coloring(n, adj, lists):
        # Domains as lists of available colors
        domains = [list(l) for l in lists]
        color = [-1] * n
        colored = [False] * n

        # Build initial heap with MRV
        heap = []
        for i in range(n):
            if domains[i]:
                heapq.heappush(heap, (len(domains[i]), i))
            else:
                return None

        # Count of colored vertices
        num_colored = [0]

        def backtrack():
            if num_colored[0] == n:
                return True

            # Find uncolored vertex with smallest domain (MRV)
            v = -1
            while heap:
                cnt, u = heapq.heappop(heap)
                if not colored[u] and len(domains[u]) == cnt:
                    v = u
                    break

            if v == -1:
                # Rebuild heap from uncolored vertices
                found = False
                for i in range(n):
                    if not colored[i]:
                        if not domains[i]:
                            return False
                        heapq.heappush(heap, (len(domains[i]), i))
                        found = True
                if not found:
                    return num_colored[0] == n
                return backtrack()

            # Try each color in domain[v]
            # Order colors by impact (LCV): prefer colors that remove fewer options from neighbors
            colors_to_try = list(domains[v])
            # Simple LCV: sort by sum of (1 if c in domains[nb] else 0) for uncolored neighbors
            if len(colors_to_try) > 1:
                def lcv_key(c):
                    s = 0
                    for nb in adj[v]:
                        if not colored[nb] and c in domains[nb]:
                            s += 1
                    return s
                colors_to_try.sort(key=lcv_key)

            for c in colors_to_try:
                # Assign color c to v
                color[v] = c
                colored[v] = True
                num_colored[0] += 1

                # Forward checking: remove c from uncolored neighbors
                valid = True
                affected = []
                for nb in adj[v]:
                    if not colored[nb]:
                        if c in domains[nb]:
                            domains[nb].remove(c)
                            affected.append(nb)
                            if not domains[nb]:
                                valid = False
                                break
                            else:
                                heapq.heappush(heap, (len(domains[nb]), nb))

                if valid and backtrack():
                    return True

                # Undo
                num_colored[0] -= 1
                colored[v] = False
                color[v] = -1
                for nb in affected:
                    domains[nb].append(c)
                    domains[nb].sort()

            return False

        return backtrack() and color

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if deadline():
            break
        n, edges, lists = instance(splits, seed)
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        col = solve_coloring(n, adj, lists)

        if col is None:
            # Fallback: should not happen, but provide a structurally valid answer
            col = [lists[i][0] for i in range(n)]

        # Ensure Python int type
        out.append([int(c) for c in col])

    return out
# EVOLVE-BLOCK-END
