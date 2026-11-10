import time
import random

# EVOLVE-BLOCK-START
P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    n = P['n']
    edges = P['edges']
    deadline = time.time() + 85  # Safety margin before 90s kill

    # Build adjacency and edge set
    adj = [set() for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add((min(u, v), max(u, v)))

    def get_cycle_edges(cycle):
        es = set()
        for i in range(len(cycle)):
            u, v = cycle[i], cycle[(i + 1) % len(cycle)]
            es.add((min(u, v), max(u, v)))
        return es

    def validate_cycles(cycles, edge_set, n):
        """Verify all cycles are valid Hamiltonian cycles on the same vertex set, edge-disjoint."""
        if not cycles:
            return False
        vertex_set = None
        used_edges = set()
        for c in cycles:
            if len(c) != n or len(set(c)) != n:
                return False
            vs = frozenset(c)
            if vertex_set is None:
                vertex_set = vs
            elif vs != vertex_set:
                return False
            for i in range(n):
                u, v = c[i], c[(i + 1) % n]
                e = (min(u, v), max(u, v))
                if e not in edge_set:
                    return False
                if e in used_edges:
                    return False
                used_edges.add(e)
        return True

    def find_ham_cycle(used_edges, start, rng):
        """Find a Hamiltonian cycle using Warnsdorff-like DFS with pruning."""
        # Build remaining adjacency
        rem_adj = [set() for _ in range(n)]
        for u in range(n):
            for v in adj[u]:
                e = (min(u, v), max(u, v))
                if e not in used_edges:
                    rem_adj[u].add(v)

        # Check minimum degree >= 2
        for u in range(n):
            if len(rem_adj[u]) < 2:
                return None

        # Check connectivity
        visited = set()
        stack = [start]
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            for nb in rem_adj[node]:
                if nb not in visited:
                    stack.append(nb)
        if len(visited) != n:
            return None

        path = [start]
        visited_set = {start}
        edges_used_local = set()
        deadline_local = deadline - 3

        def dfs(current, depth):
            if time.time() > deadline_local:
                return False
            if depth == n:
                e = (min(current, start), max(current, start))
                return e in edge_set and e not in edges_used_local

            neighbors = [v for v in rem_adj[current] if v not in visited_set]
            if not neighbors:
                return False

            # Warnsdorff-like heuristic with small random tiebreak
            def score(v):
                cnt = sum(1 for w in rem_adj[v] if w not in visited_set)
                if depth == n - 1:
                    if start not in rem_adj[v]:
                        return float('inf')
                    return -1
                return cnt + rng.random() * 0.1

            neighbors.sort(key=score)

            for v in neighbors:
                if depth == n - 1 and start not in rem_adj[v]:
                    continue
                e = (min(current, v), max(current, v))
                if e in edges_used_local:
                    continue

                path.append(v)
                visited_set.add(v)
                edges_used_local.add(e)

                if dfs(v, depth + 1):
                    return True

                path.pop()
                visited_set.discard(v)
                edges_used_local.discard(e)
            return False

        if dfs(start, 1):
            return list(path)
        return None

    # Search for up to 3 edge-disjoint Hamiltonian cycles
    best = []
    seeds = [0, 1, 2, 3, 4, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61] + list(range(62, 500))

    for seed in seeds:
        if time.time() > deadline - 5:
            break
        rng = random.Random(seed)
        used_edges = set()
        cycles = []

        for _ in range(3):
            found = None
            for s in range(n):
                if time.time() > deadline - 5:
                    break
                if not any((min(s, v), max(s, v)) not in used_edges for v in adj[s]):
                    continue
                c = find_ham_cycle(used_edges, s, rng)
                if c:
                    found = c
                    break
            if not found:
                break
            cycles.append(found)
            for i in range(len(found)):
                u, v = found[i], found[(i + 1) % len(found)]
                used_edges.add((min(u, v), max(u, v)))

        if len(cycles) > len(best):
            best = list(cycles)
            if len(best) == 3:
                break

    # Validate best result
    if best and validate_cycles(best, edge_set, n):
        return [[int(v) for v in c] for c in best]

    # Fallback: try a single cycle if we have any
    if best:
        for c in best:
            if validate_cycles([c], edge_set, n):
                return [[int(v) for v in c]]

    # Last resort: find any valid triangle
    for u in range(n):
        for v in adj[u]:
            if v > u:
                for w in adj[v]:
                    if w > v and w in adj[u]:
                        tri = [u, v, w]
                        if validate_cycles([tri], edge_set, n):
                            return [[int(x) for x in tri]]

    return [[int(x) for x in [0, 1, 2]]]
# EVOLVE-BLOCK-END
