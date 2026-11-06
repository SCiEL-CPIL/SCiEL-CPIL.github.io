import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145

    def random_graph(n, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def solve(edges):
        if not edges:
            return []

        n = max(max(e) for e in edges) + 1
        adj = [set() for _ in range(n)]
        edge_set = set()
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            edge_set.add((u, v) if u < v else (v, u))

        cliques = []
        remaining_edges = set(edge_set)

        while remaining_edges:
            if time.time() > deadline:
                for e in remaining_edges:
                    cliques.append(list(e))
                break

            # Compute degrees in remaining graph
            deg = [0] * n
            for u, v in remaining_edges:
                deg[u] += 1
                deg[v] += 1

            # Pick vertex with highest degree
            v_start = max(range(n), key=lambda x: deg[x])

            # Build initial candidate set: neighbors of v_start in remaining graph
            candidates = set()
            for u, v in remaining_edges:
                if u == v_start:
                    candidates.add(v)
                elif v == v_start:
                    candidates.add(u)

            # Greedy clique extension
            clique = [v_start]
            while candidates:
                # Pick candidate with highest degree among candidates
                # that is connected to all current clique members
                best = None
                best_deg = -1
                for c in candidates:
                    # Check if c is connected to all in clique
                    connected = True
                    for member in clique:
                        if member == c:
                            continue
                        e = (member, c) if member < c else (c, member)
                        if e not in remaining_edges:
                            connected = False
                            break
                    if connected and deg[c] > best_deg:
                        best_deg = deg[c]
                        best = c

                if best is None:
                    break

                clique.append(best)
                candidates.discard(best)

                # Update candidates: must be connected to best
                new_candidates = set()
                for c in candidates:
                    e = (best, c) if best < c else (c, best)
                    if e in remaining_edges:
                        new_candidates.add(c)
                candidates = new_candidates

            # Remove all edges within the clique from remaining
            for u, v in combinations(clique, 2):
                e = (u, v) if u < v else (v, u)
                if e in remaining_edges:
                    remaining_edges.discard(e)

            if len(clique) >= 2:
                cliques.append(clique)

        return cliques

    c1 = solve(random_graph(32, 0.6, 601))
    c2 = solve(random_graph(64, 0.55, 602))

    return [c1, c2]

# EVOLVE-BLOCK-START
def construct():
    import time
    import random
    from itertools import combinations

    start_time = time.time()
    deadline = start_time + 145

    def random_graph(n, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def solve(edges):
        if not edges:
            return []

        n = max(max(e) for e in edges) + 1
        adj = [set() for _ in range(n)]
        edge_set = set()
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            edge_set.add((u, v) if u < v else (v, u))

        cliques = []
        remaining_edges = set(edge_set)

        while remaining_edges:
            if time.time() > deadline:
                for e in remaining_edges:
                    cliques.append(list(e))
                break

            # Compute degrees in remaining graph
            deg = [0] * n
            for u, v in remaining_edges:
                deg[u] += 1
                deg[v] += 1

            # Pick vertex with highest degree
            v_start = max(range(n), key=lambda x: deg[x])

            # Build initial candidate set: neighbors of v_start in remaining graph
            candidates = set()
            for u, v in remaining_edges:
                if u == v_start:
                    candidates.add(v)
                elif v == v_start:
                    candidates.add(u)

            # Greedy clique extension with better candidate selection
            clique = [v_start]
            while candidates:
                # Find best candidate: highest degree, connected to all clique members
                best = None
                best_deg = -1
                for c in candidates:
                    connected = True
                    for member in clique:
                        if member == c:
                            continue
                        e = (member, c) if member < c else (c, member)
                        if e not in remaining_edges:
                            connected = False
                            break
                    if connected and deg[c] > best_deg:
                        best_deg = deg[c]
                        best = c

                if best is None:
                    break

                clique.append(best)
                candidates.discard(best)

                # Update candidates: must be connected to best
                new_candidates = set()
                for c in candidates:
                    e = (best, c) if best < c else (c, best)
                    if e in remaining_edges:
                        new_candidates.add(c)
                candidates = new_candidates

            # Remove all edges within the clique from remaining
            for u, v in combinations(clique, 2):
                e = (u, v) if u < v else (v, u)
                if e in remaining_edges:
                    remaining_edges.discard(e)

            if len(clique) >= 2:
                cliques.append(clique)

        return cliques

    c1 = solve(random_graph(32, 0.6, 601))
    c2 = solve(random_graph(64, 0.55, 602))

    return [c1, c2]
# EVOLVE-BLOCK-END
