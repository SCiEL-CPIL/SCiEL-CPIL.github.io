import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin vs 150s kill

    def random_graph(n, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def solve_instance(edges, n):
        if not edges:
            return []

        # Build adjacency bitmask: adj[u] has bit v set iff edge (u,v) present.
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)

        # remaining = set of undirected edges (canonical tuple) still uncovered.
        remaining = set((min(u, v), max(u, v)) for u, v in edges)

        cliques = []

        while remaining:
            if time.time() > deadline:
                # Fallback: cover every remaining edge as a size-2 clique.
                for e in remaining:
                    cliques.append([e[0], e[1]])
                break

            # Degree in remaining graph.
            deg = [0] * n
            for u, v in remaining:
                deg[u] += 1
                deg[v] += 1

            # Vertex with max degree.
            v = max(range(n), key=lambda x: deg[x])
            S = adj[v] & ~((1 << v))

            if S == 0:
                # v isolated in remaining; grab any edge.
                u, w = next(iter(remaining))
                cliques.append([u, w])
                remaining.remove((min(u, w), max(u, w)))
                continue

            C = [v]
            # Greedily extend: repeatedly add the vertex in S with the most
            # neighbors within S (densest continuation), then restrict S to its
            # neighbors so the clique stays a clique.
            while S:
                best_u = -1
                best_deg = -1
                for u in range(n):
                    if (S >> u) & 1:
                        d = (adj[u] & S).bit_count()
                        if d > best_deg:
                            best_deg = d
                            best_u = u
                if best_u == -1:
                    break
                C.append(best_u)
                S &= adj[best_u]  # keep only neighbors of best_u within S

            if len(C) >= 2:
                cliques.append(C)
                for i in range(len(C)):
                    for j in range(i + 1, len(C)):
                        a, b = C[i], C[j]
                        e = (min(a, b), max(a, b))
                        if e in remaining:
                            remaining.remove(e)
            else:
                u, w = next(iter(remaining))
                cliques.append([u, w])
                remaining.remove((min(u, w), max(u, w)))

        return cliques

    res1 = solve_instance(random_graph(32, 0.6, 601), 32)
    res2 = solve_instance(random_graph(64, 0.55, 602), 64)

    return [res1, res2]
