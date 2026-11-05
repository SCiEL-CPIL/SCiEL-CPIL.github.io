import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit

    def generate_graph(n, p, seed):
        rng = random.Random(seed)
        edges = []
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
                    adj[u].add(v)
                    adj[v].add(u)
        return edges, adj

    def solve_instance(n, p, seed):
        edges, adj = generate_graph(n, p, seed)
        if not edges:
            return []

        uncovered = set(edges)
        partition = []

        # Precompute initial degrees for ordering
        init_deg = [len(adj[i]) for i in range(n)]

        while uncovered:
            if time.time() > deadline - 5:
                # Fallback: add all remaining edges as individual cliques
                for e in uncovered:
                    partition.append(list(e))
                break

            # Find vertices involved in uncovered edges
            active_vertices = set()
            for u, v in uncovered:
                active_vertices.add(u)
                active_vertices.add(v)

            if not active_vertices:
                break

            # Compute uncovered degree for active vertices
            vert_uncovered_count = {v: 0 for v in active_vertices}
            for u, v in uncovered:
                vert_uncovered_count[u] += 1
                vert_uncovered_count[v] += 1

            # Select seed: vertex with highest uncovered degree
            seed = max(active_vertices, key=lambda x: vert_uncovered_count[x])

            # Candidates: neighbors of seed that are active
            candidates = adj[seed] & active_vertices

            # Use Bron-Kerbosch with pivot to find maximum clique containing seed
            best_clique = [seed]

            def bronk_kerbosch(R, P, X, best_clique_ref):
                if not P and not X:
                    if len(R) > len(best_clique_ref):
                        best_clique_ref.clear()
                        best_clique_ref.extend(R)
                    return

                # Pivot selection: choose u in P union X with max neighbors in P
                U = P | X
                if not U:
                    return

                # Pick pivot with max degree in P
                pivot = max(U, key=lambda u: len(adj[u] & P))

                # Only iterate over P - N(pivot)
                for v in list(P - (adj[pivot])):
                    bronk_kerbosch(R | {v}, P & adj[v], X & adj[v], best_clique_ref)
                    P.discard(v)
                    X.add(v)

            # Run BK to find max clique containing seed
            P = set(candidates)
            X = set()
            R = {seed}
            bronk_kerbosch(R, P, X, best_clique)

            # Ensure clique has at least 2 vertices
            if len(best_clique) < 2:
                if candidates:
                    v = next(iter(candidates))
                    best_clique = [seed, v]
                else:
                    # Should not happen if seed has uncovered edges
                    e = next(iter(uncovered))
                    best_clique = list(e)

            # Remove edges in best_clique from uncovered
            clique_set = set(best_clique)
            for u, v in combinations(clique_set, 2):
                edge = (min(u, v), max(u, v))
                if edge in uncovered:
                    uncovered.remove(edge)

            partition.append(sorted(best_clique))

        return partition

    res1 = solve_instance(32, 0.6, 601)
    res2 = solve_instance(64, 0.55, 602)

    return [res1, res2]
