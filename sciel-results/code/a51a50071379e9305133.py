"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    def random_graph(n, p, seed):
        import random
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

        # Build edge set for O(1) lookup
        edge_set = set((min(u, v), max(u, v)) for u, v in edges)
        remaining = set(edge_set)

        cliques = []

        while remaining:
            # Check deadline - if close, fall back to edge cliques
            if time.time() > deadline - 5:
                for e in remaining:
                    cliques.append([e[0], e[1]])
                break

            # Compute degrees in remaining graph
            deg = [0] * n
            for u, v in remaining:
                deg[u] += 1
                deg[v] += 1

            # Find vertex with max degree
            v = max(range(n), key=lambda x: deg[x])
            if deg[v] == 0:
                # Shouldn't happen if remaining is non-empty
                u, w = next(iter(remaining))
                cliques.append([u, w])
                remaining.discard((min(u, w), max(u, w)))
                continue

            # Get neighbors of v in remaining graph
            S = set()
            for u, w in remaining:
                if u == v:
                    S.add(w)
                elif w == v:
                    S.add(u)

            if not S:
                u, w = next(iter(remaining))
                cliques.append([u, w])
                remaining.discard((min(u, w), max(u, w)))
                continue

            # Build clique starting with v
            C = [v]
            current_S = set(S)

            # Greedily extend clique
            while current_S:
                # Find vertex in current_S with max degree within current_S
                best_u = -1
                best_deg = -1
                for u in current_S:
                    d = 0
                    for w in current_S:
                        if u == w:
                            continue
                        if (min(u, w), max(u, w)) in remaining:
                            d += 1
                    if d > best_deg:
                        best_deg = d
                        best_u = u

                if best_u == -1:
                    break

                C.append(best_u)

                # Update current_S: keep only vertices connected to best_u
                new_S = set()
                for w in current_S:
                    if w == best_u:
                        continue
                    if (min(best_u, w), max(best_u, w)) in remaining:
                        new_S.add(w)
                current_S = new_S

            if len(C) >= 2:
                cliques.append(C)
                # Remove all edges within C from remaining
                for i in range(len(C)):
                    for j in range(i + 1, len(C)):
                        u, w = C[i], C[j]
                        e = (min(u, w), max(u, w))
                        remaining.discard(e)
            else:
                # Fallback: just take one edge
                u, w = next(iter(remaining))
                cliques.append([u, w])
                remaining.discard((min(u, w), max(u, w)))

        return cliques

    res1 = solve_instance(random_graph(32, 0.6, 601), 32)
    res2 = solve_instance(random_graph(64, 0.55, 602), 64)

    return [res1, res2]
# EVOLVE-BLOCK-END
