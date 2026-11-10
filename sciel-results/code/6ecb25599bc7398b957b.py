import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

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

        # Use a set for fast lookup and removal
        # Represent edge as (min(u,v), max(u,v))
        edge_set = set((min(u, v), max(u, v)) for u, v in edges)
        remaining_edges = set(edge_set)

        # Build adjacency bitmask for the full graph (for clique checks)
        # adj_mask[v] is an integer bitmask of neighbors of v in the ORIGINAL graph
        adj_mask = [0] * n
        for u, v in remaining_edges:
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)

        cliques = []

        while remaining_edges:
            if time.time() > deadline:
                # Fallback: cover remaining edges as size-2 cliques
                for e in list(remaining_edges):
                    cliques.append(list(e))
                remaining_edges.clear()
                break

            # Find vertex with max degree in remaining_edges
            deg = [0] * n
            for u, v in remaining_edges:
                deg[u] += 1
                deg[v] += 1

            # Find v with max deg
            v = max(range(n), key=lambda x: deg[x])

            # Get neighbors of v in remaining_edges
            S = set()
            for u, w in remaining_edges:
                if u == v:
                    S.add(w)
                elif w == v:
                    S.add(u)

            if not S:
                # Isolated vertex? Should not happen if remaining_edges not empty and v has max deg > 0.
                u, w = next(iter(remaining_edges))
                cliques.append([u, w])
                remaining_edges.remove((min(u, w), max(u, w)))
                continue

            C = [v]
            # Greedy extension using bitmask for speed
            # S_mask tracks which vertices in S are still candidates
            S_mask = 0
            for u in S:
                S_mask |= (1 << u)

            # Common neighbors of all vertices in C (within S)
            common_mask = S_mask
            for cv in C:
                common_mask &= adj_mask[cv]
            # Remove v itself and any not in S
            common_mask &= S_mask
            common_mask &= ~(1 << v)

            while common_mask:
                # Pick u in common_mask with max degree in the induced subgraph on common_mask
                best_u = -1
                best_deg = -1
                temp = common_mask
                while temp:
                    # Extract lowest set bit
                    lb = temp & (-temp)
                    u = lb.bit_length() - 1
                    # Count neighbors of u within common_mask
                    d = (common_mask & adj_mask[u]).bit_count()
                    if d > best_deg:
                        best_deg = d
                        best_u = u
                    temp &= temp - 1  # clear lowest set bit

                if best_u == -1:
                    break

                C.append(best_u)

                # Update common_mask: intersect with adj_mask[best_u]
                common_mask &= adj_mask[best_u]
                common_mask &= ~(1 << best_u)

            if len(C) >= 2:
                cliques.append(C)
                # Remove edges in C
                for i in range(len(C)):
                    for j in range(i + 1, len(C)):
                        u, w = C[i], C[j]
                        e = (min(u, w), max(u, w))
                        if e in remaining_edges:
                            remaining_edges.remove(e)
            else:
                # Fallback, shouldn't happen
                u, w = next(iter(remaining_edges))
                cliques.append([u, w])
                remaining_edges.remove((min(u, w), max(u, w)))

        return cliques

    res1 = solve_instance(random_graph(32, 0.6, 601), 32)
    res2 = solve_instance(random_graph(64, 0.55, 602), 64)

    return [res1, res2]
