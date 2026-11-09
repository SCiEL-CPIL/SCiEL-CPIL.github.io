"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from random import Random
    import heapq

    def instance(n, d, seed):
        rng = Random(seed)
        degree, edges = [0] * n, set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def color_strong(edges, n):
        """
        Strong edge coloring where each color class is an induced matching.
        Uses a greedy approach with careful ordering and local search improvement.
        """
        # Build adjacency
        adj = {i: set() for i in range(n)}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        # Two edges are "conflicting" if they share a vertex or are joined by an edge
        # i.e., edges (a,b) and (c,d) conflict if {a,b} & {c,d} != empty OR
        #     a is adjacent to c, a to d, b to c, or b to d
        def conflict(e1, e2):
            a, b = e1
            c, d = e2
            if a == c or a == d or b == c or b == d:
                return True
            if c in adj[a] or d in adj[a] or c in adj[b] or d in adj[b]:
                return True
            return False

        # Greedy coloring with ordering by degree of endpoints (higher degree first)
        # This tends to constrain the graph more, potentially using fewer colors
        edge_degrees = []
        for e in edges:
            a, b = e
            edge_degrees.append((len(adj[a]) + len(adj[b]), e))
        edge_degrees.sort(reverse=True)

        # Assign colors greedily
        color = {}
        color_used = 0
        # For each color, keep track of which edges have that color
        color_edges = []  # list of lists of edges per color

        for _, e in edge_degrees:
            a, b = e
            # Find the smallest color that doesn't conflict
            assigned = False
            for ci, ce_list in enumerate(color_edges):
                ok = True
                for ce in ce_list:
                    if conflict(e, ce):
                        ok = False
                        break
                if ok:
                    color[e] = ci
                    ce_list.append(e)
                    assigned = True
                    break
            if not assigned:
                ci = color_used
                color[e] = ci
                color_edges.append([e])
                color_used += 1

        # Local search: try to swap colors to reduce total number of colors
        # We'll try to eliminate the highest color by moving edges to lower colors
        improved = True
        while improved:
            improved = False
            # Work from the highest color down
            for ci in range(color_used - 1, -1, -1):
                if not color_edges[ci]:
                    color_used = ci
                    improved = True
                    continue
                # Try to move each edge in color ci to a lower color
                for ei in range(len(color_edges[ci])):
                    e = color_edges[ci][ei]
                    for cj in range(ci):
                        ok = True
                        for ce in color_edges[cj]:
                            if conflict(e, ce):
                                ok = False
                                break
                        if ok:
                            color[e] = cj
                            color_edges[cj].append(e)
                            color_edges[ci].pop(ei)
                            improved = True
                            break
                    if improved:
                        break

        # Recount colors
        if color:
            color_used = max(color.values()) + 1
        else:
            color_used = 0

        return [color[e] for e in edges]

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        n, d, seed = spec
        edges = instance(n, d, seed)
        colors = color_strong(edges, n)
        out.append(colors)
    return out
# EVOLVE-BLOCK-END
