"""Strong edge colouring via greedy with degree-based ordering and DSU for induced matching checks."""
# EVOLVE-BLOCK-START
def construct():
    import random
    from collections import defaultdict

    def instance(n, d, seed):
        rng = random.Random(seed)
        degree = [0] * n
        edges = set()
        # Try many times to build a bounded-degree graph
        for _ in range(4 * n * d):
            u = rng.randrange(n)
            v = rng.randrange(n)
            if u == v:
                continue
            e = (min(u, v), max(u, v))
            if e in edges:
                continue
            if degree[u] < d and degree[v] < d:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def strong_edge_coloring(edges, n, max_deg):
        """
        Strong edge coloring: each color class is an induced matching.
        Two edges conflict if they share a vertex OR are connected by an edge.
        We use a greedy approach with a smart ordering to minimize colors.
        """
        # Build adjacency
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        # For each edge, compute its "conflict neighborhood":
        # An edge e=(u,v) conflicts with edge f=(x,y) if:
        #   - they share a vertex (u==x, u==y, v==x, v==y)
        #   - OR they are joined by an edge: i.e., one endpoint of e is adjacent to one endpoint of f
        #     specifically: u-x, u-y, v-x, v-y where {u,v} != {x,y}
        # More precisely: edges e and f are "adjacent" in the line graph of the square of G?
        # Actually, for strong edge coloring, two edges are incompatible if:
        #   1. They share a vertex, OR
        #   2. There exists an edge in G connecting a vertex of e to a vertex of f (and e != f)
        # So the "forbidden set" for edge e=(u,v) includes all edges incident to u or v,
        # plus all edges that have one endpoint in N(u) ∪ N(v) and the other endpoint not in {u,v}... 
        # Wait, let me be more careful.
        # Edge f=(x,y) conflicts with e=(u,v) if:
        #   - f shares a vertex with e: {x,y} ∩ {u,v} ≠ ∅
        #   - OR there is an edge in G between a vertex of e and a vertex of f:
        #     i.e., (u,x)∈E, (u,y)∈E, (v,x)∈E, or (v,y)∈E, where f≠e
        # Note: if f shares a vertex with e, the "joined by an edge" condition might also hold,
        # but we just need to check if ANY conflict exists.

        # For efficiency, let's precompute for each edge its conflict set.
        # But that could be O(m^2). With m up to ~240 edges, that's fine.
        
        m = len(edges)
        edge_to_idx = {e: i for i, e in enumerate(edges)}
        
        # Precompute conflict matrix
        # conflict[i][j] = True if edges[i] and edges[j] conflict
        conflict = [[False] * m for _ in range(m)]
        for i in range(m):
            u, v = edges[i]
            for j in range(i + 1, m):
                x, y = edges[j]
                # Check if they share a vertex
                if x == u or x == v or y == u or y == v:
                    conflict[i][j] = conflict[j][i] = True
                    continue
                # Check if joined by an edge
                # Edges in G: we have the adj sets
                if x in adj[u] or x in adj[v] or y in adj[u] or y in adj[v]:
                    conflict[i][j] = conflict[j][i] = True
        
        # Now greedy coloring with ordering by degree in conflict graph
        # Degree of each edge in conflict graph
        deg = [sum(conflict[i]) for i in range(m)]
        
        # Order edges by descending degree in conflict graph (worst first)
        order = sorted(range(m), key=lambda i: -deg[i])
        
        color = [-1] * m
        num_colors = 0
        
        for i in order:
            # Find the smallest available color
            used = set()
            for j in range(m):
                if color[j] != -1 and conflict[i][j]:
                    used.add(color[j])
            c = 0
            while c in used:
                c += 1
            color[i] = c
            num_colors = max(num_colors, c + 1)
        
        return [color[edge_to_idx[e]] for e in edges]

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        n, d, seed = spec
        edges = instance(n, d, seed)
        if not edges:
            out.append([])
            continue
        colors = strong_edge_coloring(edges, n, d)
        out.append(colors)
    return out
# EVOLVE-BLOCK-END
