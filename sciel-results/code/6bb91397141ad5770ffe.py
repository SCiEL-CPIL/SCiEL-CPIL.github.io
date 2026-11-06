"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from random import Random
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose(edges):
        if not edges:
            return []
        m = len(edges)
        
        # Cost matrix: 0 if chainable, BIG if not
        BIG = 10**9
        cost = np.full((m, m), BIG, dtype=np.int64)
        for i in range(m):
            end_i = edges[i][1]
            for j in range(m):
                if edges[j][0] == end_i:
                    cost[i][j] = 0
        
        # Find minimum cost perfect matching
        row_ind, col_ind = linear_sum_assignment(cost)
        
        # Build next_edge and prev_edge
        next_edge = [-1] * m
        prev_edge = [-1] * m
        for r, c in zip(row_ind, col_ind):
            if cost[r, c] == 0:
                next_edge[r] = c
                prev_edge[c] = r
        
        # Identify components
        # A component is a sequence of edges.
        # Start points are edges with prev_edge == -1.
        # Cycles are components where every edge has prev_edge != -1.
        
        visited = [False] * m
        paths = []
        
        # Process paths starting from nodes with no predecessor
        for i in range(m):
            if prev_edge[i] == -1 and not visited[i]:
                path_edges = []
                cur = i
                while cur != -1 and not visited[cur]:
                    visited[cur] = True
                    path_edges.append(cur)
                    cur = next_edge[cur]
                
                # Convert to vertices
                if path_edges:
                    vertices = [edges[path_edges[0]][0]]
                    for e_idx in path_edges:
                        vertices.append(edges[e_idx][1])
                    paths.append(vertices)
        
        # Process cycles (unvisited nodes)
        for i in range(m):
            if not visited[i]:
                # This is part of a cycle
                # Break the cycle by starting at i and following next_edge
                # We will remove the last connection to make it a path
                path_edges = []
                cur = i
                while not visited[cur]:
                    visited[cur] = True
                    path_edges.append(cur)
                    cur = next_edge[cur]
                    if cur == i:
                        break
                
                # The cycle is now a list of edges.
                # To make it a simple path, we just take the sequence.
                # The last edge in the cycle connects to the first.
                # We drop that connection.
                # The path is the sequence of edges.
                if path_edges:
                    vertices = [edges[path_edges[0]][0]]
                    for e_idx in path_edges:
                        vertices.append(edges[e_idx][1])
                    paths.append(vertices)
        
        return paths

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        edges = instance(*spec)
        paths = decompose(edges)
        results.append(paths)
    return results
# EVOLVE-BLOCK-END
