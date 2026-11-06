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
        # Build cost matrix: 0 if edge i can be followed by edge j, BIG otherwise.
        # Edge i = (u, v). Edge j = (x, y).
        # Chain i -> j if v == x.
        BIG = 10**9
        cost = np.full((m, m), BIG, dtype=np.int64)
        
        # Optimization: Group edges by their start vertex to speed up cost matrix construction?
        # Or just iterate. M is small (~150). M^2 is 22500. Very fast.
        
        # Map start vertex to list of edge indices starting there
        start_map = {}
        for idx, (u, v) in enumerate(edges):
            if u not in start_map:
                start_map[u] = []
            start_map[u].append(idx)
            
        for i, (u, v) in enumerate(edges):
            # Edge i ends at v. It can be followed by any edge j starting at v.
            if v in start_map:
                for j in start_map[v]:
                    cost[i][j] = 0

        row_ind, col_ind = linear_sum_assignment(cost)
        
        next_edge = [-1] * m
        prev_edge = [-1] * m
        for r, c in zip(row_ind, col_ind):
            if cost[r, c] == 0:
                next_edge[r] = c
                prev_edge[c] = r
        
        paths = []
        visited = [False] * m
        
        for i in range(m):
            if visited[i]:
                continue
            
            # Determine if i is part of a cycle or a path
            # If prev_edge[i] == -1, it's a path start.
            # Otherwise, it's in a cycle (since every node has in-degree 1 and out-degree 1 in the matched subgraph if not a path end/start).
            # Actually, in the perfect matching, every node has exactly one outgoing edge (next) and one incoming edge (prev) IF the match is valid (cost 0).
            # If cost is BIG, the match is invalid. We should treat invalid matches as non-existent links.
            
            # Wait, linear_sum_assignment returns a perfect matching.
            # If cost[r,c] > 0, it means we are forced to link two edges that don't connect?
            # No, we set cost to BIG for non-connecting.
            # The algorithm will try to minimize cost.
            # If it's impossible to have all 0s, it will pick some BIGs.
            # We should only consider links where cost == 0.
            
            # Re-evaluate next/prev based on valid links only.
            next_edge = [-1] * m
            prev_edge = [-1] * m
            for r, c in zip(row_ind, col_ind):
                if cost[r, c] == 0:
                    next_edge[r] = c
                    prev_edge[c] = r
            
            # Now traverse.
            # A node i is a start if prev_edge[i] == -1.
            # A node i is in a cycle if prev_edge[i] != -1 and we haven't visited it yet.
            
            if prev_edge[i] == -1:
                # Path
                cur = i
                path_edges = []
                while cur != -1:
                    visited[cur] = True
                    path_edges.append(cur)
                    cur = next_edge[cur]
                
                vertices = [edges[path_edges[0]][0]]
                for e_idx in path_edges:
                    vertices.append(edges[e_idx][1])
                paths.append(vertices)
            else:
                # Cycle
                # Find the cycle starting from i
                cycle_edges = []
                cur = i
                while not visited[cur]:
                    visited[cur] = True
                    cycle_edges.append(cur)
                    cur = next_edge[cur]
                
                # Break the cycle to make it a path.
                # We can just take the sequence as is.
                # The vertices will be v0, v1, ..., vk.
                # Since it's a cycle, v_k+1 == v_0.
                # A path cannot repeat vertices.
                # However, the "simple path" constraint applies to the vertices in the list.
                # If we output v0...vk, it is a simple path IF the cycle in the line graph corresponds to a simple cycle in the original graph?
                # Yes, if the edges are distinct and form a cycle in G, the vertices are distinct (except start/end if we closed it, but we don't close it).
                # Wait, if we have a cycle of edges e1, e2, e3 in G: (A,B), (B,C), (C,A).
                # Vertices: A, B, C, A.
                # The path A->B->C->A is NOT a simple path because A is repeated.
                # We must break the cycle such that we don't repeat the start vertex at the end.
                # So we output A, B, C. The edge (C,A) is the last one.
                # The path is A->B->C.
                # This covers edges (A,B) and (B,C).
                # What about (C,A)?
                # If we break the cycle at (C,A), we remove the link C->A?
                # No, we just don't continue from C to A.
                # So the path is A->B->C.
                # This covers 2 edges.
                # We missed (C,A).
                
                # To cover ALL edges in the cycle, we must include all edges.
                # A path covering a cycle of length L must have L edges.
                # A simple path with L edges has L+1 vertices.
                # If the cycle in G is simple (L distinct vertices), the path can be v0->v1->...->vL.
                # But vL is connected to v0 in the cycle.
                # If we stop at vL, we have used edges (v0,v1)...(vL-1,vL).
                # We have NOT used (vL,v0).
                # So we cannot cover a cycle with a single simple path?
                # Yes, we can. A simple path cannot close the loop.
                # Therefore, a cycle in the graph CANNOT be covered by a single simple path.
                # It must be covered by at least 2 paths? Or 1 path that doesn't close?
                # If we have a triangle A-B-C-A.
                # Path 1: A->B->C. Edges: (A,B), (B,C).
                # Edge (C,A) remains.
                # Path 2: C->A. Edge: (C,A).
                # Total 2 paths.
                
                # So, if the matching forms a cycle, we MUST break it into at least 2 paths?
                # Or can we break it into 1 path? No, 1 path is open.
                # So a cycle of edges requires 2 paths?
                # Wait. If we have a cycle, we can traverse it.
                # If we break the cycle at one edge, we get one path covering L-1 edges, and one path covering 1 edge.
                # Total 2 paths.
                
                # So for each cycle component in the matching, we add 2 paths?
                # Or do we add 1 path?
                # The number of paths in the decomposition is the number of components in the "path cover".
                # If we have a cycle, it's not a path.
                # We must split it.
                
                # Let's extract the cycle edges.
                # cycle_edges contains indices.
                # Let's get the vertices.
                # v_0 = edges[cycle_edges[0]][0]
                # ...
                # The cycle is closed.
                # We break it.
                # Path 1: edges[0] ... edges[L-2]
                # Path 2: edges[L-1]
                
                # Actually, we can just take the sequence of vertices and drop the last one (which is the start).
                # This gives a path of length L-1.
                # And the remaining edge is a path of length 1.
                
                # Let's implement this.
                
                # Get vertices of the cycle
                cycle_vertices = []
                for e_idx in cycle_edges:
                    cycle_vertices.append(edges[e_idx][0])
                # cycle_vertices has length L.
                # cycle_vertices[0] is start.
                # cycle_vertices[L-1] is the vertex before the start in the cycle?
                # Let's check:
                # e0: (v0, v1)
                # e1: (v1, v2)
                # ...
                # eL-1: (vL-1, v0)
                # cycle_vertices = [v0, v1, ..., vL-1]
                
                # Path 1: v0 -> v1 -> ... -> vL-1
                # This uses edges e0 ... eL-2.
                # Path 2: vL-1 -> v0
                # This uses edge eL-1.
                
                # But wait, is eL-1 the last one in the list?
                # The cycle_edges list follows next_edge.
                # So yes.
                
                p1_vertices = cycle_vertices
                p2_vertices = [cycle_vertices[-1], cycle_vertices[0]]
                
                paths.append(p1_vertices)
                paths.append(p2_vertices)

        return paths

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        edges = instance(*spec)
        paths = decompose(edges)
        results.append(paths)
    return results
# EVOLVE-BLOCK-END
