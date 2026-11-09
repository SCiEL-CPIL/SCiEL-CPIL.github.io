"""Decompose three fixed graphs into few cycles and edges (Erdős-Gallai)."""
# EVOLVE-BLOCK-START
def construct():
    import random
    import sys
    from collections import deque, defaultdict

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose(edges):
        """
        Decompose the graph into cycles and edges.
        Strategy:
        1. Use a maximum 2-matching approach to find a large collection of edge-disjoint cycles.
        2. Specifically, we model this as finding a maximum set of edge-disjoint cycles.
           This is equivalent to finding a maximum 2-matching in the line graph, but that's complex.
           Instead, we use a greedy + local search or a direct cycle packing algorithm.
           
        A simpler and effective heuristic:
        - Find edge-disjoint cycles greedily by repeatedly finding a cycle in the remaining graph.
        - To improve, we can try to find a maximum 2-matching (a set of edges where each vertex has degree <= 2)
          and then decompose the resulting 2-regular (or less) subgraph into cycles.
          The remaining edges are singletons.
        
        Maximum 2-matching can be found via reduction to bipartite matching:
        Create a bipartite graph with left nodes L_v and right nodes R_v for each vertex v.
        For each edge (u,v) in the original graph, add an edge from L_u to R_v and from L_v to R_u.
        A matching in this bipartite graph corresponds to a 2-matching in the original graph.
        The size of the 2-matching is the size of the bipartite matching.
        
        Then, the edges in the 2-matching form a subgraph where each vertex has degree <= 2.
        This subgraph consists of cycles and paths. We can extract all cycles from it.
        The remaining edges (not in the 2-matching) are singletons.
        
        However, the 2-matching might contain paths, which are not cycles.
        We want to maximize the number of cycles. A 2-matching decomposes into cycles and paths.
        The edges in the cycles are used in cycles; the edges in the paths are not.
        So we should extract only the cycles from the 2-matching.
        
        But wait, the task says "pieces are edge-disjoint and cover all edges".
        If we use a 2-matching, the edges in the 2-matching are partitioned into cycles and paths.
        The cycles become cycle pieces. The path edges become singleton pieces.
        The edges not in the 2-matching become singleton pieces.
        
        To minimize the number of pieces, we want to maximize the number of edges covered by cycles.
        A cycle of length k covers k edges with 1 piece. A singleton covers 1 edge with 1 piece.
        So using a cycle is always better than singletons for those edges.
        
        Therefore, the problem reduces to: find a set of edge-disjoint cycles that covers the maximum number of edges.
        This is the maximum 2-matching problem (where we only count the cycle components).
        
        Actually, the maximum 2-matching in a graph is a well-studied problem.
        A 2-matching is a set of edges such that each vertex has degree at most 2.
        The components are cycles and paths.
        We want to maximize the number of edges in the cycle components.
        
        This is equivalent to finding a maximum 2-matching and then taking the cycle components.
        But a maximum 2-matching might have paths. We can't just take a maximum 2-matching and hope it's all cycles.
        
        Alternative: use a greedy algorithm to find edge-disjoint cycles.
        1. Start with the full set of edges.
        2. Repeatedly find a cycle in the remaining graph.
        3. Remove the cycle's edges.
        4. Continue until no more cycles can be found.
        5. The remaining edges are singletons.
        
        To make this more effective, we can use a priority based on cycle length or use a more sophisticated method.
        
        Let's try the bipartite matching reduction for maximum 2-matching, then extract cycles.
        Even if there are paths, the cycle parts are still valid cycle pieces.
        The path edges become singletons.
        The non-2-matching edges become singletons.
        
        Is this optimal? Not necessarily, but it's a good heuristic.
        
        Let's implement the bipartite matching using Hopcroft-Karp.
        """
        if not edges:
            return []
        
        n = max(max(u, v) for u, v in edges) + 1 if edges else 0
        edge_set = set(edges)
        
        # Build adjacency for the original graph
        adj = defaultdict(list)
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Bipartite graph construction:
        # Left side: 0..n-1 (L_v)
        # Right side: n..2n-1 (R_v)
        # Edge (u,v) in original -> L_u -> R_v and L_v -> R_u in bipartite
        
        left_size = n
        right_size = n
        total_nodes = 2 * n
        
        # Bipartite adjacency: from left node to right node
        bip_adj = [[] for _ in range(left_size)]
        # We need to map each bipartite edge back to the original edge
        # Each original edge (u,v) gives two bipartite edges: (L_u, R_v) and (L_v, R_u)
        # If a bipartite edge is in the matching, it means the original edge is "used" in the 2-matching.
        # But we need to be careful: if both (L_u, R_v) and (L_v, R_u) are in the matching, 
        # that would mean the edge (u,v) is used twice, which is not allowed.
        # Actually, in the standard reduction, a matching in the bipartite graph corresponds to a 2-matching.
        # Each original edge can be used at most once.
        
        # Let's store for each bipartite edge (l, r) the original edge it represents.
        # bip_edge_info[(l, r)] = (u, v) where l = u, r = v + n (so r-n = v)
        # But we also have the reverse: (v, u + n) for the same original edge (u,v).
        
        # To avoid using an original edge twice, we need to ensure that for each original edge (u,v),
        # at most one of (L_u, R_v) or (L_v, R_u) is in the matching.
        # This is automatically handled if we treat them as distinct bipartite edges, 
        # but we need to check that we don't select both.
        
        # Actually, the standard reduction works as follows:
        # A 2-matching in G corresponds to a matching in the bipartite graph B.
        # The size of the maximum 2-matching equals the size of the maximum matching in B.
        # Each edge in the 2-matching corresponds to one edge in the matching in B.
        
        # Let's build the bipartite graph.
        # For each original edge (u, v) with u < v:
        #   Add bipartite edge (u, v + n)
        #   Add bipartite edge (v, u + n)
        
        for u, v in edges:
            bip_adj[u].append(v + n)
            bip_adj[v].append(u + n)
        
        # Hopcroft-Karp algorithm
        def hopcroft_karp(bip_adj, left_size, right_size):
            pair_u = [-1] * left_size
            pair_v = [-1] * (right_size)
            dist = [0] * left_size
            
            def bfs():
                queue = deque()
                for u in range(left_size):
                    if pair_u[u] == -1:
                        dist[u] = 0
                        queue.append(u)
                    else:
                        dist[u] = -1
                found = False
                while queue:
                    u = queue.popleft()
                    for v in bip_adj[u]:
                        pu = pair_v[v - n] if v >= n else -1  # v is on right side, so v-n is the left node index
                        # Wait, v in bip_adj[u] is an index in the right side.
                        # The right side nodes are indexed 0..right_size-1, but we stored them as v+n.
                        # So the actual right-side index is v - n.
                        # pair_v is indexed by right-side index (0..right_size-1).
                        right_idx = v - n
                        if right_idx < 0 or right_idx >= right_size:
                            continue
                        pu = pair_v[right_idx]
                        if pu == -1:
                            found = True
                        elif dist[pu] == -1:
                            dist[pu] = dist[u] + 1
                            queue.append(pu)
                return found
            
            def dfs(u):
                for v in bip_adj[u]:
                    right_idx = v - n
                    if right_idx < 0 or right_idx >= right_size:
                        continue
                    pu = pair_v[right_idx]
                    if pu == -1 or (dist[pu] == dist[u] + 1 and dfs(pu)):
                        pair_u[u] = v
                        pair_v[right_idx] = u
                        return True
                dist[u] = -1
                return False
            
            matching = 0
            while bfs():
                for u in range(left_size):
                    if pair_u[u] == -1:
                        if dfs(u):
                            matching += 1
            return pair_u, pair_v, matching
        
        # Run Hopcroft-Karp
        pair_u, pair_v, matching_size = hopcroft_karp(bip_adj, left_size, right_size)
        
        # Extract the 2-matching edges
        # For each matched left node u, it is matched to a right node v_right.
        # The original edge is (u, v_right - n).
        two_matching_edges = set()
        for u in range(left_size):
            if pair_u[u] != -1:
                v_right = pair_u[u]
                v_orig = v_right - n
                if 0 <= v_orig < n:
                    edge = (min(u, v_orig), max(u, v_orig))
                    two_matching_edges.add(edge)
        
        # Now, from the 2-matching edges, find all cycles
        # Build adjacency from the 2-matching edges
        tm_adj = defaultdict(list)
        for u, v in two_matching_edges:
            tm_adj[u].append(v)
            tm_adj[v].append(u)
        
        # Find cycles in the 2-matching subgraph
        visited_edges = set()
        cycles = []
        
        for u, v in two_matching_edges:
            if (u, v) not in visited_edges and (v, u) not in visited_edges:
                # Try to find a cycle starting from edge (u, v)
                # We can do a DFS to find a cycle
                stack = [(u, v, [(u, v)])]
                found_cycle = None
                while stack:
                    curr, prev, path = stack.pop()
                    if found_cycle:
                        break
                    for neighbor in tm_adj[curr]:
                        if neighbor == prev:
                            continue
                        if (curr, neighbor) in visited_edges or (neighbor, curr) in visited_edges:
                            continue
                        new_path = path + [(curr, neighbor)]
                        if neighbor == u:
                            # Found a cycle
                            found_cycle = new_path
                            break
                        # Avoid going back
                        stack.append((neighbor, curr, new_path))
                
                if found_cycle:
                    # Extract the cycle vertices
                    cycle_vertices = []
                    for i, (a, b) in enumerate(found_cycle):
                        if i == 0:
                            cycle_vertices.append(a)
                            cycle_vertices.append(b)
                        else:
                            cycle_vertices.append(b)
                    # Remove the last vertex which is the same as the first
                    # The cycle is closed, so the last edge returns to the start
                    # The vertices are: v0, v1, v2, ..., vk where vk = v0
                    # But our path is edges, so we have (v0,v1), (v1,v2), ..., (vk-1, vk)
                    # The cycle vertices are v0, v1, ..., vk-1
                    cycle_vertices = [found_cycle[0][0]]
                    for a, b in found_cycle:
                        cycle_vertices.append(b)
                    # The last vertex is the start, so remove it
                    if len(cycle_vertices) >= 2 and cycle_vertices[0] == cycle_vertices[-1]:
                        cycle_vertices = cycle_vertices[:-1]
                    
                    if len(cycle_vertices) >= 3:
                        cycles.append(cycle_vertices)
                        # Mark edges as visited
                        for i in range(len(found_cycle)):
                            a, b = found_cycle[i]
                            visited_edges.add((a, b))
                            visited_edges.add((b, a))
        
        # The remaining edges are singletons
        pieces = []
        for cycle in cycles:
            pieces.append(cycle)
        
        for u, v in edges:
            if (u, v) not in two_matching_edges:
                pieces.append([u, v])
            else:
                # Check if this edge is part of a cycle
                # We need to verify if (u,v) is in any of the found cycles
                in_cycle = False
                for cycle in cycles:
                    for i in range(len(cycle)):
                        a = cycle[i]
                        b = cycle[(i + 1) % len(cycle)]
                        if (min(a, b), max(a, b)) == (u, v):
                            in_cycle = True
                            break
                    if in_cycle:
                        break
                if not in_cycle:
                    pieces.append([u, v])
        
        return pieces

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        edges = instance(*spec)
        pieces = decompose(edges)
        results.append(pieces)
    return results
# EVOLVE-BLOCK-END
