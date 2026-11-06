import time

def construct():
    t0 = time.time()
    deadline = t0 + 140  # safety margin

    N = 16
    ALL = (1 << N) - 1

    # Precompute edge list
    edges_list = [(i, j) for i in range(N) for j in range(i+1, N)]
    M = len(edges_list)

    # For each edge (u,v), precompute the set of edges that would create a triangle
    # i.e., edges (u,w) or (v,w) where w is a common neighbor candidate
    # Actually, for triangle-free: adding edge (u,v) is forbidden if there exists w
    # such that (u,w) and (v,w) are both present.
    # So for each edge e=(u,v), the "conflict" edges are those sharing a vertex with e
    # and forming a triangle. Specifically, edge (u,v) conflicts with edge (u,w) iff
    # edge (v,w) is present, and vice versa.
    # Better: for each edge index, precompute which other edge indices would form
    # a triangle with it. Edge (u,v) and edge (u,w) form triangle if (v,w) present.
    # So edge e1=(u,v) conflicts with e2=(u,w) when e3=(v,w) is also present.
    # This is a 3-way constraint. Let's use a different approach.

    # For triangle-free: when adding edge (u,v), check if u and v have a common neighbor.
    # We'll maintain adjacency bitmasks.

    def is_3_colorable(adj_mask):
        """Check if graph with adjacency bitmasks is 3-colorable using backtracking with MRV."""
        color = [-1] * N
        # Precompute neighbor lists
        neighbors = [i for i in range(N)]
        
        def get_degree(v):
            return bin(adj_mask[v]).count('1')
        
        def backtrack(assigned_count):
            if assigned_count == N:
                return True
            
            # MRV: pick uncolored vertex with fewest available colors
            best_v = -1
            best_avail = 4
            best_forbidden = 0
            for v in range(N):
                if color[v] != -1:
                    continue
                forbidden = 0
                for c in range(3):
                    # Check if any colored neighbor has color c
                    for w in range(N):
                        if color[w] == c and ((adj_mask[v] >> w) & 1):
                            forbidden |= (1 << c)
                            break
                avail = 3 - bin(forbidden).count('1')
                if avail < best_avail:
                    best_avail = avail
                    best_v = v
                    best_forbidden = forbidden
                    if avail <= 0:
                        return False
            
            if best_v == -1:
                return True  # all colored
            
            for c in range(3):
                if not (best_forbidden & (1 << c)):
                    color[best_v] = c
                    if backtrack(assigned_count + 1):
                        return True
                    color[best_v] = -1
            return False
        
        return backtrack(0)

    def is_3_colorable_fast(adj_mask):
        """Faster 3-colorability check using bitmasks for color classes."""
        # Try to assign colors 0,1,2 to vertices
        # Use recursive backtracking with forward checking
        color = [-1] * N
        # Order vertices by degree descending for initial assignment
        order = sorted(range(N), key=lambda x: bin(adj_mask[x]).count('1'), reverse=True)
        
        # For each vertex, precompute which other vertices in the order are its neighbors
        # and come after it in the order
        pos = {v: i for i, v in enumerate(order)}
        
        def backtrack(idx):
            if idx == N:
                return True
            v = order[idx]
            # Find forbidden colors from already-colored neighbors
            forbidden = 0
            for j in range(idx):
                w = order[j]
                if ((adj_mask[v] >> w) & 1):
                    forbidden |= (1 << color[w])
            for c in range(3):
                if not (forbidden & (1 << c)):
                    color[v] = c
                    if backtrack(idx + 1):
                        return True
                    color[v] = -1
            return False
        
        return backtrack(0)

    def is_3_colorable_memo(adj_mask):
        """3-colorability with bitmasks for color classes - very fast."""
        # Represent coloring as three bitmasks: mask0, mask1, mask2
        # where mask_i has bit v set if vertex v has color i
        # Unassigned vertices have no bit set in any mask
        
        # We'll use a different approach: try all ways to assign color 0 to an independent set,
        # then check if the remaining is 2-colorable (i.e., bipartite).
        
        # Actually, let's just do efficient backtracking with bit operations.
        # State: for each vertex, its color (0,1,2) or -1.
        # Use the ordering by degree.
        
        order = sorted(range(N), key=lambda x: bin(adj_mask[x]).count('1'), reverse=True)
        color = [-1] * N
        
        # Precompute for each vertex in order, which earlier vertices in order are neighbors
        pred_neighbors = [[] for _ in range(N)]
        for i in range(N):
            v = order[i]
            for j in range(i):
                w = order[j]
                if ((adj_mask[v] >> w) & 1):
                    pred_neighbors[i].append(j)
        
        def backtrack(idx):
            if idx == N:
                return True
            v = order[idx]
            # Forbidden colors from predecessors
            forbidden = 0
            for j in pred_neighbors[idx]:
                forbidden |= (1 << color[order[j]])
            
            for c in range(3):
                if not (forbidden & (1 << c)):
                    color[v] = c
                    if backtrack(idx + 1):
                        return True
                    color[v] = -1
            return False
        
        return backtrack(0)

    def count_edges(adj_mask):
        return sum(bin(adj_mask[i]).count('1') for i in range(N)) // 2

    def add_edge(adj_mask, u, v):
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)

    def can_add_edge(adj_mask, u, v):
        """Check if adding edge (u,v) keeps graph triangle-free."""
        if (adj_mask[u] >> v) & 1:
            return False  # already present
        # Check for common neighbor
        common = adj_mask[u] & adj_mask[v]
        return common == 0

    # Start with the Grötzsch graph (11 vertices, 20 edges, 4-chromatic, triangle-free)
    def build_groetzsch():
        adj = [0] * N
        # 5-cycle: 0-1-2-3-4-0
        for i in range(5):
            add_edge(adj, i, (i+1) % 5)
        # Chords: 5-1, 5-3; 6-2, 6-4; 7-3, 7-0; 8-4, 8-1; 9-0, 9-2
        add_edge(adj, 5, 1); add_edge(adj, 5, 3)
        add_edge(adj, 6, 2); add_edge(adj, 6, 4)
        add_edge(adj, 7, 3); add_edge(adj, 7, 0)
        add_edge(adj, 8, 4); add_edge(adj, 8, 1)
        add_edge(adj, 9, 0); add_edge(adj, 9, 2)
        # Vertex 10 connects to 5,6,7,8,9
        for v in range(5, 10):
            add_edge(adj, 10, v)
        return adj

    # Build initial 4-chromatic graph
    best_adj = build_groetzsch()
    best_count = count_edges(best_adj)  # 20 edges

    # Verify it's not 3-colorable
    assert not is_3_colorable_memo(best_adj), "Grötzsch should be 4-chromatic"

    # Now try to add edges to maximize count while maintaining:
    # 1. Triangle-free
    # 2. Not 3-colorable
    
    # Strategy: Try to add edges one by one, always checking both constraints.
    # Use a greedy approach with multiple restarts and different orderings.
    
    def try_add_edges(adj, edge_order):
        """Try to add edges in the given order, maintaining constraints."""
        current = adj[:]
        current_count = count_edges(current)
        
        for (u, v) in edge_order:
            if time.time() > deadline:
                break
            if can_add_edge(current, u, v):
                new_adj = current[:]
                add_edge(new_adj, u, v)
                # Check 3-colorability
                if not is_3_colorable_memo(new_adj):
                    current = new_adj
                    current_count += 1
        return current, current_count

    # Generate different edge orderings
    import random
    rng = random.Random(42)
    
    best_result = best_adj[:]
    best_result_count = best_count

    # Multiple attempts with different strategies
    for attempt in range(20):
        if time.time() > deadline:
            break
        
        # Create edge order: start with edges involving new vertices (11-15) first,
        # then edges among existing vertices, shuffled
        new_edges = [(u, v) for (u, v) in edges_list if u >= 11 or v >= 11]
        old_edges = [(u, v) for (u, v) in edges_list if u < 11 and v < 11]
        
        # Remove edges already present
        present = set()
        for i in range(N):
            for j in range(i+1, N):
                if (best_adj[i] >> j) & 1:
                    present.add((i, j))
        
        new_edges = [e for e in new_edges if e not in present]
        old_edges = [e for e in old_edges if e not in present]
        
        # Shuffle
        rng.shuffle(new_edges)
        rng.shuffle(old_edges)
        
        # Different orderings
        if attempt % 3 == 0:
            edge_order = new_edges + old_edges
        elif attempt % 3 == 1:
            edge_order = old_edges + new_edges
        else:
            edge_order = new_edges + old_edges
            rng.shuffle(edge_order)
        
        result, count = try_add_edges(best_adj, edge_order)
        if count > best_result_count:
            best_result = result
            best_result_count = count

    # Local search: try to add more edges by trying all possible additions
    # and also try removing an edge to allow two others
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try adding each missing edge
        for (u, v) in edges_list:
            if time.time() > deadline:
                break
            if (best_result[u] >> v) & 1:
                continue
            if can_add_edge(best_result, u, v):
                new_adj = best_result[:]
                add_edge(new_adj, u, v)
                if not is_3_colorable_memo(new_adj):
                    best_result = new_adj
                    best_result_count += 1
                    improved = True
                    break  # restart inner loop
        
        if time.time() > deadline:
            break
        
        # Try edge swaps: remove one edge, add two
        if not improved:
            for (u, v) in edges_list:
                if time.time() > deadline:
                    break
                if not ((best_result[u] >> v) & 1):
                    continue
                # Try removing this edge
                test_adj = best_result[:]
                test_adj[u] &= ~(1 << v)
                test_adj[v] &= ~(1 << u)
                
                # Now try to add two edges
                added = 0
                for (a, b) in edges_list:
                    if time.time() > deadline:
                        break
                    if a == u or a == v or b == u or b == v:
                        continue
                    if (test_adj[a] >> b) & 1:
                        continue
                    if can_add_edge(test_adj, a, b):
                        test2 = test_adj[:]
                        add_edge(test2, a, b)
                        if not is_3_colorable_memo(test2):
                            test_adj = test2
                            added += 1
                            if added >= 2:
                                break
                
                if added >= 2:
                    best_result = test_adj
                    best_result_count += added - 1
                    improved = True
                    break

    # Final verification
    # Check triangle-free
    for u in range(N):
        for v in range(u+1, N):
            if (best_result[u] >> v) & 1:
                common = best_result[u] & best_result[v]
                if common != 0:
                    # Remove problematic edges
                    pass  # Should not happen

    # Check not 3-colorable
    if is_3_colorable_memo(best_result):
        # Fallback to Grötzsch
        best_result = build_groetzsch()
        best_result_count = count_edges(best_result)

    # Convert to edge list
    edges = []
    for u in range(N):
        for v in range(u+1, N):
            if (best_result[u] >> v) & 1:
                edges.append([u, v])

    return edges
