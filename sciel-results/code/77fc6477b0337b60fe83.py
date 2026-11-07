import time

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}


def construct():
    """
    Triangle-free diameter-four augmentation.
    
    Strategy: The base graph is a path P24 (vertices 0..23). We need to add edges
    to reduce diameter to <= 4 while keeping the graph triangle-free.
    
    Key insight: In a triangle-free graph, we can use a "double-star" or "bipartite
    hub" structure. If we pick two non-adjacent hub vertices h1 and h2, and connect
    them to many leaves, we can cover the path efficiently.
    
    Actually, let's think more carefully. A path of length 23 needs diameter <= 4.
    Without added edges, diameter is 23.
    
    For a triangle-free graph with n=24 and diameter 4, what's the minimum number of
    added edges?
    
    Let's try a constructive approach:
    - Pick hub vertices that are far apart on the path to avoid triangles.
    - If we connect vertex 6 to vertices 0, 12, 18, 24... wait, max vertex is 23.
    
    Let me think about this differently. The path is 0-1-2-...-23.
    
    If I add edge (6, 12), that creates a cycle 6-7-8-9-10-11-12-6, which is fine (no triangle).
    But I need to be careful: (6,12) plus path edges 6-7 and 11-12 don't create triangles.
    A triangle would need three mutually adjacent vertices. Since the path only has consecutive edges,
    a triangle can only be formed if we add an edge (u,v) where v = u+2 (creating triangle u, u+1, u+2)
    or if we add multiple edges that form a triangle.
    
    So the constraint is: we cannot add edge (u, u+2) because that creates triangle (u, u+1, u+2).
    More generally, if we add edges (a,b) and (b,c), we cannot also have (a,c) if they form a triangle.
    But since the base is a path, the only triangles possible are from adding an edge that skips exactly one vertex.
    
    Wait, that's not quite right. If we add (0, 2), triangle (0,1,2) forms. If we add (0,3), no triangle
    with base edges alone. But if we also add (1,3), then (0,1,3) is not a triangle unless (0,3) exists.
    
    Let me just do a search with proper triangle checking.
    
    Approach: Greedy with backtracking or smart greedy.
    
    Actually, let me think about what diameter 4 means. From any vertex, we can reach any other in <= 4 steps.
    
    For a path of 24 vertices, the center is around vertex 11-12. 
    Vertex 0 needs to reach vertex 23 in <= 4 steps. Currently it takes 23 steps.
    
    If I add edge (0, 12): dist(0,23) becomes min(23, 1 + dist(12,23)) = min(23, 1+11) = 12. Still too big.
    If I add (0, 12) and (12, 23): dist(0,23) = 2. But (12,23) - does it create a triangle? 
    Path has 11-12, 12-13, ..., 22-23. Adding (12,23) - no triangle since 12 and 23 are not distance 2.
    
    But wait, can I add (0,12) and (12,23)? Let's check triangles:
    - (0,12): neighbors of 0 are {1, 12}. Neighbors of 12 are {11, 13, 0, 23}. Common neighbors: none (1 is not neighbor of 12, 12 is not neighbor of 0 in path... wait 0's path neighbor is 1, 12's path neighbors are 11,13). So no triangle from (0,12) alone.
    - (12,23): neighbors of 12 are {11,13,0,23}. Neighbors of 23 are {22,12}. Common: none. So no triangle.
    
    Now with both: is there a triangle? We need three vertices all mutually adjacent.
    Check if any pair among {0,12,23} forms issues... 0-12 edge, 12-23 edge, 0-23? No edge. So no triangle.
    
    Let me try to find a small set of edges.
    
    Let me use a BFS-based approach to find a valid solution with few edges.
    """
    n = P['n']
    deadline = time.time() + 80  # Leave 10s margin
    
    # Build base adjacency
    base_adj = [[False] * n for _ in range(n)]
    for u, v in P['edges']:
        base_adj[u][v] = True
        base_adj[v][u] = True
    
    def is_triangle_free(adj, u, v):
        """Check if adding edge (u,v) creates a triangle."""
        if adj[u][v]:
            return False  # Already exists
        for w in range(n):
            if adj[u][w] and adj[v][w]:
                return False
        return True
    
    def compute_diameter(adj):
        """Compute diameter using BFS from each vertex."""
        max_dist = 0
        for src in range(n):
            dist = [-1] * n
            dist[src] = 0
            queue = [src]
            head = 0
            while head < len(queue):
                u = queue[head]
                head += 1
                for v in range(n):
                    if adj[u][v] and dist[v] == -1:
                        dist[v] = dist[u] + 1
                        queue.append(v)
            for d in dist:
                if d == -1:
                    return n + 1  # Not connected
                if d > max_dist:
                    max_dist = d
        return max_dist
    
    # Try greedy: start with no edges, add edges that most reduce diameter
    # First, let's try a known good structure
    
    # Strategy: Use a small number of "long-range" edges to create shortcuts.
    # For diameter 4 on a path of 24, we need significant shortcuts.
    
    # Let's try: pick hub vertices and connect them strategically.
    # A bipartite approach: divide vertices into two sets, add edges only between sets.
    
    # Actually, let me just do an intelligent search.
    # Candidate edges: all non-path edges that don't create triangles.
    
    # Generate candidate edges (sorted by potential usefulness)
    candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if not base_adj[u][v]:
                candidates.append((u, v))
    
    # Sort candidates: prefer edges that are "long" (large distance) as they reduce diameter more
    # But also prefer edges that don't conflict
    def edge_distance(u, v):
        return abs(u - v)
    
    candidates.sort(key=lambda e: -edge_distance(e[0], e[1]))
    
    best_solution = None
    best_count = n  # Upper bound
    
    # Greedy: try adding edges one by one, keeping track of best
    adj = [row[:] for row in base_adj]
    added = []
    
    # First, compute initial diameter
    current_diam = compute_diameter(adj)
    
    # Greedy addition
    for u, v in candidates:
        if time.time() > deadline:
            break
        if len(added) >= best_count:
            break
        if not is_triangle_free(adj, u, v):
            continue
        # Try adding this edge
        adj[u][v] = True
        adj[v][u] = True
        added.append((u, v))
        
        new_diam = compute_diameter(adj)
        if new_diam <= P['diameter']:
            if len(added) < best_count:
                best_count = len(added)
                best_solution = [list(e) for e in added]
            # Try to remove some edges
            # Remove last few edges and see if still valid
            for remove_idx in range(len(added) - 1, -1, -1):
                if time.time() > deadline:
                    break
                ru, rv = added[remove_idx]
                # Temporarily remove
                adj[ru][rv] = False
                adj[rv][ru] = False
                saved = added[remove_idx]
                added.pop(remove_idx)
                
                if compute_diameter(adj) <= P['diameter']:
                    if len(added) < best_count:
                        best_count = len(added)
                        best_solution = [list(e) for e in added]
                    # Try removing more
                    for remove_idx2 in range(len(added) - 1, -1, -1):
                        if time.time() > deadline:
                            break
                        ru2, rv2 = added[remove_idx2]
                        adj[ru2][rv2] = False
                        adj[rv2][ru2] = False
                        added.pop(remove_idx2)
                        if compute_diameter(adj) <= P['diameter']:
                            if len(added) < best_count:
                                best_count = len(added)
                                best_solution = [list(e) for e in added]
                        else:
                            # Put back
                            added.insert(remove_idx2, (ru2, rv2))
                            adj[ru2][rv2] = True
                            adj[rv2][ru2] = True
                else:
                    # Put back
                    added.insert(remove_idx, saved)
                    adj[ru][rv] = True
                    adj[rv][ru] = True
                break
        else:
            # Remove it, not useful
            adj[u][v] = False
            adj[v][u] = False
            added.pop()
    
    # Try a different strategy: specific hub-based constructions
    # Strategy 2: Pick 2-3 hub vertices, connect them to many others
    # Hubs should be spaced out to avoid triangles
    
    # Try hub at 6 and 17 (distance 11, no triangle issues)
    # Connect 6 to: 0, 12, 18, 23 (check triangles)
    # Connect 17 to: 3, 9, 15, 21
    
    # Let me try several specific configurations
    specific_configs = [
        # Config 1: Two hubs
        [(6, 0), (6, 12), (6, 18), (6, 23), (17, 3), (17, 9), (17, 15), (17, 21)],
        # Config 2: 
        [(5, 0), (5, 10), (5, 15), (5, 20), (18, 3), (18, 8), (18, 13), (18, 23)],
        # Config 3: Three hubs
        [(4, 0), (4, 8), (4, 12), (4, 16), (4, 20), (14, 2), (14, 6), (14, 10), (14, 18), (14, 22)],
        # Config 4: 
        [(7, 0), (7, 14), (7, 21), (16, 3), (16, 10), (16, 23)],
        # Config 5:
        [(8, 0), (8, 16), (8, 23), (15, 2), (15, 7), (15, 20)],
        # Config 6:
        [(6, 0), (6, 12), (6, 18), (6, 23), (17, 4), (17, 10), (17, 16), (17, 22)],
        # Config 7: Minimal attempt
        [(6, 0), (6, 12), (6, 18), (6, 23), (17, 3), (17, 10), (17, 17)],  # invalid, skip
        # Config 8:
        [(5, 0), (5, 10), (5, 15), (5, 20), (18, 3), (18, 8), (18, 13), (18, 22)],
    ]
    
    for config in specific_configs:
        if time.time() > deadline:
            break
        # Validate this config
        adj_test = [row[:] for row in base_adj]
        valid = True
        for u, v in config:
            if u == v or u >= n or v >= n:
                valid = False
                break
            if not is_triangle_free(adj_test, u, v):
                valid = False
                break
            adj_test[u][v] = True
            adj_test[v][u] = True
        
        if valid:
            diam = compute_diameter(adj_test)
            if diam <= P['diameter'] and len(config) < best_count:
                best_count = len(config)
                best_solution = [list(e) for e in config]
    
    # Try even smaller configs
    # Can we do it with 4 edges? 5 edges?
    # With 4 edges, each edge can reduce distance significantly.
    # Let's try random search for small solutions
    
    import random
    random.seed(42)
    
    # Try random configurations with 4-6 edges
    for trial in range(5000):
        if time.time() > deadline:
            break
        num_edges = 4 + (trial % 3)  # Try 4, 5, 6 edges
        random_edges = random.sample(candidates, num_edges)
        
        adj_test = [row[:] for row in base_adj]
        valid = True
        for u, v in random_edges:
            if not is_triangle_free(adj_test, u, v):
                valid = False
                break
            adj_test[u][v] = True
            adj_test[v][u] = True
        
        if valid:
            diam = compute_diameter(adj_test)
            if diam <= P['diameter'] and num_edges < best_count:
                best_count = num_edges
                best_solution = [list(e) for e in random_edges]
    
    # Final fallback: use the known 8-edge solution
    if best_solution is None or best_count > 8:
        fallback = [[12, 1], [12, 3], [12, 6], [12, 9], [12, 15], [12, 18], [12, 21], [12, 23]]
        # Verify fallback
        adj_test = [row[:] for row in base_adj]
        valid = True
        for u, v in fallback:
            if not is_triangle_free(adj_test, u, v):
                valid = False
                break
            adj_test[u][v] = True
            adj_test[v][u] = True
        if valid and compute_diameter(adj_test) <= P['diameter']:
            best_solution = fallback
            best_count = 8
    
    # Return as list of lists
    if best_solution is None:
        return []
    
    # Ensure all are Python ints in lists
    result = []
    for u, v in best_solution:
        result.append([int(u), int(v)])
    
    return result
