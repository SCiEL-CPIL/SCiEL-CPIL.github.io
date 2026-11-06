def construct():
    import time
    import random
    import numpy as np

    start_time = time.time()
    DEADLINE = 115.0
    n = 31
    target_edges = 91

    def time_left():
        return DEADLINE - (time.time() - start_time)

    def count_c4s(adj):
        """
        Count the number of C4s in the graph defined by adjacency matrix adj.
        A C4 is a cycle of length 4.
        Number of C4s = sum over all pairs (u, v) of (number of common neighbors of u and v choose 2).
        Wait, this formula counts C4s where (u,v) is a diagonal?
        Let's verify.
        A C4 is u-v-w-x-u.
        The pairs of opposite vertices are (u,w) and (v,x).
        For pair (u,w), the common neighbors are v and x.
        So for each pair of vertices (u,w), let c(u,w) be the number of common neighbors.
        The number of C4s where u and w are opposite is c(u,w) * (c(u,w) - 1) / 2.
        Summing this over all pairs (u,w) counts each C4 exactly twice (once for each diagonal pair).
        So total C4s = 1/2 * sum_{u < w} (c(u,w) * (c(u,w) - 1) / 2).
        """
        total = 0
        # adj is a numpy array for speed
        # common[i, j] = sum(adj[i] * adj[j])
        # We can compute this matrix product
        common = adj @ adj
        # common[i, j] is the number of common neighbors of i and j
        # We need sum over i < j of binom(common[i, j], 2)
        # Note: common[i, i] is degree(i), but we only care about i != j.
        # Also, if adj[i, j] is 1, they are neighbors.
        
        # Iterate over upper triangle
        for i in range(n):
            for j in range(i + 1, n):
                c = common[i, j]
                if c >= 2:
                    total += c * (c - 1) // 2
        return total // 2

    def creates_c4_count(adj, u, v):
        """
        Return the number of C4s created by adding edge (u, v).
        This is binom(common_neighbors(u, v), 2).
        """
        c = np.sum(adj[u] * adj[v])
        return c * (c - 1) // 2

    def remove_edge_c4_delta(adj, u, v):
        """
        Return the number of C4s removed by removing edge (u, v).
        This is the number of C4s containing edge (u, v).
        A C4 containing (u, v) looks like u-v-w-x-u or u-v-x-w-u.
        This is equivalent to finding common neighbors of u and v? No.
        If we have edge (u, v), a C4 containing it must have path u-v-w-x-u.
        This implies w is neighbor of v, x is neighbor of u, and w-x is edge.
        Also w != u, x != v.
        
        Actually, it's easier to just recount if n is small, but we can optimize.
        Number of C4s containing edge (u, v) = sum_{w in N(v), w != u} sum_{x in N(u), x != v} [w != x and adj[w, x] == 1].
        Wait, w and x must be distinct? Yes, for a simple cycle.
        Also w cannot be x.
        
        Let's just use the global count for simplicity in the swap logic if it's fast enough,
        or implement a specific delta.
        Given n=31, global count is ~31*31/2 * 31 ops ~ 15k ops. Very fast.
        """
        pass

    # 1. Generate a good starting graph
    best_edges = None
    best_c4_count = float('inf')

    def greedy_construct():
        """
        Greedily add edges to avoid C4s.
        """
        adj = np.zeros((n, n), dtype=int)
        edges = set()
        
        # Order pairs randomly
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(pairs)
        
        for u, v in pairs:
            if len(edges) >= 90:
                break
            # Check if adding (u, v) creates a C4
            # It creates a C4 if u and v have >= 2 common neighbors.
            c = np.sum(adj[u] * adj[v])
            if c < 2:
                adj[u, v] = 1
                adj[v, u] = 1
                edges.add((u, v))
                
        return adj, edges

    # Try multiple random restarts for greedy construction to find a 90-edge C4-free graph
    # If we can't find 90, we take the best we have and fill up.
    # But ex(31, C4) = 90, so it should be possible.
    
    attempts = 0
    while time_left() > 10 and attempts < 50:
        adj, edges = greedy_construct()
        if len(edges) == 90:
            break
        attempts += 1
        
    # If we didn't get 90 edges, we have a graph with < 90 edges.
    # We need to add edges to reach 91.
    # We will fill the remaining edges with those that create the fewest C4s.
    
    if len(edges) < 90:
        # Add edges that create 0 C4s first (common neighbors < 2)
        # Then edges that create 1 C4 (common neighbors = 2)
        # etc.
        
        candidates = [(i, j) for i in range(n) for j in range(i + 1, n) if (i, j) not in edges]
        # Score candidates by number of C4s created
        def score_candidate(u, v):
            c = np.sum(adj[u] * adj[v])
            return c * (c - 1) // 2
            
        # Sort candidates by score
        candidates.sort(key=lambda p: score_candidate(p[0], p[1]))
        
        for u, v in candidates:
            if len(edges) >= 91:
                break
            adj[u, v] = 1
            adj[v, u] = 1
            edges.add((u, v))
            
    # Now we have exactly 91 edges (or we stop if we can't reach 91, but we can always add edges)
    # Ensure we have exactly 91
    if len(edges) < 91:
        # This shouldn't happen as complete graph has 465 edges
        all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n) if (i, j) not in edges]
        for u, v in all_pairs:
            if len(edges) >= 91:
                break
            adj[u, v] = 1
            adj[v, u] = 1
            edges.add((u, v))
            
    if len(edges) > 91:
        # Remove excess edges (shouldn't happen with the logic above, but safety)
        edges_list = list(edges)
        random.shuffle(edges_list)
        for u, v in edges_list:
            if len(edges) > 91:
                adj[u, v] = 0
                adj[v, u] = 0
                edges.remove((u, v))
            else:
                break

    # Initial C4 count
    current_c4 = count_c4s(adj)
    best_edges_set = set(edges)
    best_adj = adj.copy()
    best_c4_count = current_c4

    # 2. Local Search: Swap edges to reduce C4 count
    # We maintain a set of edges.
    # Move: Remove edge (u, v), add edge (x, y).
    # Constraint: (x, y) not in edges, (u, v) in edges.
    # Objective: Minimize total C4s.
    
    # Precompute neighbors for speed? Maybe not needed for n=31.
    
    while time_left() > 1:
        # Try a number of random swaps
        # To be efficient, we can sample a few edges to remove
        edge_list = list(edges)
        random.shuffle(edge_list)
        
        # Limit the number of swaps attempted per "iteration" to avoid getting stuck
        # and to allow more global restarts if needed.
        # But since we only have one main run, let's do many swaps.
        
        improved = False
        # We'll try to find a swap that improves the score.
        # Randomly pick an edge to remove.
        for _ in range(50): # 50 attempts per loop
            if time_left() < 1:
                break
                
            if not edge_list:
                break
            u, v = random.choice(edge_list)
            
            # Remove (u, v)
            adj[u, v] = 0
            adj[v, u] =  0
            edges.remove((u, v))
            
            # Find best edge to add
            # We want to add an edge (x, y) that minimizes the new C4 count.
            # New C4 count = Old C4 count - C4s_removed(u, v) + C4s_added(x, y)
            # C4s_removed(u, v) is the number of C4s containing (u, v).
            # C4s_added(x, y) is the number of C4s created by (x, y).
            
            # Calculate C4s_removed(u, v)
            # A C4 containing (u, v) has form u-v-w-x-u.
            # w is neighbor of v, x is neighbor of u.
            # w != u, x != v.
            # w != x.
            # adj[w, x] must be 1.
            
            # Let's just calculate the delta by counting C4s in the modified graph?
            # That's O(N^3) per swap. 31^3 = 30k. 50 swaps * 30k = 1.5M ops. Fast.
            # But we can do better.
            
            # Let's calculate C4s_added(x, y) for all candidates (x, y) not in edges.
            # This is O(N^2 * N) = O(N^3) to evaluate all candidates.
            # Then we pick the best one.
            
            best_delta = float('inf')
            best_add = None
            
            # Candidates for addition: all non-edges
            # We can precompute common neighbors for all pairs?
            # common = adj @ adj
            # But adj changes.
            
            # Let's just iterate over non-edges
            # To speed up, we can just check a subset of non-edges if time is tight.
            # But n=31 is small.
            
            # Calculate C4s removed by (u, v)
            # c4s_removed = 0
            # for w in range(n):
            #    if adj[v, w] and w != u:
            #       for x in range(n):
            #           if adj[u, x] and x != v and x != w and adj[w, x]:
            #               c4s_removed += 1
            # Note: This counts each C4 once?
            # C4: u-v-w-x-u.
            # w is neighbor of v. x is neighbor of u.
            # Is it possible to count the same C4 twice?
            # The cycle is u-v-w-x-u.
            # w is fixed as neighbor of v. x is fixed as neighbor of u.
            # The pair (w, x) is unique for this specific traversal direction?
            # Actually, the C4 is determined by the set of vertices {u, v, w, x}.
            # The edges are (u,v), (v,w), (w,x), (x,u).
            # In our loop:
            # w is a neighbor of v (edge v-w).
            # x is a neighbor of u (edge u-x).
            # We check if edge w-x exists.
            # This uniquely identifies the C4 containing edge (u,v) and vertices w, x.
            # So yes, this counts each C4 containing (u,v) exactly once.
            
            c4s_removed = 0
            for w in range(n):
                if w == u: continue
                if adj[v, w]:
                    for x in range(n):
                        if x == v or x == w: continue
                        if adj[u, x] and adj[w, x]:
                            c4s_removed += 1
                            
            # Now find best (x, y) to add
            # We want to minimize c4s_added(x, y)
            # c4s_added(x, y) = binom(common(x, y), 2)
            # We want to minimize this.
            
            min_added = float('inf')
            best_x, best_y = -1, -1
            
            # Iterate over all non-edges
            for i in range(n):
                for j in range(i + 1, n):
                    if adj[i, j]:
                        continue
                    # Calculate common neighbors
                    c = 0
                    # Optimization: if c reaches 3, binom(3,2)=3. If c=2, binom(2,2)=1.
                    # We just need the minimum.
                    # We can break early if we find 0.
                    for k in range(n):
                        if adj[i, k] and adj[j, k]:
                            c += 1
                            if c > 2: # binom(c,2) increases with c
                                break
                    added = c * (c - 1) // 2
                    if added < min_added:
                        min_added = added
                        best_x, best_y = i, j
                        if added == 0:
                            break # Can't do better than 0
                if min_added == 0:
                    break
                    
            if best_x != -1:
                delta = min_added - c4s_removed
                if delta < 0:
                    # Accept the swap
                    adj[best_x, best_y] = 1
                    adj[best_y, best_x] = 1
                    edges.add((best_x, best_y))
                    current_c4 += delta
                    improved = True
                    if current_c4 < best_c4_count:
                        best_c4_count = current_c4
                        best_edges_set = set(edges)
                        best_adj = adj.copy()
                else:
                    # Reject swap, restore state
                    # We removed (u, v), so we must add it back
                    adj[u, v] = 1
                    adj[v, u] = 1
                    edges.add((u, v))
            else:
                # No valid edge to add? (Should not happen unless graph is complete)
                adj[u, v] = 1
                adj[v, u] = 1
                edges.add((u, v))
                
        if not improved:
            # Random restart or perturb
            # If we are stuck, maybe perturb by removing a random edge and adding a random one
            # even if it increases C4 count slightly, to escape local minima.
            # Or just break and try a new greedy start if time permits.
            if time_left() > 5:
                # Perturb
                if edges:
                    u, v = random.choice(list(edges))
                    adj[u, v] = 0
                    adj[v, u] = 0
                    edges.remove((u, v))
                    
                    # Add random non-edge
                    candidates = [(i, j) for i in range(n) for j in range(i + 1, n) if not adj[i, j]]
                    if candidates:
                        x, y = random.choice(candidates)
                        adj[x, y] = 1
                        adj[y, x] = 1
                        edges.add((x, y))
                        current_c4 = count_c4s(adj)
                        if current_c4 < best_c4_count:
                            best_c4_count = current_c4
                            best_edges_set = set(edges)
                            best_adj = adj.copy()
            else:
                break

    # Return the best result found
    # Ensure we return exactly 91 edges
    result = [[u, v] for u, v in best_edges_set]
    if len(result) != 91:
        # Fallback: just return current edges if best is invalid (should not happen)
        result = [[u, v] for u, v in edges]
        
    return result
