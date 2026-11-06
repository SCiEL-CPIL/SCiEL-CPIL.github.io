    def construct():
        start_time = time.time()
        deadline = start_time + 140.0  # Safety margin
        
        n = 40
        target_edges = 401
        
        # Helper to compute book size for a single edge (u, v)
        def book_size(adj, u, v):
            # adj is list of uint64 masks
            # Common neighbors are bits set in adj[u] AND adj[v]
            # popcount gives the number
            return (adj[u] & adj[v]).bit_count()
            
        # Helper to get max book size and the edge
        def get_max_book(adj, edges):
            max_val = -1
            max_edge = None
            for u, v in edges:
                bs = book_size(adj, u, v)
                if bs > max_val:
                    max_val = bs
                    max_edge = (u, v)
            return max_val, max_edge

        # Generate a random d-regular graph using configuration model + swaps
        def generate_random_regular(n, d):
            # Create stubs
            stubs = []
            for i in range(n):
                for _ in range(d):
                    stubs.append(i)
            
            random.shuffle(stubs)
            
            # Pair up stubs
            adj = [0] * n
            edges = []
            
            for i in range(0, len(stubs), 2):
                u = stubs[i]
                v = stubs[i+1]
                if u == v:
                    # Self loop, swap with another stub
                    # Find a valid partner
                    # Simple approach: just retry or swap with random other stub
                    # For simplicity, we'll do a repair phase
                    pass 
                # Add edge
                # We'll build the graph and then repair
                # Actually, let's just build the multigraph first
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges.append((u, v))
                
            # Repair phase: remove loops and multi-edges
            # We need to ensure simple graph
            # This is complex. 
            # Alternative: Start with a random permutation based regular graph?
            # Or use the "swap" method on a known regular graph?
            # Let's use the swap method on a random graph, but ensure it becomes regular?
            # No, the configuration model is standard.
            # Let's implement a simple repair:
            # While there are loops or multi-edges:
            #   Find one.
            #   Find two edges (u,v) and (x,y) such that swapping to (u,x) and (v,y) fixes it.
            
            # Simpler: Use the "random regular graph" generator from networkx if available? 
            # No, std lib only.
            
            # Let's stick to the swap method starting from a random graph?
            # No, random graph is not regular.
            
            # Let's try a different init:
            # Random graph with 401 edges.
            # Then use "degree balancing" swaps?
            # No, the task doesn't require regularity.
            # It just requires 401 edges.
            # So we can just generate a random graph with 401 edges!
            
            # Why did the previous program use regular?
            # Maybe to have a structured start?
            # Random graph with 401 edges:
            # Expected degree ~ 20.
            # This is fine.
            
            # Let's discard the regular graph generation and just use random graph.
            
            # Generate random graph with 401 edges
            all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
            random.shuffle(all_edges)
            edges = all_edges[:target_edges]
            
            adj = [0] * n
            for u, v in edges:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
            return adj, edges

        def add_min_book_edge(adj, edges):
            # Not needed if we generate exactly 401 edges
            pass

        def local_search(adj, edges, deadline):
            current_score, _ = get_max_book(adj, edges)
            best_score = current_score
            best_adj = adj.copy()
            best_edges = edges.copy()
            
            T = 1.0
            steps = 0
            
            # Pre-compute list of non-edges?
            # There are 780 - 401 = 379 non-edges.
            # We can maintain a set of non-edges for fast sampling.
            # Or just sample random u, v and check.
            
            while time.time() < deadline:
                steps += 1
                
                # Pick random edge to remove
                u1, v1 = edges[random.randint(0, len(edges)-1)]
                
                # Pick random non-edge to add
                # Try a few times
                found = False
                for _ in range(10):
                    u2, v2 = random.sample(range(n), 2)
                    if u2 > v2: u2, v2 = v2, u2
                    # Check if non-edge
                    if not (adj[u2] & (1 << v2)):
                        found = True
                        break
                
                if not found:
                    continue
                    
                # Check if swap creates self-loop or multi-edge
                # Removing (u1, v1) and adding (u2, v2)
                # If u2 == u1 and v2 == v1, no change.
                # If (u2, v2) is already an edge, we skip (handled by check above).
                
                # Apply swap
                # Remove (u1, v1)
                adj[u1] &= ~(1 << v1)
                adj[v1] &= ~(1 << u1)
                edges.remove((u1, v1))
                
                # Add (u2, v2)
                adj[u2] |= (1 << v2)
                adj[v2] |= (1 << u2)
                edges.append((u2, v2))
                
                new_score, _ = get_max_book(adj, edges)
                
                delta = new_score - current_score
                if delta < 0 or random.random() < np.exp(-delta / T):
                    current_score = new_score
                    if current_score < best_score:
                        best_score = current_score
                        best_adj = adj.copy()
                        best_edges = edges.copy()
                else:
                    # Revert
                    adj[u2] &= ~(1 << v2)
                    adj[v2] &= ~(1 << u2)
                    edges.remove((u2, v2))
                    
                    adj[u1] |= (1 << v1)
                    adj[v1] |= (1 << u1)
                    edges.append((u1, v1))
                    
                T *= 0.999
                if T < 0.01:
                    T = 0.01
                    
            return best_adj, best_edges

        best_adj = None
        best_score = float('inf')
        
        # Try a few random starts
        for i in range(10):
            if time.time() > deadline - 5:
                break
                
            adj, edges = generate_random_regular(n, 20) # Name is misleading, it's random graph
            adj, edges = local_search(adj, edges, deadline - 5)
            
            score, _ = get_max_book(adj, edges)
            
            if score < best_score:
                best_score = score
                best_adj = adj
                best_edges = edges
                
        # Convert best_adj to list of edges
        if best_adj is None:
            return [[a, b] for a in range(20) for b in range(20, 40)] + [[0, 1]]
            
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if best_adj[i] & (1 << j):
                    edges.append([i, j])
                
        return edges
    