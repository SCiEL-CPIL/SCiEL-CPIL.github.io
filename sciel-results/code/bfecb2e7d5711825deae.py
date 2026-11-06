    def construct():
        start_time = time.time()
        deadline = start_time + 115
        n = 16
        
        # Precompute subsets of size 4
        subsets_4 = list(itertools.combinations(range(n), 4))
        
        def find_q3(adj_masks):
            # ... implementation as above ...
            pass
            
        best_edges = set()
        best_count = 0
        
        for restart in range(50):
            if time.time() > deadline - 5:
                break
            
            # Random initialization
            adj_masks = [0] * n
            current_edges = set()
            
            # Start with p=0.5
            for u in range(n):
                for v in range(u+1, n):
                    if random.random() < 0.5:
                        adj_masks[u] |= (1 << v)
                        adj_masks[v] |= (1 << u)
                        current_edges.add((u, v))
            
            # Repair phase
            while time.time() < deadline - 5:
                q3 = find_q3(adj_masks)
                if q3 is None:
                    break
                X, Y = q3
                # Remove a random edge from the induced subgraph on X U Y
                # The Q3 edges are those between X and Y where the edge exists.
                # We need to remove an edge that is part of the Q3 structure.
                # Any edge in the bipartite graph between X and Y that is present can be removed.
                # Removing any edge from the Q3 breaks this specific Q3, but might leave others.
                
                # Collect all edges between X and Y
                edges_in_q3 = []
                for x in X:
                    for y in Y:
                        if (adj_masks[x] >> y) & 1:
                            edges_in_q3.append((x, y))
                
                if not edges_in_q3:
                    break # Should not happen
                    
                u, v = random.choice(edges_in_q3)
                adj_masks[u] &= ~(1 << v)
                adj_masks[v] &= ~(1 << u)
                current_edges.remove((u, v))
            
            # Local Search: Add edges
            improved = True
            while improved and time.time() < deadline - 5:
                improved = False
                # Try all non-edges
                for u in range(n):
                    for v in range(u+1, n):
                        if time.time() > deadline - 5:
                            break
                        if (u, v) in current_edges:
                            continue
                        
                        adj_masks[u] |= (1 << v)
                        adj_masks[v] |= (1 << u)
                        
                        if find_q3(adj_masks) is None:
                            current_edges.add((u, v))
                            improved = True
                            break
                        else:
                            adj_masks[u] &= ~(1 << v)
                            adj_masks[v] &= ~(1 << u)
                    if time.time() > deadline - 5:
                        break
                
                if time.time() > deadline - 5:
                    break
            
            if len(current_edges) > best_count:
                best_count = len(current_edges)
                best_edges = set(current_edges)
                
        return [[u, v] for u, v in best_edges]
    