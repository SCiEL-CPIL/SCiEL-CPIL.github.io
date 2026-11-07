# Inside construct():
    # ... existing code ...
    
    def find_embedding_backtrack(t_idx, used_edges, host_map, adj, host_degrees):
        """
        DFS to find embedding of tree t_idx into available edges.
        host_map: dict {tree_vertex: host_vertex} for vertices already placed
        host_degrees: list of length n, current degree used at each host vertex
        Returns True if found, modifies host_map in place.
        """
        k = t_idx + 1
        if k == 1:
            return True
        
        unplaced = [v for v in range(k) if v not in host_map]
        if not unplaced:
            return True
        
        # Pick the most constrained unplaced vertex
        best_v = None
        best_constraint = -1
        for v in unplaced:
            placed_neighbors = [host_map[nb] for nb in adj[v] if nb in host_map]
            # Constraint: number of placed neighbors + (degree in tree - placed neighbors) <= remaining capacity?
            # Actually, simpler constraint: for each placed neighbor, the edge must be available.
            # And the host vertex must have enough remaining degree.
            if len(placed_neighbors) > best_constraint:
                best_constraint = len(placed_neighbors)
                best_v = v
        
        if best_v is None:
            best_v = random.choice(unplaced)
        
        # Determine candidate host vertices for best_v
        candidates = set(range(n))
        
        # 1. Must be adjacent to all placed neighbors via available edges
        for nb in adj[best_v]:
            if nb in host_map:
                ph = host_map[nb]
                valid = set()
                for v in range(n):
                    if v == ph:
                        continue
                    e = (min(ph, v), max(ph, v))
                    if e not in used_edges:
                        valid.add(v)
                candidates &= valid
        
        # 2. Degree constraint:
        # The tree vertex best_v has degree deg_tree[best_v] in T_k.
        # Some of these edges are to placed neighbors. Let count_placed = len([nb for nb in adj[best_v] if nb in host_map])
        # The remaining edges (deg_tree[best_v] - count_placed) will be to unplaced neighbors.
        # These unplaced neighbors will also consume degree from the host vertex.
        # So the host vertex must have at least deg_tree[best_v] remaining capacity?
        # Not exactly, because the edges to unplaced neighbors will be assigned later.
        # But the host vertex itself must have enough "slots" left for all incident edges of this tree vertex.
        # The total degree of host vertex in the final packing is n-1.
        # Current used: host_degrees[hv].
        # Required: deg_tree[best_v].
        # So we need host_degrees[hv] + deg_tree[best_v] <= n - 1.
        
        tree_deg = [len(adj[v]) for v in range(k)]
        required_deg = tree_deg[best_v]
        
        filtered_candidates = []
        for hv in candidates:
            if host_degrees[hv] + required_deg <= n - 1:
                filtered_candidates.append(hv)
        
        candidates = filtered_candidates
        random.shuffle(candidates)
        
        # Also filter out vertices already used by this tree
        used_in_tree = set(host_map.values())
        candidates = [v for v in candidates if v not in used_in_tree]
        
        for hv in candidates:
            host_map[best_v] = hv
            # Update host_degrees temporarily?
            # We don't need to update host_degrees for the "used" edges yet, 
            # but we can increment it to reflect the "reservation" of this vertex's degree slots.
            # Actually, the degree constraint is a global one. 
            # If we reserve 'required_deg' slots for this vertex, we should subtract from available capacity.
            # Let's just pass a modified host_degrees or update in place and revert.
            
            old_deg = host_degrees[hv]
            host_degrees[hv] += required_deg
            
            if find_embedding_backtrack(t_idx, used_edges, host_map, adj, host_degrees):
                return True
            
            host_degrees[hv] = old_deg
            del host_map[best_v]
        
        return False
