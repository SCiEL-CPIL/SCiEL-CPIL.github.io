    def max_clique(R, P):
        nonlocal best_clique_size, best_clique_nodes
        
        if time.time() > deadline:
            return
            
        if P == 0:
            if len(R) > best_clique_size:
                best_clique_size = len(R)
                best_clique_nodes = R.copy()
            return
            
        # Greedy Coloring of P
        # We want to assign colors to nodes in P such that adjacent nodes have different colors.
        # The number of colors is an upper bound on the clique size in P.
        
        # We will store the color of each node in an array `color`
        # And we will create a list of nodes sorted by color descending.
        
        # To speed up, we only color the nodes in P.
        
        # Simple greedy:
        # 1. Get list of nodes in P.
        # 2. Sort them by degree in P (optional, but helps).
        # 3. Assign colors.
        
        nodes_in_P = []
        temp_P = P
        while temp_P:
            lsb = temp_P & -temp_P
            v = lsb.bit_length() - 1
            nodes_in_P.append(v)
            temp_P ^= lsb
            
        # Sort by degree in P (descending) for better coloring
        # Degree in P: (adj[v] & P).bit_count()
        nodes_in_P.sort(key=lambda v: (adj[v] & P).bit_count(), reverse=True)
        
        color = [-1] * N_S
        max_color = 0
        for v in nodes_in_P:
            # Find smallest color not used by neighbors in P
            used_colors = set()
            # Neighbors of v in P
            neighbors_in_P = adj[v] & P
            temp_nei = neighbors_in_P
            while temp_nei:
                lsb = temp_nei & -temp_nei
                u = lsb.bit_length() - 1
                if color[u] != -1:
                    used_colors.add(color[u])
                temp_nei ^= lsb
                
            c = 0
            while c in used_colors:
                c += 1
            color[v] = c
            if c > max_color:
                max_color = c
                
        if len(R) + max_color <= best_clique_size:
            return
            
        # Iterate through colors from max_color down to 0
        # For each color c, process nodes with that color
        # This is the Tomita heuristic
        
        # We need to iterate nodes in reverse order of color
        # Let's group nodes by color
        # But we also need to remove nodes from P as we go
        
        # Standard Tomita loop:
        # for c in range(max_color, -1, -1):
        #   for v in nodes with color c:
        #     if len(R) + c <= best: break
        #     ...
        
        # Implementation:
        # We iterate through the sorted nodes_in_P (which are sorted by degree, not color).
        # We need to process them in order of decreasing color.
        
        # Let's create a list of (color, node)
        colored_nodes = [(color[v], v) for v in nodes_in_P]
        colored_nodes.sort(key=lambda x: x[0], reverse=True)
        
        # Now iterate
        temp_P = P
        for c, v in colored_nodes:
            if time.time() > deadline:
                return
            
            # v must still be in temp_P
            if not (temp_P & (1 << v)):
                continue
                
            # Pruning
            if len(R) + c <= best_clique_size:
                break # Since colors are decreasing, all subsequent nodes have <= c
                
            # Branch on v
            new_R = R + [v]
            new_P = temp_P & adj[v]
            
            max_clique(new_R, new_P)
            
            # Remove v from temp_P
            temp_P &= ~(1 << v)
            
