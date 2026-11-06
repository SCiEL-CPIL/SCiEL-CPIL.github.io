    def expand(P, Q, depth):
        nonlocal best_size, best_clique
        
        if time.time() > deadline:
            return
            
        if not P:
            if depth > best_size:
                best_size = depth
                best_clique = Q[:]
            return
            
        # Extract vertices
        verts = []
        temp_P = P
        while temp_P:
            lsb = temp_P & -temp_P
            v = lsb.bit_length() - 1
            verts.append(v)
            temp_P ^= lsb
            
        # Greedy Coloring
        color_of = [0] * M
        
        # Sort vertices by degree in P descending to get better coloring?
        # Let's try sorting by index for now, or maybe degree.
        # Degree calculation:
        # deg[v] = (comp_adj[v] & P).bit_count()
        
        # To save time, let's just use the order from extraction.
        
        for v in verts:
            used_colors = set()
            neighbors_mask = comp_adj[v] & P
            while neighbors_mask:
                lsb = neighbors_mask & -neighbors_mask
                u = lsb.bit_length() - 1
                if color_of[u] != 0:
                    used_colors.add(color_of[u])
                neighbors_mask ^= lsb
                
            c = 1
            while c in used_colors:
                c += 1
            color_of[v] = c
            
        # Sort by color descending
        verts_sorted = sorted(verts, key=lambda x: color_of[x], reverse=True)
        
        for v in verts_sorted:
            # Check if v is still in P
            if not (P & (1 << v)):
                continue
                
            c = color_of[v]
            if depth + c <= best_size:
                break
                
            # Include v
            new_P = P & comp_adj[v]
            new_Q = Q + [v]
            expand(new_P, new_Q, depth + 1)
            
            # Remove v from P
            P &= ~(1 << v)
