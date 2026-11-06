    def solve_mis(elements):
        n = len(elements)
        if n == 0:
            return []
        
        idx_map = {val: i for i, val in enumerate(elements)}
        
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if elements[i] + elements[j] in idx_map:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        comp_adj = [0] * n
        for i in range(n):
            all_mask = (1 << n) - 1
            all_mask ^= (1 << i)
            comp_adj[i] = all_mask & ~adj[i]
            
        odd_set = [i for i in range(n) if elements[i] % 2 != 0]
        best_size = len(odd_set)
        best_clique = odd_set[:]
        
        def popcount(x):
            return bin(x).count('1')
            
        def expand(P, X, current_clique):
            nonlocal best_size, best_clique
            if P == 0:
                if len(current_clique) > best_size:
                    best_size = len(current_clique)
                    best_clique = current_clique[:]
                return
            
            # Greedy coloring bound
            # Extract vertices
            verts = []
            temp_P = P
            while temp_P:
                lsb = temp_P & -temp_P
                idx = lsb.bit_length() - 1
                verts.append(idx)
                temp_P ^= lsb
                
            # If even the max possible size (len(X) + len(verts)) is not better, return
            if len(current_clique) + len(verts) <= best_size:
                return
                
            # Coloring
            col = [0] * n
            max_color = 0
            # Order by degree in P? 
            # Let's just use the order in verts for now.
            for v in verts:
                neighbors = comp_adj[v] & P
                used = 0
                temp_neigh = neighbors
                while temp_neigh:
                    lsb = temp_neigh & -temp_neigh
                    u = lsb.bit_length() - 1
                    if col[u] != 0:
                        used |= (1 << col[u])
                    temp_neigh ^= lsb
                    
                c = 1
                while (used >> c) & 1:
                    c += 1
                col[v] = c
                if c > max_color:
                    max_color = c
                    
            if len(current_clique) + max_color <= best_size:
                return
            
            # Branch
            # Process vertices in reverse order of color?
            # Tomita: process in order of decreasing color.
            # We need to map vertices to their colors.
            # Let's create a list of (color, vertex) and sort by color descending.
            # This helps finding large cliques faster.
            
            verts_with_color = [(col[v], v) for v in verts]
            verts_with_color.sort(key=lambda x: x[0], reverse=True)
            
            for c, v in verts_with_color:
                # If len(X) + c <= best_size, we can break?
                # No, c is the color of v. The max clique in the remaining P 
                # is at most the max color remaining.
                # Since we are processing in descending color, the remaining colors are <= c.
                # So if len(current_clique) + c <= best_size, we can break.
                if len(current_clique) + c <= best_size:
                    break
                
                # Include v
                new_P = P & comp_adj[v]
                new_X = X | (1 << v)
                new_clique = current_clique + [v]
                expand(new_P, new_X, new_clique)
                
                # Exclude v
                P ^= (1 << v)
                
                # Prune
                if len(current_clique) + popcount(P) <= best_size:
                    break

        expand((1 << n) - 1, 0, [])
        
        return [elements[i] for i in best_clique]
