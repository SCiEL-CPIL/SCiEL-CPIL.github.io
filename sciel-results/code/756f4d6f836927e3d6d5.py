    def max_independent_set_size(n, adj):
        """
        Fast B&B for MIS size using bitsets and greedy coloring bound.
        """
        adj_mask = [0] * n
        for u in range(n):
            for v in adj[u]:
                adj_mask[u] |= (1 << v)
        
        # We search for max clique in the complement graph?
        # No, let's stick to MIS on G.
        # An independent set in G is a clique in G_bar.
        # It is often faster to find Max Clique in a dense graph.
        # Is G_bar dense?
        # If G is triangle-free, it can be sparse or dense.
        # Maximal triangle-free graphs are often quite dense.
        # So G_bar is sparse.
        # Finding Max Clique in a sparse graph is hard.
        # Finding MIS in a dense graph is hard.
        # So both might be hard.
        
        # Let's use the standard "Coloring" bound for Max Clique on G_bar.
        # We need the adjacency masks for G_bar.
        # G_bar edge (u,v) exists if NOT (u,v) in G.
        
        comp_adj_mask = [0] * n
        for u in range(n):
            # All vertices except u
            all_mask = (1 << n) - 1
            # Remove u
            all_mask &= ~(1 << u)
            # Remove neighbors of u in G
            all_mask &= ~adj_mask[u]
            comp_adj_mask[u] = all_mask

        best_clique_size = 0
        
        # Order vertices by degree in comp graph?
        # Let's just use index order.
        
        def expand_clique(available, current_clique_size):
            nonlocal best_clique_size
            if current_clique_size + available.bit_count() <= best_clique_size:
                return
            
            if available == 0:
                if current_clique_size > best_clique_size:
                    best_clique_size = current_clique_size
                return
            
            # Greedy coloring of the available subgraph in the COMP graph
            # We want to color the vertices in 'available' such that adjacent vertices (in comp graph) have different colors.
            # The number of colors is an upper bound on the clique size in the comp graph.
            
            # We will perform a greedy coloring.
            # We maintain a list of color classes.
            # We iterate through vertices in 'available' (in some order) and assign the smallest possible color.
            
            # To make it fast, we can just count colors.
            # Standard Tomita algorithm uses a specific ordering.
            
            # Let's implement a simple greedy coloring count.
            # We'll process vertices in 'available' one by one.
            # For each vertex v, we check which colors are used by its neighbors in 'available'.
            
            # To be efficient, we can pre-compute nothing, just iterate.
            # n=40, so this is cheap.
            
            colors = [0] * n  # color of each vertex, 0 means uncolored
            color_count = 0
            # We need to process vertices in a specific order.
            # Let's pick vertices from 'available' in increasing index order.
            # This might not be optimal but is simple.
            
            # Actually, a better bound is obtained by ordering vertices by degree in the available subgraph.
            # But for n=40, simple order is likely fine.
            
            # Let's try to color 'available' vertices.
            # We'll use a list of uncolored vertices from 'available'.
            # We'll just iterate through bits in 'available'.
            
            # Efficient way:
            # We maintain a set of uncolored vertices (bitmask).
            uncolored = available
            while uncolored:
                color_count += 1
                # Find a vertex in uncolored
                # We can pick the first one
                v_idx = (uncolored & -uncolored).bit_length() - 1
                v_mask = 1 << v_idx
                uncolored &= ~v_mask
                colors[v_idx] = color_count
                
                # Now, we can color other vertices that are NOT adjacent to v_idx in the comp graph
                # i.e., they are neighbors in G? No.
                # In comp graph, u and v are adjacent if NOT edge in G.
                # So if we assign color c to v, we cannot assign color c to any u in uncolored
                # such that u and v are adjacent in comp graph.
                # u and v are adjacent in comp graph if NOT (u in adj[v] in G).
                # So we CAN assign color c to u if u IS in adj[v] in G (plus u=v).
                # Wait.
                # Comp edge (u,v) exists if G edge (u,v) does NOT exist.
                # So if G edge (u,v) exists, they are NOT adjacent in comp, so they CAN have same color.
                # If G edge (u,v) does NOT exist, they ARE adjacent in comp, so they CANNOT have same color.
                
                # So, for a given color class, we can add vertex u to the class of v
                # if u is in adj_G[v] (so no comp edge) OR u == v.
                # But we also need u to be in 'uncolored'.
                
                # Let's define the set of vertices compatible with color c (assigned to v)
                # Compatible if (u in adj_G[v]) OR (u == v).
                # Note: v is already removed from uncolored.
                
                # So compatible_mask = adj_mask[v] | (1 << v_idx)
                # But we only care about vertices in 'uncolored'.
                
                compatible = uncolored & (adj_mask[v_idx] | v_mask)
                
                # We can greedily add all vertices in 'compatible' to this color class?
                # No, they might be adjacent to each other in the comp graph.
                # If u1, u2 are in compatible, are they adjacent in comp?
                # u1 in adj_G[v] and u2 in adj_G[v].
                # Are u1, u2 adjacent in comp?
                # They are adjacent in comp if NOT edge in G.
                # So if edge (u1, u2) is NOT in G, they are adjacent in comp and cannot share color.
                # So we can't just take all of 'compatible'. We need to find a maximal independent set in the comp graph induced by 'compatible'?
                # That's too complex for a bound.
                
                # Standard Tomita algorithm picks one vertex, colors it, and removes it, then repeats for the next color.
                # It doesn't try to fill the color class maximally in one go, it just assigns colors to vertices one by one.
                
                # Let's stick to the simple "assign color to one vertex" approach.
                # This gives a valid coloring, hence a valid upper bound.
                # It might be a loose bound, but it's fast.
                
                # Actually, the standard efficient coloring for B&B colors the *entire* available set.
                # Let's implement the simple loop:
                # While uncolored:
                #   color_count++
                #   pick v from uncolored
                #   uncolored remove v
                #   (Optional: remove from uncolored any vertices that are incompatible with v for this color? No, that's for the color class itself)
                #   Actually, the simple approach is:
                #   For each color, pick a vertex, remove it from uncolored.
                #   This is just counting the number of vertices? No, that's the trivial bound.
                #   The coloring bound works because if we color the graph with k colors, any clique has size <= k.
                #   To get a tight k, we should assign the same color to as many non-adjacent (in comp) vertices as possible.
                
                # Let's try to fill the color class greedily.
                # Current color class candidates: uncolored.
                # We picked v.
                # Any u in uncolored that is NOT adjacent to v in comp (i.e., u in adj_G[v]) can potentially share this color.
                # But u1, u2 in that set might be adjacent to each other in comp.
                # So we need to pick an independent set in the comp graph induced by (uncolored & adj_G[v]).
                # This is recursive.
                
                # For n=40, let's just use the simple "one vertex per color" bound?
                # That bound is just |available|, which is weak.
                
                # Let's use a slightly better heuristic:
                # We will color the vertices in 'available' one by one.
                # We maintain a list of colors used by neighbors.
                
                # Let's implement the standard Tomita coloring step properly.
                # It's a bit of code but worth it.
                
                # Refining the coloring logic:
                # We want to color the subgraph induced by 'available' in the COMP graph.
                # We process vertices in 'available' in a specific order (e.g., degree order).
                # For each vertex v, we assign the smallest color c such that no neighbor of v in the comp graph (among already colored vertices) has color c.
                
                # Let's do that.
                # We need to know the neighbors in the comp graph for each vertex in 'available'.
                # comp_neighbors(v) = available & ~adj_mask[v] & ~(1<<v)
                
                # We'll iterate through vertices in 'available'.
                # To make it deterministic and fast, we'll just iterate 0..n-1.
                
                color_map = {} # vertex -> color
                next_color = 1
                # We need to process vertices in 'available'.
                # Let's get the list of vertices in 'available'
                # avail_list = [i for i in range(n) if (available >> i) & 1]
                # For each v in avail_list:
                #    used_colors = set()
                #    for u in avail_list:
                #       if u < v and (comp_adj_mask[v] >> u) & 1: # u is neighbor of v in comp
                #          if u in color_map:
                #             used_colors.add(color_map[u])
                #    # find smallest color not in used_colors
                #    while next_color in used_colors:
                #       next_color += 1 # This logic is wrong, next_color is global
                #    # Actually, we just need to find a color for v.
                #    # Let's use a list of used colors for v.
                
                # This O(N^2) coloring is fine for N=40.
                
                # Let's implement it.
                
                # Order: just 0..n-1
                # We need to check neighbors in 'available' that are already colored.
                
                # Let's store colors in an array `col` of size n, initialized to 0.
                col = [0] * n
                max_color = 0
                for v in range(n):
                    if not ((available >> v) & 1):
                        continue
                    # Find colors used by neighbors of v in comp graph that are in 'available' and already colored
                    # Neighbor u in comp graph if NOT (u in adj_G[v])
                    # And u in available.
                    # And u is already colored (u < v in our loop? No, we just need to check all u in available with col[u] > 0)
                    # Actually, we can just check all u in available.
                    used = set()
                    for u in range(n):
                        if u == v: continue
                        if not ((available >> u) & 1): continue
                        if col[u] == 0: continue
                        # Check if u and v are adjacent in comp graph
                        # Adjacent in comp if NOT (u in adj_G[v])
                        # adj_mask[v] has bit u set if u in adj_G[v]
                        if not ((adj_mask[v] >> u) & 1):
                            used.add(col[u])
                    
                    # Assign smallest color not in used
                    c = 1
                    while c in used:
                        c += 1
                    col[v] = c
                    if c > max_color:
                        max_color = c
                
                # Upper bound is current_clique_size + max_color
                if current_clique_size + max_color <= best_clique_size:
                    return

            # Now branch
            # Pick a vertex v from 'available'
            # Heuristic: pick vertex with highest degree in comp graph within 'available'
            # Or just first one.
            # Let's pick the one with highest degree in comp graph to prune better.
            best_v = -1
            best_deg = -1
            for v in range(n):
                if ((available >> v) & 1):
                    deg = (available & comp_adj_mask[v]).bit_count()
                    if deg > best_deg:
                        best_deg = deg
                        best_v = v
            
            if best_v == -1:
                return
                
            v_mask = 1 << best_v
            
            # Branch 1: Include best_v in clique
            # New available: available & comp_adj_mask[best_v]
            # (Must be adjacent to best_v in comp graph)
            expand_clique(available & comp_adj_mask[best_v], current_clique_size + 1)
            
            # Branch 2: Exclude best_v
            expand_clique(available & ~v_mask, current_clique_size)

        expand_clique((1 << n) - 1, 0)
        return best_clique_size
