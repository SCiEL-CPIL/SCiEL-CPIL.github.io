    def construct():
        _start = time.time()
        deadline = _start + 140.0
        
        # Start with Paley(31)
        n0 = 31
        adj = _paley_graph(n0)
        n = n0
        
        # Helper to find an independent set
        def _find_indep_set(adj, n):
            # Greedy: pick vertex with min degree, remove it and its neighbors
            # Or random
            # We want a set I such that alpha(G[V\I]) < 9
            # Let's try a few random independent sets
            rng = random.Random(int(time.time()*1000) % 10000)
            for _ in range(20):
                # Generate random independent set
                # Start with all vertices
                avail = (1 << n) - 1
                I = 0
                while avail:
                    # Pick random vertex from avail
                    # To make it faster, just pick lsb or random
                    # Let's pick random
                    # Get list of set bits? No, too slow.
                    # Pick random index 0..n-1, if in avail, add to I, remove from avail and neighbors
                    # This is O(n^2) per set. n is small (<41).
                    # Better: Greedy random
                    temp_avail = avail
                    temp_I = 0
                    while temp_avail:
                        # Pick random vertex
                        # Generate random number 0..n-1
                        # Check if in temp_avail
                        # If yes, add to I, remove neighbors
                        # If no, retry (or just pick lsb for speed, but random is better for variety)
                        # For speed, let's just pick lsb (deterministic greedy)
                        lsb = temp_avail & -temp_avail
                        v = lsb.bit_length() - 1
                        temp_I |= lsb
                        temp_avail &= ~lsb
                        temp_avail &= ~adj[v]
                    # Check condition
                    # Subgraph induced by V \ I
                    # V_mask = (1<<n)-1
                    # Comp_I = V_mask & ~temp_I
                    # Check alpha of subgraph induced by Comp_I
                    # We can run _max_independent_set_size_bitmask on the subgraph?
                    # _max_independent_set_size_bitmask expects adj list for 0..n-1.
                    # We need to mask it.
                    # Let's create a temporary adj list or modify the function to accept a mask?
                    # Easier: Just check if _max_independent_set_size_bitmask(adj, n) < 10 for the whole graph?
                    # No, we need alpha(G[V\I]) < 9.
                    
                    # Optimization: If alpha(G) < 9, any I works?
                    # No. alpha(G[V\I]) <= alpha(G).
                    # So if alpha(G) < 9, then alpha(G[V\I]) < 9 is always true.
                    # So if current alpha < 9, we can pick ANY independent set.
                    # If current alpha >= 9, we need to be careful.
                    
                    # Let's compute alpha(G) once per iteration.
                    pass
                
                # Simplified logic:
                # 1. Compute alpha(G).
                # 2. If alpha(G) < 9:
                #    Pick any independent set I (e.g., greedy max).
                #    Add vertex connected to I.
                # 3. Else:
                #    Try random independent sets I until alpha(G[V\I]) < 9.
                
                return temp_I # This is inside the loop, logic needs restructuring
            return 0 # Fallback

        # Restructured construct
        while n < 41 and time.time() < deadline:
            alpha_g = _max_independent_set_size_bitmask(adj, n)
            if alpha_g >= 10:
                break # Should not happen if we started valid and maintained condition
            
            # Find an independent set I
            # If alpha_g < 9, any independent set works for the condition alpha(G[V\I]) < 9
            # If alpha_g >= 9, we need to search.
            
            I = None
            if alpha_g < 9:
                # Greedy independent set
                avail = (1 << n) - 1
                I = 0
                while avail:
                    lsb = avail & -avail
                    v = lsb.bit_length() - 1
                    I |= lsb
                    avail &= ~lsb
                    avail &= ~adj[v]
            else:
                # Try random greedy sets
                rng = random.Random(n)
                for _ in range(100):
                    avail = (1 << n) - 1
                    I_temp = 0
                    while avail:
                        # Pick random vertex from avail
                        # Simple method: pick lsb, but shuffle order?
                        # Just pick lsb for now, maybe it's enough.
                        # To randomize, we can pick a random bit from avail.
                        # Get list of bits?
                        # For n<41, we can iterate.
                        bits = []
                        m = avail
                        while m:
                            lsb = m & -m
                            bits.append(lsb.bit_length() - 1)
                            m ^= lsb
                        if not bits: break
                        v = bits[rng.randint(0, len(bits)-1)]
                        mask_v = 1 << v
                        I_temp |= mask_v
                        avail &= ~mask_v
                        avail &= ~adj[v]
                    
                    # Check alpha of complement
                    # Create subgraph mask
                    # We need alpha of G[avail]
                    # We can reuse _max_independent_set_size_bitmask if we modify it to take a mask?
                    # Or just check if alpha_g < 10? No.
                    # Let's assume if alpha_g < 10, we are safe?
                    # Risk: alpha(G) = 9. alpha(G[V\I]) could be 9.
                    # Then 1 + 9 = 10. Fail.
                    # So we must check.
                    
                    # To check alpha(G[avail]), we can run B&B on the subgraph.
                    # This is expensive.
                    # However, if alpha_g < 9, we skip this.
                    # If alpha_g >= 9, we are in a tight spot.
                    # Given we start at 31 with alpha=6, it will take many steps to reach 9.
                    # So this branch might rarely be hit.
                    
                    # For now, let's just accept I_temp if we can't check easily, 
                    # but verify the final graph.
                    # Better: Verify the final graph after adding.
                    I = I_temp
                    break # Just take the first one, verify later

            if I is None:
                break

            # Add vertex n
            new_n = n + 1
            new_adj = adj + [0]
            new_adj[n] = I
            # Update neighbors
            # I is a bitmask of vertices in 0..n-1
            m = I
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                new_adj[v] |= (1 << n)
                m ^= lsb
            
            adj = new_adj
            n = new_n
            
            # Verify validity (Triangle-free and alpha < 10)
            if _has_triangle_bitmask(adj, n):
                break # Should not happen
            if _max_independent_set_size_bitmask(adj, n) >= 10:
                # Revert
                adj = adj[:n-1]
                n -= 1
                break

        # Convert to edges
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i] & (1 << j):
                    edges.append([i, j])
        
        return {'n': n, 'edges': edges}
    