<<<<<<< SEARCH
        # --- Phase 2: Local Search Refinement ---
        sel_indices = np.where(selected == 1)[0]
        if len(sel_indices) == 0:
            continue
            
        improved = True
        max_moves = 100
        move_count = 0
        
        while improved and move_count < max_moves and time.time() < deadline - 5.0:
            improved = False
            move_count += 1
            
            # Shuffle selected indices to avoid bias
            order = rng.permutation(len(sel_indices))
            
            for i in order:
                if time.time() > deadline - 5.0:
                    break
                    
                v = sel_indices[i]
                
                # Compute mask of vertices compatible with (S \ {v})
                mask_except_v = np.ones(n, dtype=np.uint8)
                for s in sel_indices:
                    if s != v:
                        mask_except_v = mask_except_v & compat[s]
                
                mask_except_v[sel_indices] = 0 # Exclude all selected
                
                cand_indices = np.where(mask_except_v > 0)[0]
                
                if len(cand_indices) < 2:
                    continue
                    
                # Try to find a compatible pair (u, w) in cand_indices
                # This is finding an edge in the compat graph induced by cand_indices.
                # Or, finding a non-edge in the conflict graph.
                
                # To be faster, we can just check if the induced subgraph has any 0 in conflict.
                # We don't need to find the specific pair immediately if we just want to know if a move is possible.
                # But we need to pick the pair.
                
                # Let's try to pick the pair that is "best" (e.g., min degree in remaining graph).
                # For now, just find any pair.
                
                # Get conflict submatrix
                c_sub = conflict[np.ix_(cand_indices, cand_indices)]
                
                # Find first compatible pair
                # Use np.argwhere? Slow.
                # Use:
                # mask = c_sub == 0
                # np.fill_diagonal(mask, False)
                # if mask.any():
                #    u_idx, w_idx = np.argwhere(mask)[0]
                
                # Faster:
                # For each u, check if it has any compatible w.
                # This is O(|C|^2).
                
                # Let's try a heuristic: pick a random u, check if it has compatible w.
                # Repeat a few times.
                
                found_pair = False
                # Try a few random candidates
                for _ in range(10):
                    if len(cand_indices) == 0:
                        break
                    u = cand_indices[rng.integers(len(cand_indices))]
                    # Find w compatible with u
                    compat_w = compat[u, cand_indices]
                    compat_w[cand_indices == u] = 0 # Exclude self
                    
                    if compat_w.any():
                        w = cand_indices[np.argmax(compat_w)]
                        # Perform move: Remove v, add u, w
                        selected[v] = 0
                        selected[u] = 1
                        selected[w] = 1
                        
                        # Update sel_indices
                        sel_indices = np.where(selected == 1)[0]
                        
                        improved = True
                        found_pair = True
                        break
                        
                if found_pair:
                    break # Restart loop with new state
                
                # Try 1-for-3 move if 1-for-2 failed
                if not improved and len(cand_indices) >= 3:
                    # Find a compatible triple
                    # This is harder.
                    # Pick u, then find w compatible with u, then find x compatible with u and w.
                    
                    found_triple = False
                    for _ in range(5):
                        if len(cand_indices) == 0:
                            break
                        u = cand_indices[rng.integers(len(cand_indices))]
                        compat_u = compat[u, cand_indices]
                        compat_u[cand_indices == u] = 0
                        
                        if compat_u.sum() < 2:
                            continue
                        
                        # Pick w
                        w = cand_indices[np.argmax(compat_u)]
                        
                        # Find x compatible with both u and w
                        compat_ux = compat_u & compat[w, cand_indices]
                        compat_ux[cand_indices == u] = 0
                        compat_ux[cand_indices == w] = 0
                        
                        if compat_ux.any():
                            x = cand_indices[np.argmax(compat_ux)]
                            
                            # Perform move
                            selected[v] = 0
                            selected[u] = 1
                            selected[w] = 1
                            selected[x] = 1
                            
                            sel_indices = np.where(selected == 1)[0]
                            improved = True
                            found_triple = True
                            break
                            
                    if found_triple:
                        break
=======
        # --- Phase 2: Local Search Refinement (Vectorized) ---
        sel_indices = np.where(selected == 1)[0]
        if len(sel_indices) == 0:
            continue
            
        improved = True
        max_moves = 200
        move_count = 0
        
        while improved and move_count < max_moves and time.time() < deadline - 5.0:
            improved = False
            move_count += 1
            
            sel_indices = np.where(selected == 1)[0]
            k_sel = len(sel_indices)
            
            if k_sel == 0:
                break
            
            # Vectorized 1-for-2: For each v in S, find all (u,w) in C_v that are compatible
            # C_v = {x : x compatible with all s in S\{v}, x not in S}
            # We compute mask_except_v for each v efficiently.
            
            # First, compute the "intersection mask" for all of S:
            # all_compat[i] = 1 if i is compatible with ALL selected vertices
            # Then for v, mask_except_v = all_compat & ~selected[v] & ~selected (but we handle exclusion)
            
            # Actually, let's compute per-v masks but vectorize the pair-finding.
            # For each v, we need:
            #   M_v = (AND of compat[s] for s in S, s!=v) & ~selected
            #   Then find pairs (u,w) in M_v such that compat[u,w]=1
            
            # Precompute the full intersection
            full_mask = np.ones(n, dtype=np.uint8)
            for s in sel_indices:
                full_mask = full_mask & compat[s]
            full_mask[sel_indices] = 0
            
            # For each v, M_v = full_mask | compat[v] (since we exclude v from the AND)
            # But we need to be careful: M_v should include vertices compatible with S\{v}
            # full_mask excludes v itself (since selected[v]=1, and we set full_mask[sel_indices]=0)
            # So M_v = (full_mask | compat[v]) & ~selected & ~compat[v's own exclusion]
            # Actually: M_v = vertices compatible with all s in S\{v} and not in S
            # = (full_mask | compat[v]) & ~selected  (since full_mask already has selected=0)
            # But wait, full_mask has selected=0, so full_mask | compat[v] might re-include other selected.
            # Let me just compute: M_v = (full_mask | compat[v]) & ~selected
            
            # For each v, try to find a compatible pair in M_v
            # Vectorize: for each v, check if there exists u,w in M_v with compat[u,w]=1
            
            # Efficient approach: For each v, get M_v, then check if the induced subgraph has an edge.
            # We'll do this semi-vectorized: iterate over v, but vectorize the pair search.
            
            for v in sel_indices:
                if time.time() > deadline - 5.0:
                    break
                    
                M_v = (full_mask | compat[v]) & ~selected
                M_v[v] = 0  # Exclude v itself
                
                cand = np.where(M_v > 0)[0]
                if len(cand) < 2:
                    continue
                
                # Find a compatible pair in cand
                # Check if any pair (u,w) in cand has compat[u,w]=1
                # Vectorized: for each u in cand, check if it has any compatible w in cand
                compat_sub = compat[np.ix_(cand, cand)]
                np.fill_diagonal(compat_sub, 0)
                
                if compat_sub.any():
                    # Find a specific pair
                    u_idx, w_idx = np.where(compat_sub)[0][:2]
                    u, w = cand[u_idx], cand[w_idx]
                    
                    # Perform move: remove v, add u, w
                    selected[v] = 0
                    selected[u] = 1
                    selected[w] = 1
                    improved = True
                    break  # Restart the while loop with new state
                    
            # If no 1-for-2 move found, try 1-for-1 (swap)
            if not improved:
                for v in sel_indices:
                    if time.time() > deadline - 5.0:
                        break
                    
                    M_v = (full_mask | compat[v]) & ~selected
                    M_v[v] = 0
                    
                    cand = np.where(M_v > 0)[0]
                    if len(cand) >= 1:
                        # Try swapping v with a random candidate
                        u = cand[rng.integers(len(cand))]
                        selected[v] = 0
                        selected[u] = 1
                        improved = True
                        break  # Restart with new state
                        
            # If still no improvement, try 1-for-3
            if not improved:
                for v in sel_indices:
                    if time.time() > deadline - 5.0:
                        break
                        
                    M_v = (full_mask | compat[v]) & ~selected
                    M_v[v] = 0
                    
                    cand = np.where(M_v > 0)[0]
                    if len(cand) < 3:
                        continue
                    
                    # Try to find a compatible triple in cand
                    # Pick u, then w compatible with u, then x compatible with both
                    for _ in range(5):
                        u = cand[rng.integers(len(cand))]
                        compat_u = compat[u, cand]
                        compat_u[cand == u] = 0
                        
                        if compat_u.sum() < 2:
                            continue
                        
                        w = cand[np.argmax(compat_u)]
                        
                        compat_ux = compat_u & compat[w, cand]
                        compat_ux[cand == u] = 0
                        compat_ux[cand == w] = 0
                        
                        if compat_ux.any():
                            x = cand[np.argmax(compat_ux)]
                            
                            selected[v] = 0
                            selected[u] = 1
                            selected[w] = 1
                            selected[x] = 1
                            improved = True
                            break
                            
                    if improved:
                        break
>>>>>>> REPLACE

