<<<<<<< SEARCH
    def local_search(selected_indices, max_iter=100):
        """
        Local search: try to remove blocks or swap blocks to improve the solution.
        Returns an improved list of global block indices.
        """
        if not selected_indices:
            return []
        
        current = list(selected_indices)
        current_set = set(current)
        n_sel = len(current)
        
        # Compute coverage counts
        total_count = np.zeros(N_t, dtype=np.int32)
        for g in current:
            for si in block_tsubs[g]:
                total_count[si] += 1
        
        # Try removals first
        improved = True
        iter_count = 0
        while improved and iter_count < max_iter and time.time() < DEADLINE:
            improved = False
            iter_count += 1
            for i in range(len(current)):
                g = current[i]
                # Check if removing g keeps the cover valid
                removable = True
                for si in block_tsubs[g]:
                    if total_count[si] == 1:
                        removable = False
                        break
                if removable:
                    # Remove g
                    current.pop(i)
                    current_set.discard(g)
                    for si in block_tsubs[g]:
                        total_count[si] -= 1
                    improved = True
                    break  # Restart loop after removal
        
        # Try swaps: replace one block with another that covers more uncovered t-subsets
        # For simplicity, try random swaps
        random.shuffle(current)
        for i in range(len(current)):
            if time.time() >= DEADLINE:
                break
            g_old = current[i]
            # Find blocks that could replace g_old
            # A good replacement covers t-subsets that g_old covers, plus maybe more
            # For now, try random blocks from a candidate set
            # This is expensive, so limit candidates
            candidates = []
            # Sample some random blocks
            for _ in range(50):
                g_new = random.randint(0, N_blocks - 1)
                if g_new not in current_set:
                    candidates.append(g_new)
            
            for g_new in candidates:
                # Check if replacing g_old with g_new keeps the cover valid
                # Remove g_old, add g_new, check coverage
                temp_count = total_count.copy()
                for si in block_tsubs[g_old]:
                    temp_count[si] -= 1
                for si in block_tsubs[g_new]:
                    temp_count[si] += 1
                
                if np.all(temp_count >= 1):
                    # Valid replacement
                    current[i] = g_new
                    current_set.discard(g_old)
                    current_set.add(g_new)
                    total_count = temp_count
                    improved = True
                    break
            
            if improved:
                break  # Restart after swap
        
        return current
=======
    def local_search(selected_indices, max_iter=100):
        """
        Local search: try to remove blocks or swap blocks to improve the solution.
        Returns an improved list of global block indices.
        """
        if not selected_indices:
            return []
        
        current = list(selected_indices)
        current_set = set(current)
        n_sel = len(current)
        
        # Compute coverage counts
        total_count = np.zeros(N_t, dtype=np.int32)
        for g in current:
            for si in block_tsubs[g]:
                total_count[si] += 1
        
        # Try removals first
        improved = True
        iter_count = 0
        while improved and iter_count < max_iter and time.time() < DEADLINE:
            improved = False
            iter_count += 1
            for i in range(len(current)):
                g = current[i]
                # Check if removing g keeps the cover valid
                removable = True
                for si in block_tsubs[g]:
                    if total_count[si] == 1:
                        removable = False
                        break
                if removable:
                    # Remove g
                    current.pop(i)
                    current_set.discard(g)
                    for si in block_tsubs[g]:
                        total_count[si] -= 1
                    improved = True
                    break  # Restart loop after removal
        
        # Try swaps: replace one block with another that covers more uncovered t-subsets
        random.shuffle(current)
        for i in range(len(current)):
            if time.time() >= DEADLINE:
                break
            g_old = current[i]
            # Find blocks that could replace g_old
            candidates = []
            for _ in range(50):
                g_new = random.randint(0, N_blocks - 1)
                if g_new not in current_set:
                    candidates.append(g_new)
            
            for g_new in candidates:
                # Check if replacing g_old with g_new keeps the cover valid
                temp_count = total_count.copy()
                for si in block_tsubs[g_old]:
                    temp_count[si] -= 1
                for si in block_tsubs[g_new]:
                    temp_count[si] += 1
                
                if np.all(temp_count >= 1):
                    # Valid replacement
                    current[i] = g_new
                    current_set.discard(g_old)
                    current_set.add(g_new)
                    total_count = temp_count
                    improved = True
                    break
            
            if improved:
                break  # Restart after swap
        
        # 2-for-1 exchange: try replacing 2 blocks with 1 block
        if time.time() < DEADLINE:
            # Recompute total_count after swaps
            total_count = np.zeros(N_t, dtype=np.int32)
            for g in current:
                for si in block_tsubs[g]:
                    total_count[si] += 1
            
            n_cur = len(current)
            if n_cur >= 2:
                # For each pair, check if a single block can cover all critical t-subsets
                # A pair (i,j) is "2-for-1 removable" if there exists a block b
                # such that removing i and j and adding b keeps the cover valid.
                # Equivalently: all t-subsets that become uncovered when removing i and j
                # must be covered by b.
                
                # Precompute: for each block, the set of t-subsets it covers
                # We need: for pair (i,j), find t-subsets with count==1 after removing both,
                # or count==2 where both i and j are the only two covering it.
                
                # Strategy: sample pairs and check
                max_pair_checks = 200
                checked = 0
                random.shuffle(current)
                for i in range(n_cur):
                    if time.time() >= DEADLINE or checked >= max_pair_checks:
                        break
                    for j in range(i + 1, n_cur):
                        if time.time() >= DEADLINE or checked >= max_pair_checks:
                            break
                        checked += 1
                        
                        g_i = current[i]
                        g_j = current[j]
                        
                        # Find t-subsets that would become uncovered
                        # i.e., t-subsets where total_count == 2 and both are covered by g_i and g_j,
                        # or total_count == 1 and covered by one of them (but that's already handled by removal)
                        # Actually: after removing g_i and g_j, a t-subset si is uncovered if
                        # total_count[si] <= 2 and (g_i covers si or g_j covers si) and
                        # total_count[si] - (1 if g_i covers si else 0) - (1 if g_j covers si else 0) == 0
                        
                        # Collect critical t-subsets
                        critical = set()
                        for si in block_tsubs[g_i]:
                            new_count = total_count[si] - 1
                            if new_count == 0:
                                # Only g_i covers si -> can't remove g_i alone, but with g_j?
                                # If g_j also covers si, new_count would be total_count[si]-2
                                if g_j in set(block_tsubs[si]):  # This is O(k choose t) per check, too slow
                                    pass
                                # Simpler: just check all t-subsets in g_i
                                critical.add(si)
                        for si in block_tsubs[g_j]:
                            if si not in critical:
                                # Check if g_i also covers si
                                # We need to know if removing both makes it uncovered
                                pass
                            # Actually, let's just compute properly
                        # Recompute: for all si in union of tsubs of g_i and g_j
                        # Check if count after removing both is 0
                        all_critical = set()
                        for si in block_tsubs[g_i]:
                            cnt = total_count[si]
                            # g_i covers si, so at least 1 removed
                            # g_j covers si? Need to check
                            # Precompute: is g_j in tsub_blocks[si]?
                            # tsub_blocks[si] is a list of block indices
                            # Check if g_j is in that list
                            # This is O(len(tsub_blocks[si])) which could be large
                            # Alternative: just use a set for g_j's tsubs
                            pass
                        
                        # Let's use a different approach: build set of g_j's tsubs
                        g_j_tsubs_set = set(block_tsubs[g_j])
                        for si in block_tsubs[g_i]:
                            cnt = total_count[si]
                            if si in g_j_tsubs_set:
                                new_cnt = cnt - 2
                            else:
                                new_cnt = cnt - 1
                            if new_cnt <= 0:
                                all_critical.add(si)
                        for si in block_tsubs[g_j]:
                            if si not in g_j_tsubs_set:  # This is always false, fix logic
                                pass
                            # si is in g_j_tsubs_set by construction
                            # Check if g_i also covers si
                            # We already handled g_i's tsubs above
                            # For si in g_j but not in g_i:
                            if si not in set(block_tsubs[g_i]):
                                cnt = total_count[si]
                                new_cnt = cnt - 1
                                if new_cnt <= 0:
                                    all_critical.add(si)
                        
                        if not all_critical:
                            continue
                        
                        # Now find a block b not in current that covers all_critical
                        # and doesn't add any new requirement (i.e., all_critical ⊆ tsubs(b))
                        # Since all_critical could be large, limit search
                        if len(all_critical) > 30:
                            continue
                        
                        # Sample random blocks to find one covering all_critical
                        found = False
                        for _ in range(20):
                            b = random.randint(0, N_blocks - 1)
                            if b in current_set:
                                continue
                            b_tsubs = set(block_tsubs[b])
                            if all_critical.issubset(b_tsubs):
                                # Valid 2-for-1 exchange
                                # Remove i and j, add b
                                # Remove j first (higher index)
                                current.pop(j)
                                current.pop(i)
                                current.append(b)
                                current_set.discard(g_i)
                                current_set.discard(g_j)
                                current_set.add(b)
                                
                                # Update total_count
                                for si in block_tsubs[g_i]:
                                    total_count[si] -= 1
                                for si in block_tsubs[g_j]:
                                    total_count[si] -= 1
                                for si in block_tsubs[b]:
                                    total_count[si] += 1
                                
                                found = True
                                break
                        
                        if found:
                            break  # Restart pair search after a successful exchange
        
        return current
>>>>>>> REPLACE

