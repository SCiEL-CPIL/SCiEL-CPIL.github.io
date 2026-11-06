    def dfs(S_mask, C_mask, depth, best_size):
        if C_mask == 0:
            if depth < best_size[0]:
                best_size[0] = depth
            return
        
        # Heuristic: pick p in C_mask with fewest blockers
        # Or just pick the first one?
        # Let's pick the one with smallest number of blockers to force a decision.
        # Actually, if we pick a p with 0 blockers, we MUST add it.
        # If we pick a p with many blockers, we have many choices.
        # So pick p with MINIMUM blockers.
        
        min_blockers = None
        min_p = -1
        tmp_C = C_mask
        while tmp_C:
            lsb = tmp_C & -tmp_C
            p = lsb.bit_length() - 1
            tmp_C ^= lsb
            
            # Calculate blockers for p
            # Blockers are q in C_mask such that q is in iso[p][a] for some a in S_mask
            blockers = 0
            tmp_S = S_mask
            while tmp_S:
                lsb_s = tmp_S & -tmp_S
                a = lsb_s.bit_length() - 1
                tmp_S ^= lsb_s
                blockers |= iso[p][a]
            
            blockers &= C_mask # only candidates
            
            if min_blockers is None or popcount(blockers) < min_blockers:
                min_blockers = popcount(blockers)
                min_p = p
                if min_blockers == 0:
                    break # Can't get better than 0
        
        # Now we have min_p
        # Calculate blockers for min_p again (or store them)
        # ...
        
        # Option 1: Add min_p
        # Option 2: Add a blocker q
        
        # Branch and bound
        if depth >= best_size[0]:
            return
    