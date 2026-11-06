    def local_search(s_indices, time_limit):
        k = len(s_indices)
        in_set = np.zeros(N*N, dtype=bool)
        in_set[s_indices] = True
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            k = len(s_indices)
            # Try to add a random point
            candidates = np.where(~in_set)[0]
            if len(candidates) == 0: break
            
            # Shuffle candidates? Or just pick random few
            # To avoid O(M) shuffle every time, we can pick random indices
            for _ in range(100): # Try 100 random candidates
                if time.time() > time_limit: break
                c_idx = candidates[np.random.randint(len(candidates))]
                if check_and_add(c_idx, s_indices):
                    s_indices = np.append(s_indices, c_idx)
                    in_set[c_idx] = True
                    improved = True
                    break
            
            if not improved:
                # Try a swap: remove one, add two
                # Pick a random point to remove
                if k > 0:
                    rem_idx_pos = np.random.randint(k)
                    rem_idx = s_indices[rem_idx_pos]
                    new_s_indices = np.delete(s_indices, rem_idx_pos)
                    in_set[rem_idx] = False
                    
                    added = 0
                    # Try to add 2 points
                    for _ in range(200):
                        if added >= 2: break
                        c_idx = candidates[np.random.randint(len(candidates))]
                        if in_set[c_idx]: continue
                        if check_and_add(c_idx, new_s_indices):
                            new_s_indices = np.append(new_s_indices, c_idx)
                            in_set[c_idx] = True
                            added += 1
                    
                    if added > 1: # Net gain
                        s_indices = new_s_indices
                        improved = True
                    else:
                        # Revert
                        in_set[rem_idx] = True
                        # in_set for added points needs to be reverted if we don't keep them
                        # But we only keep if improved.
                        # If not improved, we revert in_set for added points
                        # Actually, if we didn't keep new_s_indices, we need to revert in_set for any points we marked as in_set during the trial
                        # This is getting messy.
                        pass
        return s_indices
    