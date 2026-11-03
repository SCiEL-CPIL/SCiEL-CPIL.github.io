<<<<<<< SEARCH
    def polish(selected, time_limit):
        if not selected:
            return selected
            
        # Compute cover_count
        cc = np.zeros(total_words, dtype=np.int32)
        for c in selected:
            cc[balls[c]] += 1
            
        # Helper to check if a word is in selected (for swap logic)
        # Using a set for O(1) lookup
        selected_set = set(selected)
        
        # Pass 1: Removals
        changed = True
        while changed and time.time() < time_limit:
            changed = False
            for i in range(len(selected)):
                c = selected[i]
                # Check if any word in balls[c] has cc == 1
                # If min(cc[balls[c]]) > 1, we can remove
                if np.min(cc[balls[c]]) > 1:
                    cc[balls[c]] -= 1
                    selected.pop(i)
                    selected_set.remove(c)
                    changed = True
                    break # Restart scan
        
        # Pass 2: Swaps to reduce bottlenecks
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            current_bottlenecks = np.count_nonzero(cc == 1)
            
            for i in range(len(selected)):
                if time.time() > time_limit:
                    break
                c = selected[i]
                
                # Find unique words covered by c (bottlenecks assigned to c)
                # unique_mask is not a bitmask, it's a list of indices
                unique_words = [w for w in balls[c] if cc[w] == 1]
                
                if not unique_words:
                    # This codeword is redundant
                    cc[balls[c]] -= 1
                    selected.pop(i)
                    selected_set.remove(c)
                    improved = True
                    break # Restart
                    
                # Find a candidate c' not in selected
                # such that ball_masks[c'] covers unique_words
                # (balls[c'] contains all w in unique_words)
                
                best_cand = -1
                best_delta = 1000 # infinity
                # We want to minimize the increase in bottlenecks
                # Delta = (new bottlenecks from losing c) - (bottlenecks removed by gaining c')
                
                # To speed up, we can iterate all candidates
                for c_prime in range(total_words):
                    if c_prime in selected_set:
                        continue
                    
                    # Check coverage of unique_words
                    # If any w in unique_words is NOT in balls[c_prime], skip
                    # This check is O(len(unique_words))
                    # unique_words is small (usually 1-3)
                    valid = True
                    for w in unique_words:
                        if w not in balls[c_prime]:
                            valid = False
                            break
                    if not valid:
                        continue
                    
                    # Calculate Delta Bottlenecks
                    # Losing words: w in balls[c] but NOT in balls[c_prime]
                    # Gaining words: w in balls[c_prime] but NOT in balls[c]
                    
                    # We need to count:
                    # 1. w in balls[c] \ balls[c_prime] with cc[w] == 2 (become 1)
                    # 2. w in balls[c_prime] \ balls[c] with cc[w] == 1 (become 2)
                    
                    # Optimization: Pre-compute sets? No, just loop.
                    # balls[c] and balls[c_prime] are size 15.
                    # Intersection/Difference is small.
                    
                    # Create a set for balls[c_prime] for fast lookup?
                    # Or just use numpy?
                    # Given small size, Python sets are fine.
                    set_cp = set(balls[c_prime])
                    set_c = set(balls[c])
                    
                    losing = set_c - set_cp
                    gaining = set_cp - set_c
                    
                    new_bottlenecks_from_loss = 0
                    for w in losing:
                        if cc[w] == 2:
                            new_bottlenecks_from_loss += 1
                            
                    bottlenecks_removed_from_gain = 0
                    for w in gaining:
                        if cc[w] == 1:
                            bottlenecks_removed_from_gain += 1
                            
                    delta = new_bottlenecks_from_loss - bottlenecks_removed_from_gain
                    
                    if delta < best_delta:
                        best_delta = delta
                        best_cand = c_prime
                        
                if best_cand != -1:
                    # Perform swap
                    cc[balls[c]] -= 1
                    cc[balls[best_cand]] += 1
                    selected_set.remove(c)
                    selected[i] = best_cand
                    selected_set.add(best_cand)
                    improved = True
                    break # Restart scan
        
        # Pass 3: Final Removals
        changed = True
        while changed and time.time() < time_limit:
            changed = False
            for i in range(len(selected)):
                c = selected[i]
                if np.min(cc[balls[c]]) > 1:
                    cc[balls[c]] -= 1
                    selected.pop(i)
                    selected_set.remove(c)
                    changed = True
                    break
                    
        return selected
=======
    def polish(selected, time_limit):
        if not selected:
            return selected
            
        # Compute cover_count
        cc = np.zeros(total_words, dtype=np.int32)
        for c in selected:
            cc[balls[c]] += 1
            
        selected_set = set(selected)
        
        def try_remove_redundant():
            """Remove any codeword that doesn't uniquely cover any word."""
            nonlocal cc, selected, selected_set
            for i in range(len(selected)):
                c = selected[i]
                if np.min(cc[balls[c]]) > 1:
                    cc[balls[c]] -= 1
                    selected.pop(i)
                    selected_set.remove(c)
                    return True
            return False
        
        # Pass 1: Removals
        changed = True
        while changed and time.time() < time_limit:
            changed = try_remove_redundant()
        
        # Pass 2: Aggressive swaps using reverse_cover
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            
            for i in range(len(selected)):
                if time.time() > time_limit:
                    break
                c = selected[i]
                
                # Find unique words covered by c
                unique_words = [w for w in balls[c] if cc[w] == 1]
                
                if not unique_words:
                    cc[balls[c]] -= 1
                    selected.pop(i)
                    selected_set.remove(c)
                    improved = True
                    break
                    
                # Use reverse_cover to find candidates that cover ALL unique_words
                # For each unique word, get list of codewords covering it
                # Intersect these lists to find candidates
                candidate_set = None
                for w in unique_words:
                    rc = reverse_cover[w]
                    if candidate_set is None:
                        candidate_set = set(rc)
                    else:
                        candidate_set &= set(rc)
                
                if not candidate_set:
                    continue
                
                # Remove already-selected codewords from candidates
                candidate_set -= selected_set
                
                if not candidate_set:
                    continue
                
                # Score each candidate by bottleneck delta
                best_cand = -1
                best_delta = 1000
                
                set_c = ball_sets[c]
                
                for c_prime in candidate_set:
                    set_cp = ball_sets[c_prime]
                    
                    losing = set_c - set_cp
                    gaining = set_cp - set_c
                    
                    new_bottlenecks_from_loss = 0
                    for w in losing:
                        if cc[w] == 2:
                            new_bottlenecks_from_loss += 1
                            
                    bottlenecks_removed_from_gain = 0
                    for w in gaining:
                        if cc[w] == 1:
                            bottlenecks_removed_from_gain += 1
                            
                    delta = new_bottlenecks_from_loss - bottlenecks_removed_from_gain
                    
                    if delta < best_delta:
                        best_delta = delta
                        best_cand = c_prime
                        
                if best_cand != -1:
                    cc[balls[c]] -= 1
                    cc[balls[best_cand]] += 1
                    selected_set.remove(c)
                    selected[i] = best_cand
                    selected_set.add(best_cand)
                    improved = True
                    break # Restart scan
        
        # Pass 3: 2-for-1 swaps (remove two, add one)
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            
            n_sel = len(selected)
            for i in range(n_sel):
                if time.time() > time_limit:
                    break
                c1 = selected[i]
                
                for j in range(i + 1, n_sel):
                    if time.time() > time_limit:
                        break
                    c2 = selected[j]
                    
                    # Words uniquely covered by c1 or c2
                    unique1 = [w for w in balls[c1] if cc[w] == 1]
                    unique2 = [w for w in balls[c2] if cc[w] == 1]
                    
                    if not unique1 and not unique2:
                        # Both redundant
                        cc[balls[c1]] -= 1
                        cc[balls[c2]] -= 1
                        selected.pop(j)
                        selected.pop(i)
                        selected_set.remove(c1)
                        selected_set.remove(c2)
                        improved = True
                        break
                    
                    if not unique1 or not unique2:
                        continue
                    
                    # Find candidate that covers all unique words from both
                    all_unique = set(unique1 + unique2)
                    candidate_set = None
                    for w in all_unique:
                        rc = reverse_cover[w]
                        if candidate_set is None:
                            candidate_set = set(rc)
                        else:
                            candidate_set &= set(rc)
                    
                    if not candidate_set:
                        continue
                    
                    candidate_set -= selected_set
                    
                    if not candidate_set:
                        continue
                    
                    best_cand = -1
                    best_delta = 1000
                    
                    set_c1 = ball_sets[c1]
                    set_c2 = ball_sets[c2]
                    
                    for c_prime in candidate_set:
                        set_cp = ball_sets[c_prime]
                        
                        losing = (set_c1 | set_c2) - set_cp
                        gaining = set_cp - (set_c1 | set_c2)
                        
                        new_bottlenecks_from_loss = 0
                        for w in losing:
                            if cc[w] == 2:
                                new_bottlenecks_from_loss += 1
                                
                        bottlenecks_removed_from_gain = 0
                        for w in gaining:
                            if cc[w] == 1:
                                bottlenecks_removed_from_gain += 1
                                
                        # We're removing 2, adding 1, so net -1 if valid
                        # Delta measures bottleneck change
                        delta = new_bottlenecks_from_loss - bottlenecks_removed_from_gain
                        
                        if delta < best_delta:
                            best_delta = delta
                            best_cand = c_prime
                    
                    if best_cand != -1:
                        cc[balls[c1]] -= 1
                        cc[balls[c2]] -= 1
                        cc[balls[best_cand]] += 1
                        selected.pop(j)
                        selected.pop(i)
                        selected_set.remove(c1)
                        selected_set.remove(c2)
                        selected.append(best_cand)
                        selected_set.add(best_cand)
                        improved = True
                        break
                
                if improved:
                    break
        
        # Pass 4: Final Removals
        changed = True
        while changed and time.time() < time_limit:
            changed = try_remove_redundant()
                    
        return selected
>>>>>>> REPLACE

