<<<<<<< SEARCH
    # Local Search Loop
    no_improve_counter = 0
    max_no_improve = 20
    
    while time.time() < deadline - 3.0:
        # 1. Delete Phase
        removed = run_delete_phase(current_code, cnt)
        if removed > 0:
            if len(current_code) < best_size:
                best_size = len(current_code)
                best_code = list(current_code)
                no_improve_counter = 0
            else:
                no_improve_counter += 1
        else:
            no_improve_counter += 1
        
        # 2. Swap Phase with smarter candidate selection
        swaps_attempted = 0
        max_swaps = 800
        while swaps_attempted < max_swaps and time.time() < deadline - 3.0:
            swaps_attempted += 1
            if len(current_code) == 0:
                break
                
            # Pick random codeword to remove
            c = random.choice(list(current_code))
            
            # Get valid swaps
            valid_swaps = get_valid_swaps(c, current_code, cnt)
            
            if valid_swaps is None:
                # Any non-code word works; pick a random one
                c_prime = random.randint(0, total - 1)
                while c_prime in current_code:
                    c_prime = random.randint(0, total - 1)
            elif len(valid_swaps) == 0:
                no_improve_counter += 1
                continue
            else:
                # Pick from valid swaps
                c_prime = random.choice(valid_swaps)
            
            # Perform swap
            current_code.remove(c)
            current_code.add(c_prime)
            cnt[ball_indices_np[c]] -= 1
            cnt[ball_indices_np[c_prime]] += 1
            
            # After swap, run delete phase
            removed = run_delete_phase(current_code, cnt)
            if removed > 0:
                if len(current_code) < best_size:
                    best_size = len(current_code)
                    best_code = list(current_code)
                    no_improve_counter = 0
                else:
                    no_improve_counter += 1
            else:
                no_improve_counter += 1
        
        # If no improvement for many iterations, try a restart
        if no_improve_counter > max_no_improve and time.time() < deadline - 15.0:
            # Try a new greedy initialization
            seed = int(time.time() * 1000) % 100000
            init_code = greedy_build(seed=seed)
            if len(init_code) < len(current_code) + 3:
                current_code = set(init_code)
                cnt = np.zeros(total, dtype=np.int32)
                for c in current_code:
                    cnt[ball_indices_np[c]] += 1
                no_improve_counter = 0
                
        if best_size <= 160:
            break
=======
    # Local Search Loop with improved swap evaluation
    no_improve_counter = 0
    max_no_improve = 15
    
    def try_swap(c_remove, c_add, code_set, cnt_arr):
        """Try a swap. Returns True if swap was made."""
        ball_remove = ball_indices_np[c_remove]
        ball_add = ball_indices_np[c_add]
        
        # Check: after removing c_remove, are all words still covered?
        # Words that become uncovered are those with cnt==1 in ball_remove
        # and not in ball_add
        will_be_uncovered = np.sum((cnt_arr[ball_remove] == 1) & ~ball_mem[c_add][ball_remove])
        if will_be_uncovered > 0:
            return False
        
        # Perform swap
        code_set.remove(c_remove)
        code_set.add(c_add)
        cnt_arr[ball_remove] -= 1
        cnt_arr[ball_add] += 1
        return True
    
    def find_best_swaps(code_set, cnt_arr, max_candidates=500):
        """Find promising swaps using vectorized evaluation."""
        code_list = list(code_set)
        if len(code_list) == 0:
            return []
        
        # Sample random codewords to consider for removal
        n_to_try = min(max_candidates, len(code_list))
        rng = random.Random()
        candidates = rng.sample(code_list, n_to_try)
        
        best_swaps = []
        for c in candidates:
            ball_c = ball_indices_np[c]
            unique_mask = cnt_arr[ball_c] == 1
            unique_count = np.sum(unique_mask)
            
            if unique_count == 0:
                # Any non-code word works; this is a "free" swap
                best_swaps.append((c, None, 0))  # 0 = any word
                continue
            
            # Find words that cover all unique words in ball_c
            unique_words = ball_c[unique_mask]
            valid_mask = np.ones(total, dtype=bool)
            for u in unique_words:
                valid_mask &= ball_mem[u]
            
            if not np.any(valid_mask):
                continue
            
            # Exclude words already in code
            code_mask = np.zeros(total, dtype=bool)
            code_mask[list(code_set)] = True
            valid_mask &= ~code_mask
            
            valid_indices = np.where(valid_mask)[0]
            if len(valid_indices) > 0:
                # Score by how many additional words this swap covers
                # (words in ball_add that have cnt < 2)
                sample = rng.choice(valid_indices, min(10, len(valid_indices)), replace=False)
                best_add = -1
                best_gain = -1
                for v in sample:
                    gain = np.sum(cnt_arr[ball_indices_np[v]] < 2)
                    if gain > best_gain:
                        best_gain = gain
                        best_add = v
                best_swaps.append((c, int(best_add), best_gain))
        
        # Sort by gain (descending), with None (free) swaps first
        best_swaps.sort(key=lambda x: (x[1] is not None, -x[2]))
        return best_swaps[:20]
    
    while time.time() < deadline - 3.0:
        # 1. Delete Phase
        removed = run_delete_phase(current_code, cnt)
        if removed > 0:
            if len(current_code) < best_size:
                best_size = len(current_code)
                best_code = list(current_code)
                no_improve_counter = 0
            else:
                no_improve_counter += 1
        else:
            no_improve_counter += 1
        
        # 2. Swap Phase with vectorized evaluation
        swaps_attempted = 0
        max_swaps = 600
        while swaps_attempted < max_swaps and time.time() < deadline - 3.0:
            swaps_attempted += 1
            if len(current_code) == 0:
                break
            
            # Find best swaps
            best_swaps = find_best_swaps(current_code, cnt, max_candidates=300)
            
            if not best_swaps:
                no_improve_counter += 1
                break
            
            improved = False
            for c_remove, c_add, gain in best_swaps:
                if c_add is None:
                    # Free swap: pick a random non-code word
                    c_add = random.randint(0, total - 1)
                    while c_add in current_code:
                        c_add = random.randint(0, total - 1)
                
                if try_swap(c_remove, c_add, current_code, cnt):
                    # After successful swap, run delete phase
                    removed = run_delete_phase(current_code, cnt)
                    if removed > 0:
                        if len(current_code) < best_size:
                            best_size = len(current_code)
                            best_code = list(current_code)
                            no_improve_counter = 0
                        else:
                            no_improve_counter += 1
                    else:
                        no_improve_counter += 1
                    improved = True
                    break
            
            if not improved:
                no_improve_counter += 1
        
        # If no improvement for many iterations, try a restart
        if no_improve_counter > max_no_improve and time.time() < deadline - 15.0:
            seed = int(time.time() * 1000) % 100000
            init_code = greedy_build(seed=seed)
            if len(init_code) < len(current_code) + 2:
                current_code = set(init_code)
                cnt = np.zeros(total, dtype=np.int32)
                for c in current_code:
                    cnt[ball_indices_np[c]] += 1
                no_improve_counter = 0
        
        if best_size <= 160:
            break
    
    # Final polish: aggressive delete + targeted swaps
    if time.time() < deadline - 2.0:
        run_delete_phase(current_code, cnt)
        if len(current_code) < best_size:
            best_size = len(current_code)
            best_code = list(current_code)
        
        # One more round of targeted swaps
        for _ in range(100):
            if time.time() > deadline - 2.0:
                break
            if len(current_code) == 0:
                break
            c = random.choice(list(current_code))
            ball_c = ball_indices_np[c]
            unique_mask = cnt[ball_c] == 1
            if np.sum(unique_mask) == 0:
                c_add = random.randint(0, total - 1)
                while c_add in current_code:
                    c_add = random.randint(0, total - 1)
                if try_swap(c, c_add, current_code, cnt):
                    run_delete_phase(current_code, cnt)
                    if len(current_code) < best_size:
                        best_size = len(current_code)
                        best_code = list(current_code)
            else:
                unique_words = ball_c[unique_mask]
                valid_mask = np.ones(total, dtype=bool)
                for u in unique_words:
                    valid_mask &= ball_mem[u]
                code_mask = np.zeros(total, dtype=bool)
                code_mask[list(current_code)] = True
                valid_mask &= ~code_mask
                valid_indices = np.where(valid_mask)[0]
                if len(valid_indices) > 0:
                    c_add = random.choice(valid_indices)
                    if try_swap(c, c_add, current_code, cnt):
                        run_delete_phase(current_code, cnt)
                        if len(current_code) < best_size:
                            best_size = len(current_code)
                            best_code = list(current_code)
>>>>>>> REPLACE

