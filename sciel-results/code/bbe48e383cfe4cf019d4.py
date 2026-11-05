    def sa_optimize(initial_blocks):
        current_blocks = list(initial_blocks)
        cover_count = np.zeros(n_subsets, dtype=np.int32)
        
        for b_idx in current_blocks:
            for sid in block_subsets[b_idx]:
                cover_count[sid] += 1
        
        best_blocks = list(current_blocks)
        best_count = len(current_blocks)
        
        T = 1.0
        T_min = 0.01
        cooling = 0.9995
        start_time = time.time()
        time_limit = 110.0
        
        active_set = set(current_blocks)
        
        while time.time() - start_time < time_limit:
            move_type = random.random()
            
            if move_type < 0.6: # 60% removal attempts
                if not current_blocks:
                    continue
                idx = random.randint(0, len(current_blocks)-1)
                b_idx = current_blocks[idx]
                
                can_remove = True
                for sid in block_subsets[b_idx]:
                    if cover_count[sid] == 1:
                        can_remove = False
                        break
                
                if can_remove:
                    current_blocks.pop(idx)
                    active_set.remove(b_idx)
                    for sid in block_subsets[b_idx]:
                        cover_count[sid] -= 1
                    
                    if len(current_blocks) < best_count:
                        best_count = len(current_blocks)
                        best_blocks = list(current_blocks)
            else: # 40% swap attempts
                if not current_blocks:
                    continue
                idx = random.randint(0, len(current_blocks)-1)
                old_b_idx = current_blocks[idx]
                
                new_b_idx = random.randint(0, n_blocks-1)
                if new_b_idx in active_set:
                    continue
                
                # Check validity
                valid_swap = True
                criticals = []
                for sid in block_subsets[old_b_idx]:
                    if cover_count[sid] == 1:
                        criticals.append(sid)
                
                if criticals:
                    new_subs_set = set(block_subsets[new_b_idx])
                    for sid in criticals:
                        if sid not in new_subs_set:
                            valid_swap = False
                            break
                
                if valid_swap:
                    current_blocks[idx] = new_b_idx
                    active_set.remove(old_b_idx)
                    active_set.add(new_b_idx)
                    
                    for sid in block_subsets[old_b_idx]:
                        cover_count[sid] -= 1
                    for sid in block_subsets[new_b_idx]:
                        cover_count[sid] += 1
            
            T *= cooling
            if T < T_min:
                T = T_min
                
        return best_blocks
