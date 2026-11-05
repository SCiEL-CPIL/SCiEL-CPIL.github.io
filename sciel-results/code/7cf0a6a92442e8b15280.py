<<<<<<< SEARCH
        # Strategy 2: Perturb Best + Repair + Minimize (more aggressive)
        if best_blocks is not None and len(best_blocks) > 10:
            perturbed = list(best_blocks)
            # Remove 5 random blocks for more aggressive perturbation
            num_remove = min(5, len(perturbed) - 1)
            if num_remove > 0:
                remove_indices = random.sample(range(len(perturbed)), num_remove)
                blocks_to_remove = set(perturbed[i] for i in remove_indices)
                
                # Build uncovered mask
                cover_count_temp = np.zeros(num_subsets, dtype=np.int32)
                for b_idx in perturbed:
                    if b_idx not in blocks_to_remove:
                        cover_count_temp[block_to_subsets_arr[b_idx]] += 1
                        
                uncovered_mask = (cover_count_temp == 0).astype(bool)
                
                # Repair
                repair_res = run_greedy_partial(uncovered_mask, trial + 1000)
                
                # Combine
                new_set = [b for b in perturbed if b not in blocks_to_remove]
                new_set.extend(repair_res)
                
                # Minimize
                min_res2 = minimize(new_set, trial + 2000)
                
                if len(min_res2) < best_len:
                    best_len = len(min_res2)
                    best_blocks = min_res2
=======
        # Strategy 2: Perturb Best + Repair + Minimize (more aggressive)
        if best_blocks is not None and len(best_blocks) > 10:
            perturbed = list(best_blocks)
            # Remove 3 random blocks (less aggressive to maintain validity)
            num_remove = min(3, len(perturbed) - 1)
            if num_remove > 0:
                remove_indices = random.sample(range(len(perturbed)), num_remove)
                blocks_to_remove = set(perturbed[i] for i in remove_indices)
                
                # Build uncovered mask
                cover_count_temp = np.zeros(num_subsets, dtype=np.int32)
                for b_idx in perturbed:
                    if b_idx not in blocks_to_remove:
                        cover_count_temp[block_to_subsets_arr[b_idx]] += 1
                        
                uncovered_mask = (cover_count_temp == 0).astype(bool)
                
                # Repair
                repair_res = run_greedy_partial(uncovered_mask, trial + 1000)
                
                # Combine
                new_set = [b for b in perturbed if b not in blocks_to_remove]
                new_set.extend(repair_res)
                
                # Deduplicate
                new_set = list(set(new_set))
                
                # Minimize
                min_res2 = minimize(new_set, trial + 2000)
                
                if len(min_res2) < best_len:
                    best_len = len(min_res2)
                    best_blocks = min_res2
>>>>>>> REPLACE

