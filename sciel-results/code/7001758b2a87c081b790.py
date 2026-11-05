<<<<<<< SEARCH
        # Strategy 2: Perturb Best + Repair + Minimize
        if best_blocks is not None and len(best_blocks) > 10:
            perturbed = list(best_blocks)
            # Remove 3 random blocks
            num_remove = min(3, len(perturbed) - 1)
            if num_remove > 0:
                remove_indices = random.sample(range(len(perturbed)), num_remove)
                # Remove from end to start to keep indices valid, or use set
                # Easier: create a set of blocks to remove
                blocks_to_remove = set(perturbed[i] for i in remove_indices)
                
                # Build uncovered mask
                selected_mask = np.zeros(num_blocks, dtype=np.int32)
                for b_idx in perturbed:
                    selected_mask[b_idx] = 1
                cover_count_temp = incidence_matrix.T.dot(selected_mask).astype(np.int32)
                
                for b_idx in blocks_to_remove:
                    cover_count_temp -= incidence_matrix[b_idx].toarray().ravel()
                    
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
        # Strategy 2: Perturb Best + Repair + Minimize
        if best_blocks is not None and len(best_blocks) > 10:
            perturbed = list(best_blocks)
            # Remove 3 random blocks
            num_remove = min(3, len(perturbed) - 1)
            if num_remove > 0:
                remove_indices = random.sample(range(len(perturbed)), num_remove)
                blocks_to_remove = set(perturbed[i] for i in remove_indices)
                
                # Build cover count more efficiently
                selected_mask = np.zeros(num_blocks, dtype=np.int32)
                for b_idx in perturbed:
                    selected_mask[b_idx] = 1
                cover_count_temp = incidence_matrix.T.dot(selected_mask).astype(np.int32)
                
                b_indptr = incidence_matrix.indptr
                b_indices = incidence_matrix.indices
                for b_idx in blocks_to_remove:
                    start = b_indptr[b_idx]
                    end = b_indptr[b_idx + 1]
                    block_subsets = b_indices[start:end]
                    cover_count_temp[block_subsets] -= 1
                    
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
>>>>>>> REPLACE
