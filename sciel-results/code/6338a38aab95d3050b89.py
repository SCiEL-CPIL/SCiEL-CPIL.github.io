<<<<<<< SEARCH
    def run_greedy_partial(uncovered_mask, random_seed=None):
        if random_seed is not None:
            rng = np.random.default_rng(random_seed)
        else:
            rng = np.random.default_rng()
            
        uncovered_indices = np.where(uncovered_mask)[0]
        if len(uncovered_indices) == 0:
            return []
            
        gains = np.zeros(num_blocks, dtype=np.int32)
        for s_idx in uncovered_indices:
            for b_idx in subset_block_arrays[s_idx]:
                gains[b_idx] += 1
                
        selected_blocks = []
        current_uncovered = uncovered_mask.copy()
        
        while True:
            if not np.any(current_uncovered):
                break
                
            noise = rng.uniform(0, 0.3, size=num_blocks)
            scores = gains + noise
            best_idx = int(np.argmax(scores))
            
            if gains[best_idx] == 0:
                break
                
            selected_blocks.append(best_idx)
            
            b_subs = block_subset_arrays[best_idx]
            newly_covered = b_subs[current_uncovered[b_subs]]
            
            if len(newly_covered) == 0:
                continue
                
            current_uncovered[newly_covered] = False
            for s in newly_covered:
                for b in subset_block_arrays[s]:
                    gains[b] -= 1
                    
        return selected_blocks
=======
    def run_greedy_partial(uncovered_mask, random_seed=None):
        if random_seed is not None:
            rng = np.random.default_rng(random_seed)
        else:
            rng = np.random.default_rng()
            
        uncovered_indices = np.where(uncovered_mask)[0]
        if len(uncovered_indices) == 0:
            return []
            
        gains = np.zeros(num_blocks, dtype=np.int32)
        for s_idx in uncovered_indices:
            for b_idx in subset_block_arrays[s_idx]:
                gains[b_idx] += 1
                
        selected_blocks = []
        current_uncovered = uncovered_mask.copy()
        
        while True:
            if not np.any(current_uncovered):
                break
                
            noise = rng.uniform(0, 0.3, size=num_blocks)
            scores = gains.astype(np.float64) + noise
            best_idx = int(np.argmax(scores))
            
            if gains[best_idx] == 0:
                break
                
            selected_blocks.append(best_idx)
            
            b_subs = block_subset_arrays[best_idx]
            newly_covered = b_subs[current_uncovered[b_subs]]
            
            if len(newly_covered) == 0:
                continue
                
            current_uncovered[newly_covered] = False
            for s in newly_covered:
                for b in subset_block_arrays[s]:
                    gains[b] -= 1
                    
        return selected_blocks

    def run_lla_construction(random_seed=None):
        if random_seed is not None:
            rng = np.random.default_rng(random_seed)
        else:
            rng = np.random.default_rng()
            
        uncovered = np.ones(num_subsets, dtype=bool)
        selected_blocks = []
        used_blocks = np.zeros(num_blocks, dtype=bool)
        
        while np.any(uncovered):
            unc_indices = np.where(uncovered)[0]
            if len(unc_indices) == 0:
                break
            target_subset = unc_indices[rng.integers(0, len(unc_indices))]
            
            candidates = subset_block_arrays[target_subset]
            valid_candidates = [b for b in candidates if not used_blocks[b]]
            if not valid_candidates:
                break
                
            chosen_block = valid_candidates[rng.integers(0, len(valid_candidates))]
            
            selected_blocks.append(chosen_block)
            used_blocks[chosen_block] = True
            
            b_subs = block_subset_arrays[chosen_block]
            newly_covered = b_subs[uncovered[b_subs]]
            if len(newly_covered) > 0:
                uncovered[newly_covered] = False
                
        return selected_blocks
>>>>>>> REPLACE

