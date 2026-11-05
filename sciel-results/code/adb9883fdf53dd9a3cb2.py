def _local_search(blocks):
    block_indices = set(blocks)
    counts = np.zeros(N_T, dtype=np.int32)
    for b_idx in block_indices:
        counts[block_t_ids[b_idx]] += 1
        
    improved = True
    while improved:
        improved = False
        
        # 1. Prune Redundant Blocks
        pruned_any = False
        for b_idx in list(block_indices):
            t_ids = block_t_ids[b_idx]
            if np.all(counts[t_ids] > 1):
                block_indices.remove(b_idx)
                counts[t_ids] -= 1
                pruned_any = True
        if pruned_any:
            improved = True
            continue # Restart loop to prune again
            
        # 2. Try Swaps
        swapped_any = False
        for b_old in list(block_indices):
            t_ids_old = block_t_ids[b_old]
            critical_mask = counts[t_ids_old] == 1
            if not np.any(critical_mask):
                # Redundant, should have been pruned
                block_indices.remove(b_old)
                counts[t_ids_old] -= 1
                swapped_any = True
                break
            critical_t_ids = t_ids_old[critical_mask]
            
            # Find candidates
            # Use precomputed sets
            candidates = t_to_blocks_set[critical_t_ids[0]]
            for tid in critical_t_ids[1:]:
                candidates = candidates & t_to_blocks_set[tid]
                if not candidates:
                    break
                    
            if not candidates:
                continue
                
            candidates = candidates - block_indices
            
            if not candidates:
                continue
                
            b_new = random.choice(list(candidates))
            
            # Perform swap
            block_indices.remove(b_old)
            block_indices.add(b_new)
            
            counts[t_ids_old] -= 1
            t_ids_new = block_t_ids[b_new]
            counts[t_ids_new] += 1
            
            swapped_any = True
            break # Restart loop to prune
            
        if swapped_any:
            improved = True

    return list(block_indices)
