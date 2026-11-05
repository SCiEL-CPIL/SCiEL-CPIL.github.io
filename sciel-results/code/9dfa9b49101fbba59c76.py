<<<<<<< SEARCH
def construct(v, k, t):
    if v != 19 or k != 7 or t != 4:
        # Fallback for other parameters (not expected in this task)
        uncovered = set(itertools.combinations(range(v), t))
        blocks = []
        all_k_subsets = list(itertools.combinations(range(v), k))
        random.shuffle(all_k_subsets)
        for b in all_k_subsets:
            if not uncovered:
                break
            covered = set(itertools.combinations(b, t))
            if uncovered & covered:
                blocks.append(list(b))
                uncovered -= covered
        return blocks

    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    def fast_greedy():
        """Fast weighted greedy using numpy for gain computation"""
        counts = np.zeros(N_SUBSETS, dtype=np.int32)
        current_blocks = []
        uncovered_mask = (1 << N_SUBSETS) - 1
        
        # Convert weights to numpy
        weights_np = np.array(weights, dtype=np.float64)
        
        while uncovered_mask:
            # Compute gain for each block: sum of weights of uncovered subsets it covers
            # For each block, gain = sum(weights[b] for b in block_bits if b is uncovered)
            
            # Efficient approach: for each uncovered bit, add its weight to all blocks containing it
            # But this is O(N_SUBSETS * avg_blocks_per_subset) per iteration
            
            # Alternative: iterate through blocks and compute gain
            best_idx = -1
            best_gain = 0.0
            
            # Only consider blocks that have at least one uncovered subset
            # Precompute: for each block, which subsets it covers
            # We need: gain[block] = sum(weights[subset] for subset in block_subsets if subset is uncovered)
            
            # Faster: maintain a running gain array
            # When we add a block, subtract its contribution from all other blocks' potential gain
            # This is complex, so let's use a simpler approach with early termination
            
            # Use numpy: for each block, compute dot product of its indicator vector with weights * uncovered_indicator
            # But creating indicator vectors is expensive
            
            # Practical approach: iterate blocks, compute gain by iterating its 35 subsets
            for i in range(N_BLOCKS):
                bits = block_bit_indices[i]
                gain = 0.0
                for bit in bits:
                    if (uncovered_mask >> bit) & 1:
                        gain += weights[bit]
                        if gain > best_gain:
                            # Early exit if we already beat the best
                            pass
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            
            if best_gain == 0:
                break
            
            current_blocks.append(best_idx)
            # Update uncovered_mask
            uncovered_mask &= ~all_masks[best_idx]
            
        return current_blocks
    
    # 1. Initial greedy
    current_blocks = fast_greedy()
    best_len = len(current_blocks)
    best_blocks = current_blocks.copy()
    
    # Build counts for current solution
    counts = np.zeros(N_SUBSETS, dtype=np.int32)
    for idx in current_blocks:
        for bit in block_bit_indices[idx]:
            counts[bit] += 1
    
    def is_redundant(idx, counts):
        """Check if block idx is redundant given current counts"""
        for bit in block_bit_indices[idx]:
            if counts[bit] == 1:
                return False
        return True
    
    def try_remove_blocks(block_indices, counts):
        """Try to remove a set of blocks. Returns True if successful."""
        # Check if all blocks are redundant
        for idx in block_indices:
            if not is_redundant(idx, counts):
                return False
        # Remove them
        for idx in block_indices:
            for bit in block_bit_indices[idx]:
                counts[bit] -= 1
            current_blocks.remove(idx)
        return True
    
    def try_swap_remove_add(remove_indices, add_indices, counts):
        """Remove some blocks and add others to maintain coverage"""
        # Check coverage after removal
        temp_counts = counts.copy()
        for idx in remove_indices:
            for bit in block_bit_indices[idx]:
                temp_counts[bit] -= 1
        
        # Check if any subset would become uncovered
        uncovered_bits = []
        for bit in set().union(*[set(block_bit_indices[idx]) for idx in remove_indices]):
            if temp_counts[bit] == 0:
                uncovered_bits.append(bit)
        
        if not uncovered_bits:
            # No uncovered bits, just remove
            for idx in remove_indices:
                for bit in block_bit_indices[idx]:
                    counts[bit] -= 1
                current_blocks.remove(idx)
            return True
        
        # Need to cover uncovered_bits with add_indices
        # Check if add_indices cover all uncovered_bits
        add_mask = 0
        for idx in add_indices:
            add_mask |= all_masks[idx]
        
        for bit in uncovered_bits:
            if not (add_mask >> bit) & 1:
                return False
        
        # Also check that add_indices don't create duplicates
        current_set = set(current_blocks)
        for idx in add_indices:
            if idx in current_set:
                return False
        
        # Apply the swap
        for idx in remove_indices:
            for bit in block_bit_indices[idx]:
                counts[bit] -= 1
            current_blocks.remove(idx)
        
        for idx in add_indices:
            for bit in block_bit_indices[idx]:
                counts[bit] += 1
            current_blocks.append(idx)
        
        return True
    
    # 2. Local Search with multiple strategies
    no_improve_count = 0
    max_no_improve = 200
    restart_count = 0
    max_restarts = 10
    
    while time.time() < deadline:
        # Strategy 1: Try to remove single redundant blocks
        removed_any = False
        indices = list(range(len(current_blocks)))
        random.shuffle(indices)
        
        for i in indices:
            if time.time() > deadline: break
            if is_redundant(current_blocks[i], counts):
                idx = current_blocks[i]
                for bit in block_bit_indices[idx]:
                    counts[bit] -= 1
                current_blocks.remove(idx)
                removed_any = True
                if len(current_blocks) < best_len:
                    best_len = len(current_blocks)
                    best_blocks = current_blocks.copy()
                break
        
        if removed_any:
            no_improve_count = 0
            continue
        
        # Strategy 2: Try 1-exchange (remove 1, add 1)
        if len(current_blocks) > 0:
            i = random.randint(0, len(current_blocks) - 1)
            idx_to_remove = current_blocks[i]
            
            # Find uncovered bits if we remove this block
            temp_counts = counts.copy()
            for bit in block_bit_indices[idx_to_remove]:
                temp_counts[bit] -= 1
            
            uncovered_bits = [bit for bit in block_bit_indices[idx_to_remove] if temp_counts[bit] == 0]
            
            if not uncovered_bits:
                # Block is redundant, remove it
                for bit in block_bit_indices[idx_to_remove]:
                    counts[bit] -= 1
                current_blocks.remove(idx_to_remove)
                if len(current_blocks) < best_len:
                    best_len = len(current_blocks)
                    best_blocks = current_blocks.copy()
                no_improve_count = 0
                continue
            
            # Find candidate blocks that cover all uncovered_bits
            if len(uncovered_bits) <= 5:
                # Use intersection of blocks containing each uncovered bit
                candidates = None
                for bit in uncovered_bits:
                    blocks_with_bit = set(subset_to_blocks[bit])
                    if candidates is None:
                        candidates = blocks_with_bit
                    else:
                        candidates &= blocks_with_bit
                
                if candidates:
                    candidates.discard(idx_to_remove)
                    if candidates:
                        # Try a random candidate
                        j = random.choice(list(candidates))
                        if try_swap_remove_add([idx_to_remove], [j], counts):
                            no_improve_count = 0
                            continue
        
        # Strategy 3: Try 2-exchange (remove 2, add 1-2)
        if len(current_blocks) > 1 and no_improve_count > 50:
            i1 = random.randint(0, len(current_blocks) - 1)
            i2 = random.randint(0, len(current_blocks) - 1)
            if i1 != i2:
                idx1 = current_blocks[i1]
                idx2 = current_blocks[i2]
                
                # Find uncovered bits if we remove both
                temp_counts = counts.copy()
                for bit in block_bit_indices[idx1]:
                    temp_counts[bit] -= 1
                for bit in block_bit_indices[idx2]:
                    temp_counts[bit] -= 1
                
                uncovered_bits = set()
                for bit in set(block_bit_indices[idx1]) | set(block_bit_indices[idx2]):
                    if temp_counts[bit] == 0:
                        uncovered_bits.add(bit)
                
                if uncovered_bits and len(uncovered_bits) <= 8:
                    # Try to find 1-2 blocks that cover all uncovered_bits
                    # First try 1 block
                    candidates = None
                    for bit in list(uncovered_bits)[:3]:  # Limit for speed
                        blocks_with_bit = set(subset_to_blocks[bit])
                        if candidates is None:
                            candidates = blocks_with_bit
                        else:
                            candidates &= blocks_with_bit
                    
                    if candidates:
                        candidates.discard(idx1)
                        candidates.discard(idx2)
                        if candidates:
                            j = random.choice(list(candidates))
                            if try_swap_remove_add([idx1, idx2], [j], counts):
                                no_improve_count = 0
                                continue
                    
                    # Try 2 blocks
                    if not candidates or len(candidates) < 3:
                        for _ in range(5):
                            j1 = random.randint(0, N_BLOCKS - 1)
                            j2 = random.randint(0, N_BLOCKS - 1)
                            if j1 == j2 or j1 in (idx1, idx2) or j2 in (idx1, idx2):
                                continue
                            if j1 in current_blocks or j2 in current_blocks:
                                continue
                            
                            combined_mask = all_masks[j1] | all_masks[j2]
                            covers_all = all((combined_mask >> bit) & 1 for bit in uncovered_bits)
                            if covers_all:
                                if try_swap_remove_add([idx1, idx2], [j1, j2], counts):
                                    no_improve_count = 0
                                    break
        
        # Strategy 4: Perturbation if stuck
        if not removed_any and no_improve_count > max_no_improve:
            # Perturb: add a random block, then try to prune
            c = random.randint(0, N_BLOCKS - 1)
            if c not in current_blocks:
                for bit in block_bit_indices[c]:
                    counts[bit] += 1
                current_blocks.append(c)
            
            # Try to prune
            prune_attempts = 0
            while prune_attempts < 20 and time.time() < deadline:
                prune_attempts += 1
                indices = list(range(len(current_blocks)))
                random.shuffle(indices)
                pruned = False
                for i in indices:
                    if is_redundant(current_blocks[i], counts):
                        idx = current_blocks[i]
                        for bit in block_bit_indices[idx]:
                            counts[bit] -= 1
                        current_blocks.remove(idx)
                        pruned = True
                        if len(current_blocks) < best_len:
                            best_len = len(current_blocks)
                            best_blocks = current_blocks.copy()
                        break
                if not pruned:
                    break
            
            no_improve_count = 0
            
            # If we've done many restarts, consider restarting from best
            restart_count += 1
            if restart_count > max_restarts:
                # Restart from best solution
                current_blocks = best_blocks.copy()
                counts = np.zeros(N_SUBSETS, dtype=np.int32)
                for idx in current_blocks:
                    for bit in block_bit_indices[idx]:
                        counts[bit] += 1
                restart_count = 0
                no_improve_count = 0
        else:
            no_improve_count += 1
    
    # Final cleanup
    for i in range(len(current_blocks)):
        if time.time() > deadline - 5:
            break
        idx = current_blocks[i]
        if is_redundant(idx, counts):
            for bit in block_bit_indices[idx]:
                counts[bit] -= 1
            current_blocks.remove(idx)
            i -= 1
            if len(current_blocks) < best_len:
                best_len = len(current_blocks)
                best_blocks = current_blocks.copy()
    
    return [list(all_blocks[i]) for i in best_blocks]
=======
def construct(v, k, t):
    if v != 19 or k != 7 or t != 4:
        # Fallback for other parameters (not expected in this task)
        uncovered = set(itertools.combinations(range(v), t))
        blocks = []
        all_k_subsets = list(itertools.combinations(range(v), k))
        random.shuffle(all_k_subsets)
        for b in all_k_subsets:
            if not uncovered:
                break
            covered = set(itertools.combinations(b, t))
            if uncovered & covered:
                blocks.append(list(b))
                uncovered -= covered
        return blocks

    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # Precompute numpy arrays for fast operations
    weights_np = np.array(weights, dtype=np.float64)
    
    def fast_greedy():
        """Fast weighted greedy using vectorized gain computation"""
        current_blocks = []
        # Track which subsets are uncovered using a boolean array
        uncovered = np.ones(N_SUBSETS, dtype=bool)
        uncovered_count = N_SUBSETS
        
        while uncovered_count > 0:
            # Compute gain for each block: sum of weights of uncovered subsets it covers
            # gain[i] = sum(weights_np[subset] for subset in block_subsets[i] if uncovered[subset])
            # Using matrix multiplication: gain = block_subsets_matrix @ (weights_np * uncovered)
            effective_weights = weights_np * uncovered
            gains = block_subsets_matrix @ effective_weights
            
            # Find block with maximum gain
            best_idx = int(np.argmax(gains))
            best_gain = gains[best_idx]
            
            if best_gain <= 0:
                break
            
            current_blocks.append(best_idx)
            
            # Update uncovered
            for bit in block_bit_indices[best_idx]:
                if uncovered[bit]:
                    uncovered[bit] = False
                    uncovered_count -= 1
            
            # Early termination check
            if time.time() > deadline - 30:
                break
        
        return current_blocks
    
    # 1. Initial greedy
    current_blocks = fast_greedy()
    best_len = len(current_blocks)
    best_blocks = current_blocks.copy()
    
    # Build counts for current solution
    counts = np.zeros(N_SUBSETS, dtype=np.int32)
    for idx in current_blocks:
        for bit in block_bit_indices[idx]:
            counts[bit] += 1
    
    def is_redundant(idx, counts):
        """Check if block idx is redundant given current counts"""
        for bit in block_bit_indices[idx]:
            if counts[bit] == 1:
                return False
        return True
    
    def try_swap_remove_add(remove_indices, add_indices, counts):
        """Remove some blocks and add others to maintain coverage"""
        # Check coverage after removal
        temp_counts = counts.copy()
        for idx in remove_indices:
            for bit in block_bit_indices[idx]:
                temp_counts[bit] -= 1
        
        # Check if any subset would become uncovered
        uncovered_bits = []
        for idx in remove_indices:
            for bit in block_bit_indices[idx]:
                if temp_counts[bit] == 0:
                    uncovered_bits.append(bit)
                    break
        
        if not uncovered_bits:
            # No uncovered bits, just remove
            for idx in remove_indices:
                for bit in block_bit_indices[idx]:
                    counts[bit] -= 1
                current_blocks.remove(idx)
            return True
        
        # Need to cover uncovered_bits with add_indices
        # Check if add_indices cover all uncovered_bits
        add_covered = set()
        for idx in add_indices:
            for bit in block_bit_indices[idx]:
                add_covered.add(bit)
        
        for bit in uncovered_bits:
            if bit not in add_covered:
                return False
        
        # Also check that add_indices don't create duplicates
        current_set = set(current_blocks)
        for idx in add_indices:
            if idx in current_set:
                return False
        
        # Apply the swap
        for idx in remove_indices:
            for bit in block_bit_indices[idx]:
                counts[bit] -= 1
            current_blocks.remove(idx)
        
        for idx in add_indices:
            for bit in block_bit_indices[idx]:
                counts[bit] += 1
            current_blocks.append(idx)
        
        return True
    
    # 2. Local Search with multiple strategies
    no_improve_count = 0
    max_no_improve = 100
    restart_count = 0
    max_restarts = 20
    
    while time.time() < deadline:
        # Strategy 1: Try to remove single redundant blocks (multiple passes)
        removed_any = False
        for _ in range(3):  # Multiple removal passes
            indices = list(range(len(current_blocks)))
            random.shuffle(indices)
            for i in indices:
                if time.time() > deadline: break
                if is_redundant(current_blocks[i], counts):
                    idx = current_blocks[i]
                    for bit in block_bit_indices[idx]:
                        counts[bit] -= 1
                    current_blocks.remove(idx)
                    removed_any = True
                    if len(current_blocks) < best_len:
                        best_len = len(current_blocks)
                        best_blocks = current_blocks.copy()
                    break
            if not removed_any:
                break
        
        if removed_any:
            no_improve_count = 0
            continue
        
        # Strategy 2: Try 1-exchange (remove 1, add 1)
        if len(current_blocks) > 0:
            i = random.randint(0, len(current_blocks) - 1)
            idx_to_remove = current_blocks[i]
            
            # Find uncovered bits if we remove this block
            temp_counts = counts.copy()
            for bit in block_bit_indices[idx_to_remove]:
                temp_counts[bit] -= 1
            
            uncovered_bits = [bit for bit in block_bit_indices[idx_to_remove] if temp_counts[bit] == 0]
            
            if not uncovered_bits:
                # Block is redundant, remove it
                for bit in block_bit_indices[idx_to_remove]:
                    counts[bit] -= 1
                current_blocks.remove(idx_to_remove)
                if len(current_blocks) < best_len:
                    best_len = len(current_blocks)
                    best_blocks = current_blocks.copy()
                no_improve_count = 0
                continue
            
            # Find candidate blocks that cover all uncovered_bits
            if len(uncovered_bits) <= 6:
                # Use intersection of blocks containing each uncovered bit
                candidates = None
                for bit in uncovered_bits:
                    blocks_with_bit = set(subset_to_blocks[bit])
                    if candidates is None:
                        candidates = blocks_with_bit
                    else:
                        candidates &= blocks_with_bit
                
                if candidates:
                    candidates.discard(idx_to_remove)
                    if candidates:
                        # Try multiple random candidates
                        cand_list = list(candidates)
                        random.shuffle(cand_list)
                        for j in cand_list[:3]:
                            if try_swap_remove_add([idx_to_remove], [j], counts):
                                no_improve_count = 0
                                break
        
        # Strategy 3: Try 2-exchange (remove 2, add 1)
        if len(current_blocks) > 1 and no_improve_count > 30:
            i1 = random.randint(0, len(current_blocks) - 1)
            i2 = random.randint(0, len(current_blocks) - 1)
            if i1 != i2:
                idx1 = current_blocks[i1]
                idx2 = current_blocks[i2]
                
                # Find uncovered bits if we remove both
                temp_counts = counts.copy()
                for bit in block_bit_indices[idx1]:
                    temp_counts[bit] -= 1
                for bit in block_bit_indices[idx2]:
                    temp_counts[bit] -= 1
                
                uncovered_bits = set()
                for bit in set(block_bit_indices[idx1]) | set(block_bit_indices[idx2]):
                    if temp_counts[bit] == 0:
                        uncovered_bits.add(bit)
                
                if uncovered_bits and len(uncovered_bits) <= 10:
                    # Try to find 1 block that covers all uncovered_bits
                    candidates = None
                    for bit in list(uncovered_bits)[:4]:  # Limit for speed
                        blocks_with_bit = set(subset_to_blocks[bit])
                        if candidates is None:
                            candidates = blocks_with_bit
                        else:
                            candidates &= blocks_with_bits
                    
                    if candidates:
                        candidates.discard(idx1)
                        candidates.discard(idx2)
                        if candidates:
                            cand_list = list(candidates)
                            random.shuffle(cand_list)
                            for j in cand_list[:3]:
                                if try_swap_remove_add([idx1, idx2], [j], counts):
                                    no_improve_count = 0
                                    break
        
        # Strategy 4: Perturbation if stuck
        if not removed_any and no_improve_count > max_no_improve:
            # Perturb: add 1-2 random blocks, then try to prune
            n_perturb = random.choice([1, 2])
            for _ in range(n_perturb):
                c = random.randint(0, N_BLOCKS - 1)
                if c not in current_blocks:
                    for bit in block_bit_indices[c]:
                        counts[bit] += 1
                    current_blocks.append(c)
            
            # Try to prune aggressively
            prune_attempts = 0
            while prune_attempts < 30 and time.time() < deadline:
                prune_attempts += 1
                indices = list(range(len(current_blocks)))
                random.shuffle(indices)
                pruned = False
                for i in indices:
                    if is_redundant(current_blocks[i], counts):
                        idx = current_blocks[i]
                        for bit in block_bit_indices[idx]:
                            counts[bit] -= 1
                        current_blocks.remove(idx)
                        pruned = True
                        if len(current_blocks) < best_len:
                            best_len = len(current_blocks)
                            best_blocks = current_blocks.copy()
                        break
                if not pruned:
                    break
            
            no_improve_count = 0
            
            # If we've done many restarts, consider restarting from best
            restart_count += 1
            if restart_count > max_restarts:
                # Restart from best solution
                current_blocks = best_blocks.copy()
                counts = np.zeros(N_SUBSETS, dtype=np.int32)
                for idx in current_blocks:
                    for bit in block_bit_indices[idx]:
                        counts[bit] += 1
                restart_count = 0
                no_improve_count = 0
        else:
            no_improve_count += 1
    
    # Final cleanup
    for i in range(len(current_blocks)):
        if time.time() > deadline - 5:
            break
        idx = current_blocks[i]
        if is_redundant(idx, counts):
            for bit in block_bit_indices[idx]:
                counts[bit] -= 1
            current_blocks.remove(idx)
            i -= 1
            if len(current_blocks) < best_len:
                best_len = len(current_blocks)
                best_blocks = current_blocks.copy()
    
    return [list(all_blocks[i]) for i in best_blocks]
>>>>>>> REPLACE
