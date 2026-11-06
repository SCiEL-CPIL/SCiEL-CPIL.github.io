    def backtrack(covered_mask, current_blocks):
        nonlocal best_blocks
        if len(current_blocks) == target_size:
            best_blocks = list(current_blocks)
            return True
            
        if time.time() - start_time > 110:
            return False
            
        available = (~covered_mask) & ((1 << n_5sets) - 1)
        if available == 0:
            return len(current_blocks) == target_size

        # MRV Heuristic: Find uncovered 5-set with fewest compatible blocks
        min_count = float('inf')
        min_idx = -1
        min_candidates = []
        
        # Iterate through set bits in available
        # available is a Python int
        avail = available
        while avail:
            lowest_bit = avail & -avail
            idx_5set = lowest_bit.bit_length() - 1
            avail -= lowest_bit # clear the bit
            
            candidates = blocks_by_5set[idx_5set]
            compatible = []
            for b_idx in candidates:
                if not (block_masks[b_idx] & covered_mask):
                    compatible.append(b_idx)
            
            count = len(compatible)
            if count == 0:
                return False
            if count < min_count:
                min_count = count
                min_idx = idx_5set
                min_candidates = compatible
                if count == 1:
                    break # Forced move, no need to check others
        
        # Now branch on min_candidates
        for b_idx in min_candidates:
            mask = block_masks[b_idx]
            current_blocks.append(b_idx)
            if backtrack(covered_mask | mask, current_blocks):
                return True
            current_blocks.pop()
            
        return False
