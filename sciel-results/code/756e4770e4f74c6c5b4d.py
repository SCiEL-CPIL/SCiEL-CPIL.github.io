    def minimize_and_improve(blocks):
        if not blocks:
            return blocks
            
        # Initialize cover counts
        # n_subsets is 11628
        cover_count = [0] * n_subsets
        for b_idx in blocks:
            mask = masks[b_idx]
            # Iterate set bits in mask
            m = mask
            while m:
                # Get lowest set bit
                lsb = m & -m
                idx = lsb.bit_length() - 1
                cover_count[idx] += 1
                m ^= lsb
        
        # Phase 1: Remove redundant
        changed = True
        while changed:
            changed = False
            new_blocks = []
            for i, b_idx in enumerate(blocks):
                # Check if b_idx has any unique subsets
                mask = masks[b_idx]
                m = mask
                is_redundant = True
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    if cover_count[idx] == 1:
                        is_redundant = False
                        break
                    m ^= lsb
                
                if is_redundant:
                    # Remove
                    m = masks[b_idx]
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        cover_count[idx] -= 1
                        m ^= lsb
                    changed = True
                else:
                    new_blocks.append(b_idx)
            
            if changed:
                blocks = new_blocks
        
        # Phase 2: 1-Exchange
        # We want to swap a block if it leads to a reduction?
        # Or just maintain validity and hope for the best?
        # Let's try to find a swap that allows a subsequent removal.
        # This is complex.
        # Simpler: Just try swaps that are "valid" (cover unique subsets).
        # If we swap, we keep the same size.
        # The reduction comes from Phase 1.
        # So we alternate Phase 1 and Phase 2.
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i, b_idx in enumerate(blocks):
                mask = masks[b_idx]
                m = mask
                unique_mask = 0
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    if cover_count[idx] == 1:
                        unique_mask |= lsb
                    m ^= lsb
                
                if unique_mask == 0:
                    # Remove
                    m = masks[b_idx]
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        cover_count[idx] -= 1
                        m ^= lsb
                    blocks.pop(i)
                    improved = True
                    break # Restart loop
                
                # Look for replacement
                # We need a block j not in blocks such that masks[j] covers unique_mask
                # Optimization: We can precompute for each subset the blocks that cover it?
                # No, just iterate.
                
                found_swap = False
                for j in range(n_blocks):
                    if j in blocks:
                        continue
                    # Check if j covers unique_mask
                    if (masks[j] & unique_mask) == unique_mask:
                        # Perform swap
                        # Remove b_idx
                        m = masks[b_idx]
                        while m:
                            lsb = m & -m
                            idx = lsb.bit_length() - 1
                            cover_count[idx] -= 1
                            m ^= lsb
                        
                        # Add j
                        m = masks[j]
                        while m:
                            lsb = m & -m
                            idx = lsb.bit_length() - 1
                            cover_count[idx] += 1
                            m ^= lsb
                        
                        blocks[i] = j
                        improved = True
                        found_swap = True
                        break
                
                if found_swap:
                    break # Restart loop
        
        return blocks
