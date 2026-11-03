    def local_search_count(blocks_list):
        # cover_count[t_idx] = number of blocks covering t_idx
        cover_count = [0] * n_t_subsets
        
        # Initialize
        for b_idx in blocks_list:
            mask = masks[b_idx]
            # Iterate set bits in mask
            m = mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                cover_count[idx] += 1
                m ^= lsb
                
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks_list) - 1, -1, -1):
                b_idx = blocks_list[i]
                mask = masks[b_idx]
                
                # Check if any t-subset is covered ONLY by this block
                is_redundant = True
                m = mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    if cover_count[idx] == 1:
                        is_redundant = False
                        break
                    m ^= lsb
                    
                if is_redundant:
                    # Remove block
                    blocks_list.pop(i)
                    # Update cover_count
                    m = mask
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        cover_count[idx] -= 1
                        m ^= lsb
                    changed = True
                    break
        return blocks_list
