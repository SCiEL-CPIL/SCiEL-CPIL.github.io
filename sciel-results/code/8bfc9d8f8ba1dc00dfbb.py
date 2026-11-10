    def post_process(selected_indices):
        cover_count = [0] * num_t_subsets
        zero_count = num_t_subsets # Initially all uncovered? No, we build counts from selected blocks.
        
        # Actually, we start with counts 0.
        # Then we add blocks.
        # Initially, all 2024 are 0. So zero_count = 2024.
        
        for idx in selected_indices:
            mask = masks[idx]
            m = mask
            while m:
                lsb = m & -m
                bit_pos = lsb.bit_length() - 1
                if cover_count[bit_pos] == 0:
                    zero_count -= 1
                cover_count[bit_pos] += 1
                m ^= lsb
                
        improved = True
        current_indices = list(selected_indices)
        
        while improved:
            improved = False
            new_indices = []
            for idx in current_indices:
                mask = masks[idx]
                # Decrement counts
                m = mask
                while m:
                    lsb = m & -m
                    bit_pos = lsb.bit_length() - 1
                    cover_count[bit_pos] -= 1
                    if cover_count[bit_pos] == 0:
                        zero_count += 1
                    m ^= lsb
                    
                # Check if any count is 0
                if zero_count > 0:
                    # Put it back
                    m = mask
                    while m:
                        lsb = m & -m
                        bit_pos = lsb.bit_length() - 1
                        if cover_count[bit_pos] == 0:
                            zero_count -= 1
                        cover_count[bit_pos] += 1
                        m ^= lsb
                    new_indices.append(idx)
                else:
                    # Keep it removed
                    improved = True
                    
            current_indices = new_indices
        return current_indices
