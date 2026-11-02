            # Find uncovered t-subsets
            if uncovered_mask == 0:
                improved = True
                continue
            
            if uncovered_mask.bit_count() > 56:
                # No single block can cover all these
                add_block(idx_old)
                continue
