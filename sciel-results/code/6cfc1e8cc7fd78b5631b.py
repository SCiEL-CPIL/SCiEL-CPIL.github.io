    def compute_unique_dominator_counts(active_list, in_mask):
        n = len(active_list)
        # Map original indices to 0..n-1 for faster bitmask ops?
        # Or just use original indices in the mask.
        # in_mask is size p. active_mask is bits of active_list.
        
        counts = {v: 0 for v in active_list}
        
        # Iterate all 4-combinations of active_list
        for combo in combinations(active_list, 4):
            v1, v2, v3, v4 = combo
            # Compute common dominators
            # in_mask[x] has bit y if y->x
            # We need y such that y->v1, y->v2, y->v3, y->v4
            # So y is in in_mask[v1] AND in_mask[v2] AND in_mask[v3] AND in_mask[v4]
            common = in_mask[v1] & in_mask[v2] & in_mask[v3] & in_mask[v4]
            # Filter to active vertices
            # We need to count how many active vertices are in 'common'
            # If exactly 1, increment its count.
            
            # Efficient popcount and extraction of the single bit?
            # If common & active_mask has exactly 1 bit set:
            active_common = common & active_mask
            if active_common:
                # Check if power of 2
                if active_common & (active_common - 1) == 0:
                    # Find the vertex
                    # We need to map bit index back to vertex value
                    # This is slow if done naively.
                    # We can precompute a map: bit_index -> vertex_value?
                    # But vertex values are 0..66.
                    # We can use a lookup table for low bits?
                    # Or just:
                    # v = (active_common.bit_length() - 1)
                    # But wait, the bits in in_mask correspond to original indices 0..p-1.
                    # So yes, v = index.
                    v = active_common.bit_length() - 1
                    counts[v] += 1
        return counts
    