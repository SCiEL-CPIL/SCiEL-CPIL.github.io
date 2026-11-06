"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    n = 128
    colors = [0] * n
    # Precompute bit masks for colors 0..127
    bit_masks = [1 << i for i in range(128)]
    
    for x in range(1, n + 1):
        idx = x - 1
        found = False
        # Try colors in increasing order to minimize the number of distinct colors used
        for c in range(128):
            valid = True
            mask_c = bit_masks[c]
            # Check all 4-APs ending at x
            # x, x-d, x-2d, x-3d
            # x-3d >= 1 => 3d <= x-1 => d <= (x-1)//3
            max_d = (x - 1) // 3
            for d in range(1, max_d + 1):
                i1 = x - 1 - 3 * d
                i2 = x - 1 - 2 * d
                i3 = x - 1 - d
                
                # Retrieve masks for the previous three terms
                m1 = bit_masks[colors[i1]]
                m2 = bit_masks[colors[i2]]
                m3 = bit_masks[colors[i3]]
                
                # Combine with current color
                union = m1 | m2 | m3 | mask_c
                
                # Constraint: at least 3 distinct colors
                if union.bit_count() < 3:
                    valid = False
                    break
            
            if valid:
                colors[idx] = c
                found = True
                break
        
        if not found:
            # Fallback: if no color works (should not happen for this N), 
            # we might need a more complex search, but for now we leave it as 0
            # or raise an error. Given the problem constraints, a solution exists.
            pass

    return colors
# EVOLVE-BLOCK-END
