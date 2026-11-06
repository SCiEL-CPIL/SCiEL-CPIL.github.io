"""Optimized program using bitwise integer packing for constraint checking."""

def construct():
    colors = [0] * 128
    colors[0] = 0
    
    # Precompute bit masks for each color value
    # mask[k] = 1 << k
    # Since colors are in 0..127, we can use a list of 128 integers.
    bit_masks = [1 << i for i in range(128)]
    
    for i in range(1, 128):
        forbidden = 0
        
        # 4-term progressions ending at i: (i-3d, i-2d, i-d, i)
        # d can be 1 to i//3
        for d in range(1, i // 3 + 1):
            A = colors[i - d]
            B = colors[i - 2 * d]
            D = colors[i - 3 * d]
            
            # Check if A, B, D are all same
            if A == B and B == D:
                # This is a critical failure state.
                # We should have prevented this.
                # For now, mark all as forbidden.
                forbidden = (1 << 128) - 1
                break
            
            # Check if exactly 2 distinct
            if A == B or A == D or B == D:
                # At least two are equal. Since not all three, exactly two are equal.
                # Find the two distinct values.
                if A == B:
                    v1 = A
                    v2 = D
                elif A == D:
                    v1 = A
                    v2 = B
                else: # B == D
                    v1 = B
                    v2 = A
                
                forbidden |= bit_masks[v1]
                forbidden |= bit_masks[v2]
        
        # 3-term conditions: (i-2d, i-d, i)
        # If colors[i-2d] == colors[i-d], then c cannot be that color.
        for d in range(1, i // 2 + 1):
            A = colors[i - d]
            B = colors[i - 2 * d]
            if A == B:
                forbidden |= bit_masks[A]
        
        # Pick the smallest color not in forbidden
        # We can iterate 0..127
        found = False
        for c in range(128):
            if not (forbidden & bit_masks[c]):
                colors[i] = c
                found = True
                break
        
        if not found:
            # Fallback: pick 0 (should not happen if logic is correct)
            colors[i] = 0
    
    return colors
