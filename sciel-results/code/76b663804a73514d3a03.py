"""Optimized program using bitwise integer packing for constraint checking."""

def construct():
    # Precompute masks for each color value 0..127
    masks = [1 << i for i in range(128)]
    
    colors = [0] * 128
    colors[0] = 0
    
    for i in range(1, 128):
        forbidden = 0
        
        # 4-term progressions ending at i: (i-3d, i-2d, i-d, i)
        # d can be 1 to i//3
        for d in range(1, i // 3 + 1):
            A = colors[i - 3 * d]
            B = colors[i - 2 * d]
            C = colors[i - d]
            
            # Based on analysis, A==B==C is prevented by previous steps.
            # We handle cases where at least two are equal.
            if A == B:
                # Forbid A and C (if A==C, we forbid A, which covers both, but logic holds)
                # If A==B==C, we forbid A and C (same), but actually all colors are forbidden.
                # However, A==B==C should not happen. If it does, we might need to handle it.
                # The 3-term check ensures C != A if A==B. So A!=C here.
                forbidden |= masks[A] | masks[C]
            elif B == C:
                forbidden |= masks[B] | masks[A]
            elif A == C:
                forbidden |= masks[A] | masks[B]
            # If all distinct, no restriction from this 4-AP.
            
        # 3-term conditions: (i-2d, i-d, i)
        # If colors[i-2d] == colors[i-d], then c_i cannot be that color.
        for d in range(1, i // 2 + 1):
            A = colors[i - d]
            B = colors[i - 2 * d]
            if A == B:
                forbidden |= masks[A]
        
        # Pick the smallest color not in forbidden
        # We can iterate 0..127
        for c in range(128):
            if not (forbidden & masks[c]):
                colors[i] = c
                break
        else:
            # Fallback: pick 0 (should not happen if logic is correct)
            colors[i] = 0
            
    return colors
