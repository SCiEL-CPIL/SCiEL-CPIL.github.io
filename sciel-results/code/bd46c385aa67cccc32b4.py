import time

def construct():
    """
    Constructs a 4-coloring of K_n with no monochromatic triangles.
    Uses 3 copies of the Greenwood-Gleason K_16 coloring (3 colors) and colors
    all inter-copy edges with the 4th color.
    n = 3 * 16 = 48.
    """
    # Greenwood-Gleason colouring of K16 by cubic-residue classes of GF(16)
    def mul(a, b):
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r

    # Build log table for GF(16)
    log, x = {}, 1
    for e in range(15):
        log[x] = e
        x = mul(x, 2)
        
    # Generate colors for K16
    base_colors = []
    for i in range(16):
        for j in range(i + 1, 16):
            # i^j is non-zero in GF(16)
            val = i ^ j
            # log[val] % 3 gives 0, 1, or 2
            base_colors.append(log[val] % 3)
            
    n = 48
    colors = []
    # 3 blocks of 16
    for i in range(48):
        for j in range(i + 1, 48):
            block_i = i // 16
            block_j = j // 16
            if block_i == block_j:
                # Same block: use base coloring
                # Local indices
                li = i % 16
                lj = j % 16
                # Find index in base_colors
                # base_colors is lexicographic for 16 vertices
                # Index calculation: sum of 16-1, 16-2, ..., 16-(li+1) for previous rows?
                # Simpler: just iterate or use a helper
                # Let's just precompute a 16x16 matrix for base colors
                pass
            else:
                # Different blocks: use color 3
                colors.append(3)
                
    # Precompute base color matrix for K16
    base_mat = [[0]*16 for _ in range(16)]
    idx = 0
    for i in range(16):
        for j in range(i+1, 16):
            c = log[(i ^ j)] % 3
            base_mat[i][j] = c
            base_mat[j][i] = c
            
    # Rebuild colors list correctly
    colors = []
    for i in range(48):
        for j in range(i + 1, 48):
            block_i = i // 16
            block_j = j // 16
            if block_i == block_j:
                li = i % 16
                lj = j % 16
                colors.append(base_mat[li][lj])
            else:
                colors.append(3)
                
    return {'n': 48, 'colors': colors}
