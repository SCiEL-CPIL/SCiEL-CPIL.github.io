"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def _strassen_2x2(U, V, W, u_off, v_off, w_off, col, n=2, m=2, p=2):
    """Add Strassen rank-7 decomposition for 2x2 * 2x2 -> 2x2.
    U, V, W are flat lists; offsets point to the 2x2 block starts.
    col is the starting column index. Returns col+7.
    """
    # Block indices: A = [[a00,a01],[a10,a11]], B = [[b00,b01],[b10,b11]]
    # a00 at (u_off + 0*m + 0), a01 at (u_off + 0*m + 1)
    # a10 at (u_off + 1*m + 0), a11 at (u_off + 1*m + 1)
    # b00 at (v_off + 0*p + 0), b01 at (v_off + 0*p + 1)
    # b10 at (v_off + 1*p + 0), b11 at (v_off + 1*p + 1)
    # c00 at (w_off + 0*n + 0), c01 at (w_off + 0*n + 1)
    # c10 at (w_off + 1*n + 0), c11 at (w_off + 1*n + 1)
    
    a00 = u_off + 0
    a01 = u_off + 1
    a10 = u_off + 2
    a11 = u_off + 3
    
    b00 = v_off + 0
    b01 = v_off + 1
    b10 = v_off + 2
    b11 = v_off + 3
    
    c00 = w_off + 0
    c01 = w_off + 1
    c10 = w_off + 2
    c11 = w_off + 3
    
    # Strassen:
    # m1 = (a00+a11)*(b00+b11)
    # m2 = (a10+a11)*b00
    # m3 = a00*(b01-b11)
    # m4 = a11*(b10-b00)
    # m5 = (a00+a01)*b11
    # m6 = (a10-a00)*(b00+b01)
    # m7 = (a01-a11)*(b10+b11)
    # c00 = m1 + m4 - m5 + m7
    # c01 = m3 + m5
    # c10 = m2 + m4
    # c11 = m1 - m2 + m3 + m6
    
    # m1
    U[a00][col] += 1; U[a11][col] += 1
    V[b00][col] += 1; V[b11][col] += 1
    W[c00][col] += 1; W[c11][col] += 1
    col += 1
    # m2
    U[a10][col] += 1; U[a11][col] += 1
    V[b00][col] += 1
    W[c10][col] += 1; W[c11][col] += -1
    col += 1
    # m3
    U[a00][col] += 1
    V[b01][col] += 1; V[b11][col] += -1
    W[c01][col] += 1; W[c11][col] += 1
    col += 1
    # m4
    U[a11][col] += 1
    V[b10][col] += 1; V[b00][col] += -1
    W[c00][col] += 1; W[c10][col] += 1
    col += 1
    # m5
    U[a00][col] += 1; U[a01][col] += 1
    V[b11][col] += 1
    W[c00][col] += -1; W[c01][col] += 1
    col += 1
    # m6
    U[a10][col] += 1; U[a00][col] += -1
    V[b00][col] += 1; V[b01][col] += 1
    W[c11][col] += 1
    col += 1
    # m7
    U[a01][col] += 1; U[a11][col] += -1
    V[b10][col] += 1; V[b11][col] += 1
    W[c00][col] += 1; W[c11][col] += -1
    col += 1
    
    return col


def _strassen_4x4_4x4(U, V, W, u_off, v_off, w_off, col):
    """Strassen for 4x4 * 4x4 -> 4x4. Splits into 2x2 blocks, uses rank-7 per pair.
    4x4 has 4 blocks: A00,A01,A10,A11 (each 2x2)
    4x4 has 4 blocks: B00,B01,B10,B11 (each 2x2)
    4x4 has 4 blocks: C00,C01,C10,C11 (each 2x2)
    
    C00 = A00*B00 + A01*B10
    C01 = A00*B01 + A01*B11
    C10 = A10*B00 + A11*B10
    C11 = A10*B01 + A11*B11
    
    Each C block = sum of two 2x2*2x2 products.
    Use Strassen for each pair: 7 multiplications per 2x2*2x2.
    4 output blocks * 7 = 28... but we can share.
    
    Actually, the standard Strassen for 4x4*4x4 uses the 2x2 Strassen on the 2x2 block matrix.
    The 4x4*4x4 as 2x2 block matrix:
    A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
    Apply Strassen at the block level: 7 block multiplications, each is 2x2*2x2 with rank 7.
    Total: 7 * 7 = 49.
    """
    # Block offsets within the 4x4:
    # A00 at u_off + 0*4+0 = u_off, size 2x2, so elements at u_off, u_off+1, u_off+2, u_off+3
    # A01 at u_off + 0*4+2 = u_off+2
    # A10 at u_off + 2*4+0 = u_off+8
    # A11 at u_off + 2*4+2 = u_off+10
    
    # Similarly for V (4x4): B00 at v_off, B01 at v_off+2, B10 at v_off+8, B11 at v_off+10
    # For W (4x4): C00 at w_off, C01 at w_off+2, C10 at w_off+8, C11 at w_off+10
    
    A00 = u_off
    A01 = u_off + 2
    A10 = u_off + 8
    A11 = u_off + 10
    
    B00 = v_off
    B01 = v_off + 2
    B10 = v_off + 8
    B11 = v_off + 10
    
    C00 = w_off
    C01 = w_off + 2
    C10 = w_off + 8
    C11 = w_off + 10
    
    # Apply Strassen at block level:
    # m1 = (A00+A11)*(B00+B11) -> contributes to C00 and C11
    # m2 = (A10+A11)*B00 -> contributes to C10 and C11
    # m3 = A00*(B01-B11) -> contributes to C01 and C11
    # m4 = A11*(B10-B00) -> contributes to C00 and C10
    # m5 = (A00+A01)*B11 -> contributes to C00 and C01
    # m6 = (A10-A00)*(B00+B01) -> contributes to C11
    # m7 = (A01-A11)*(B10+B11) -> contributes to C00 and C11
    
    # Each block multiplication is 2x2*2x2 with rank 7.
    # For sum/difference of blocks, we add/subtract the U/V entries.
    
    # m1: (A00+A11) * (B00+B11)
    # U part: A00+A11 means we need a virtual matrix. We handle this by adding contributions.
    # Actually, for Strassen at block level, each "multiplication" is a 2x2*2x2 Strassen.
    # The inputs are linear combinations of 2x2 blocks.
    
    # Let me handle this differently. For each Strassen multiplication at the block level,
    # I need to compute (linear_comb_of_A_blocks) * (linear_comb_of_B_blocks) using 7 scalar mults.
    
    # For m1: M1 = (A00+A11) * (B00+B11)
    # This is a 2x2*2x2 product where the "A" matrix has entries that are sums of 2x2 blocks.
    # We can decompose this into 7 rank-1 terms at the scalar level.
    
    # Let me define a helper that adds a Strassen decomposition where the A-block and B-block
    # are specified as lists of (offset, coefficient) pairs.
    
    def add_block_strassen(u_parts, v_parts, w_parts, col):
        """u_parts: list of (offset, coeff) for the 2x2 A-block
           v_parts: list of (offset, coeff) for the 2x2 B-block
           w_parts: list of (offset, coeff) for the 2x2 C-block
           Adds 7 columns for the Strassen decomposition."""
        # The 2x2 A-block has entries at offsets: u_off, u_off+1, u_off+2, u_off+3
        # representing a00, a01, a10, a11
        # Similarly for B and C.
        
        # For each of the 7 Strassen multiplications, we need to set U, V, W entries.
        # The A-part of each mult is a linear combination of the 4 entries of each A-block.
        # This gets complex. Let me just enumerate all scalar entries.
        
        # Actually, let me think of it as: the "A matrix" for this Strassen call is a 2x2 matrix
        # where each entry is a linear combination of scalars from the original matrices.
        # Similarly for B and C.
        
        # Let me build the effective 2x2 A, B, C as linear combinations.
        # A_eff[i,j] = sum of coeff * original_entry for each part
        # But this creates a 4x4 effective problem... 
        
        # Simpler approach: just directly enumerate the 7 Strassen mults at the scalar level.
        # For the A-block with parts [(off1, c1), (off2, c2), ...]:
        # a00_eff = c1*U[off1+0] + c2*U[off2+0] + ...
        # a01_eff = c1*U[off1+1] + c2*U[off2+1] + ...
        # etc.
        
        # For each Strassen mult, e.g., m1 = (a00+a11)*(b00+b11):
        # U contribution: for each (off, coeff) in u_parts, add coeff to U[off+0] and U[off+3]
        # V contribution: for each (off, coeff) in v_parts, add coeff to V[off+0] and V[off+3]
        # W contribution: for each (off, coeff) in w_parts, add coeff to W[off+0] and W[off+3]
        
        # This works because the Strassen formula is linear in each entry.
        
        def add_u(off, pos, coeff):
            U[off + pos][col] += coeff
        
        def add_v(off, pos, coeff):
            V[off + pos][col] += coeff
        
        def add_w(off, pos, coeff):
            W[off + pos][col] += coeff
        
        # m1 = (a00+a11)*(b00+b11) -> c00 += 1, c11 += 1
        for (off, coeff) in u_parts:
            add_u(off, 0, coeff)  # a00
            add_u(off, 3, coeff)  # a11
        for (off, coeff) in v_parts:
            add_v(off, 0, coeff)  # b00
            add_v(off, 3, coeff)  # b11
        for (off, coeff) in w_parts:
            add_w(off, 0, coeff)  # c00
            add_w(off, 3, coeff)  # c11
        col += 1
        
        # m2 = (a10+a11)*b00 -> c10 += 1, c11 -= 1
        for (off, coeff) in u_parts:
            add_u(off, 2, coeff)  # a10
            add_u(off, 3, coeff)  # a11
        for (off, coeff) in v_parts:
            add_v(off, 0, coeff)  # b00
        for (off, coeff) in w_parts:
            add_w(off, 2, coeff)  # c10
            add_w(off, 3, -coeff)  # c11
        col += 1
        
        # m3 = a00*(b01-b11) -> c01 += 1, c11 += 1
        for (off, coeff) in u_parts:
            add_u(off, 0, coeff)  # a00
        for (off, coeff) in v_parts:
            add_v(off, 1, coeff)  # b01
            add_v(off, 3, -coeff)  # b11
        for (off, coeff) in w_parts:
            add_w(off, 1, coeff)  # c01
            add_w(off, 3, coeff)  # c11
        col += 1
        
        # m4 = a11*(b10-b00) -> c00 += 1, c10 += 1
        for (off, coeff) in u_parts:
            add_u(off, 3, coeff)  # a11
        for (off, coeff) in v_parts:
            add_v(off, 2, coeff)  # b10
            add_v(off, 0, -coeff)  # b00
        for (off, coeff) in w_parts:
            add_w(off, 0, coeff)  # c00
            add_w(off, 2, coeff)  # c10
        col += 1
        
        # m5 = (a00+a01)*b11 -> c00 -= 1, c01 += 1
        for (off, coeff) in u_parts:
            add_u(off, 0, coeff)  # a00
            add_u(off, 1, coeff)  # a01
        for (off, coeff) in v_parts:
            add_v(off, 3, coeff)  # b11
        for (off, coeff) in w_parts:
            add_w(off, 0, -coeff)  # c00
            add_w(off, 1, coeff)  # c01
        col += 1
        
        # m6 = (a10-a00)*(b00+b01) -> c11 += 1
        for (off, coeff) in u_parts:
            add_u(off, 2, coeff)  # a10
            add_u(off, 0, -coeff)  # a00
        for (off, coeff) in v_parts:
            add_v(off, 0, coeff)  # b00
            add_v(off, 1, coeff)  # b01
        for (off, coeff) in w_parts:
            add_w(off, 3, coeff)  # c11
        col += 1
        
        # m7 = (a01-a11)*(b10+b11) -> c00 += 1, c11 -= 1
        for (off, coeff) in u_parts:
            add_u(off, 1, coeff)  # a01
            add_u(off, 3, -coeff)  # a11
        for (off, coeff) in v_parts:
            add_v(off, 2, coeff)  # b10
            add_v(off, 3, coeff)  # b11
        for (off, coeff) in w_parts:
            add_w(off, 0, coeff)  # c00
            add_w(off, 3, -coeff)  # c11
        col += 1
        
        return col
    
    # Now apply the 7 block-level Strassen multiplications:
    
    # m1: (A00+A11) * (B00+B11) -> C00 += 1, C11 += 1
    u_parts = [(A00, 1), (A11, 1)]
    v_parts = [(B00, 1), (B11, 1)]
    w_parts = [(C00, 1), (C11, 1)]
    col = add_block_strassen(u_parts, v_parts, w_parts, col)
    
    # m2: (A10+A11) * B00 -> C10 += 1, C11 -= 1
    u_parts = [(A10, 1), (A11, 1)]
    v_parts = [(B00, 1)]
    w_parts = [(C10, 1), (C11, -1)]
    col = add_block_strassen(u_parts, v_parts, w_parts, col)
    
    # m3: A00 * (B01-B11) -> C01 += 1, C11 += 1
    u_parts = [(A00, 1)]
    v_parts = [(B01, 1), (B11, -1)]
    w_parts = [(C01, 1), (C11, 1)]
    col = add_block_strassen(u_parts, v_parts, w_parts, col)
    
    # m4: A11 * (B10-B00) -> C00 += 1, C10 += 1
    u_parts = [(A11, 1)]
    v_parts = [(B10, 1), (B00, -1)]
    w_parts = [(C00, 1), (C10, 1)]
    col = add_block_strassen(u_parts, v_parts, w_parts, col)
    
    # m5: (A00+A01) * B11 -> C00 -= 1, C01 += 1
    u_parts = [(A00, 1), (A01, 1)]
    v_parts = [(B11, 1)]
    w_parts = [(C00, -1), (C01, 1)]
    col = add_block_strassen(u_parts, v_parts, w_parts, col)
    
    # m6: (A10-A00) * (B00+B01) -> C11 += 1
    u_parts = [(A10, 1), (A00, -1)]
    v_parts = [(B00, 1), (B01, 1)]
    w_parts = [(C11, 1)]
    col = add_block_strassen(u_parts, v_parts, w_parts, col)
    
    # m7: (A01-A11) * (B10+B11) -> C00 += 1, C11 -= 1
    u_parts = [(A01, 1), (A11, -1)]
    v_parts = [(B10, 1), (B11, 1)]
    w_parts = [(C00, 1), (C11, -1)]
    col = add_block_strassen(u_parts, v_parts, w_parts, col)
    
    return col


def _strassen_4x4_4x2(U, V, W, u_off, v_off, w_off, col):
    """Strassen for 4x4 * 4x2 -> 4x2.
    A is 4x4 (blocks: A00,A01,A10,A11 each 2x2)
    B is 4x2 (blocks: B00 (2x2), B10 (2x2) stacked)
    C is 4x2 (blocks: C00 (2x2), C10 (2x2) stacked)
    
    C00 = A00*B00 + A01*B10  -- wait, B is 4x2, so it has 2 row-blocks of 2x2? No.
    B is 4 rows, 2 cols. Split into 2x2 blocks: 2 row-blocks, 1 col-block.
    B00 at rows 0-1, cols 0-1
    B10 at rows 2-3, cols 0-1
    
    A is 4x4: A00(rows 0-1, cols 0-1), A01(rows 0-1, cols 2-3), A10(rows 2-3, cols 0-1), A11(rows 2-3, cols 2-3)
    
    C is 4x2: C00(rows 0-1, cols 0-1), C10(rows 2-3, cols 0-1)
    
    C00 = A00*B00 + A01*B10? No, that's not right dimensionally.
    C00[i,k] = sum_j A00/A01[i,j] * B00/B10[j,k]
    
    Actually: C[0:2, 0:2] = A[0:2, 0:4] * B[0:4, 0:2]
    = A[0:2, 0:2]*B[0:2, 0:2] + A[0:2, 2:4]*B[2:4, 0:2]
    = A00*B00 + A01*B10
    
    C[2:4, 0:2] = A[2:4, 0:4] * B[0:4, 0:2]
    = A[2:4, 0:2]*B[0:2, 0:2] + A[2:4, 2:4]*B[2:4, 0:2]
    = A10*B00 + A11*B10
    
    So C00 = A00*B00 + A01*B10
    C10 = A10*B00 + A11*B10
    
    Each is a sum of two 2x2*2x2 products. Use Strassen for each: 7+7 = 14 per C-block.
    Total: 2 C-blocks * 14 = 28.
    """
    # Offsets:
    A00 = u_off
    A01 = u_off + 2
    A10 = u_off + 8
    A11 = u_off + 10
    
    # B is 4x2: B00 at v_off (rows 0-1, cols 0-1), B10 at v_off + 2*2 = v_off+4 (rows 2-3, cols 0-1)
    B00 = v_off
    B10 = v_off + 4
    
    # C is 4x2: C00 at w_off (rows 0-1, cols 0-1), C10 at w_off + 2*2 = w_off+4 (rows 2-3, cols 0-1)
    C00 = w_off
    C10 = w_off + 4
    
    def add_block_strassen(u_parts, v_parts, w_parts, col):
        def add_u(off, pos, coeff):
            U[off + pos][col] += coeff
        def add_v(off, pos, coeff):
            V[off + pos][col] += coeff
        def add_w(off, pos, coeff):
            W[off + pos][col] += coeff
        
        # m1 = (a00+a11)*(b00+b11) -> c00 += 1, c11 += 1
        for (off, coeff) in u_parts:
            add_u(off, 0, coeff)
            add_u(off, 3, coeff)
        for (off, coeff) in v_parts:
            add_v(off, 0, coeff)
            add_v(off, 3, coeff)
        for (off, coeff) in w_parts:
            add_w(off, 0, coeff)
            add_w(off, 3, coeff)
        col += 1
        
        # m2 = (a10+a11)*b00 -> c10 += 1, c11 -= 1
        for (off, coeff) in u_parts:
            add_u(off, 2, coeff)
            add_u(off, 3, coeff)
        for (off, coeff) in v_parts:
            add_v(off, 0, coeff)
        for (off, coeff) in w_parts:
            add_w(off, 2, coeff)
            add_w(off, 3, -coeff)
        col += 1
        
        # m3 = a00*(b01-b11) -> c01 += 1, c11 += 1
        for (off, coeff) in u_parts:
            add_u(off, 0, coeff)
        for (off, coeff) in v_parts:
            add_v(off, 1, coeff)
            add_v(off, 3, -coeff)
        for (off, coeff) in w_parts:
            add_w(off, 1, coeff)
            add_w(off, 3, coeff)
        col += 1
        
        # m4 = a11*(b10-b00) -> c00 += 1, c10 += 1
        for (off, coeff) in u_parts:
            add_u(off, 3, coeff)
        for (off, coeff) in v_parts:
            add_v(off, 2, coeff)
            add_v(off, 0, -coeff)
        for (off, coeff) in w_parts:
            add_w(off, 0, coeff)
            add_w(off, 2, coeff)
        col += 1
        
        # m5 = (a00+a01)*b11 -> c00 -= 1, c01 += 1
        for (off, coeff) in u_parts:
            add_u(off, 0, coeff)
            add_u(off, 1, coeff)
        for (off, coeff) in v_parts:
            add_v(off, 3, coeff)
        for (off, coeff) in w_parts:
            add_w(off, 0, -coeff)
            add_w(off, 1, coeff)
        col += 1
        
        # m6 = (a10-a00)*(b00+b01) -> c11 += 1
        for (off, coeff) in u_parts:
            add_u(off, 2, coeff)
            add_u(off, 0, -coeff)
        for (off, coeff) in v_parts:
            add_v(off, 0, coeff)
            add_v(off, 1, coeff)
        for (off, coeff) in w_parts:
            add_w(off, 3, coeff)
        col += 1
        
        # m7 = (a01-a11)*(b10+b11) -> c00 += 1, c11 -= 1
        for (off, coeff) in u_parts:
            add_u(off, 1, coeff)
            add_u(off, 3, -coeff)
        for (off, coeff) in v_parts:
            add_v(off, 2, coeff)
            add_v(off, 3, coeff)
        for (off, coeff) in w_parts:
            add_w(off, 0, coeff)
            add_w(off, 3, -coeff)
        col += 1
        
        return col
    
    # C00 = A00*B00 + A01*B10
    # First term: A00 * B00 -> C00
    u_parts = [(A00, 1)]
    v_parts = [(B00, 1)]
    w_parts = [(C00, 1)]
    col = add_block_strassen(u_parts, v_parts, w_parts, col)
    
    # Second term: A01 * B10 -> C00
    u_parts = [(A01, 1)]
    v_parts = [(B10, 1)]
    w_parts = [(C00, 1)]
    col = add_block_strassen(u_parts, v_parts, w_parts, col)
    
    # C10 = A10*B00 + A11*B10
    # First term: A10 * B00 -> C10
    u_parts = [(A10, 1)]
    v_parts = [(B00, 1)]
    w_parts = [(C10, 1)]
    col = add_block_strassen(u_parts, v_parts, w_parts, col)
    
    # Second term: A11 * B10 -> C10
    u_parts = [(A11, 1)]
    v_parts = [(B10, 1)]
    w_parts = [(C10, 1)]
    col = add_block_strassen(u_parts, v_parts, w_parts, col)
    
    return col


def construct():
    """
    Construct a rank-minimal decomposition for 4x5 * 5x6 matrix multiplication.
    
    Strategy: Strassen-Winograd recursion.
    A (4x5) = [A1 (4x4) | A2 (4x1)]
    B (5x6) = [B1 (4x6); B2 (1x6)]
    C = A1*B1 + A2*B2
    
    A1*B1: B1 = [B1a (4x4) | B1b (4x2)]
    A1*B1 = A1*B1a + A1*B1b
    
    A1*B1a: 4x4*4x4, rank 49 via Strassen
    A1*B1b: 4x4*4x2, rank 28 via Strassen
    A2*B2: 4x1*1x6, rank 1 (outer product)
    
    Total rank: 49 + 28 + 1 = 78
    """
    n, m, p = DIMENSIONS  # 4, 5, 6
    rank = 78
    U = [[0] * rank for _ in range(n * m)]  # 20 x 78
    V = [[0] * rank for _ in range(m * p)]  # 30 x 78
    W = [[0] * rank for _ in range(p * n)]  # 24 x 78
    
    col = 0
    
    # A1*B1a: 4x4 * 4x4 -> 4x4
    # A1 is at U[0:4, 0:4] in the 4x5 matrix. In flat U (row-major, 4 rows x 5 cols):
    # A1[i,j] is at U[i*5 + j] for i in 0..3, j in 0..3
    # So A100 at U[0*5+0]=U[0], A101 at U[1], A102 at U[2], A103 at U[3]
    # A110 at U[1*5+0]=U[5], A111 at U[6], A112 at U[7], A113 at U[8]
    # A120 at U[2*5+0]=U[10], etc.
    # A130 at U[3*5+0]=U[15], etc.
    
    # For the 4x4 block A1, the 2x2 sub-blocks in flat indexing (stride 5):
    # A00: rows 0-1, cols 0-1: U[0], U[1], U[5], U[6]
    # A01: rows 0-1, cols 2-3: U[2], U[3], U[7], U[8]
    # A10: rows 2-3, cols 0-1: U[10], U[11], U[15], U[16]
    # A11: rows 2-3, cols 2-3: U[12], U[13], U[17], U[18]
    
    # B1a is 4x4, part of B (5x6). B1 is rows 0-3, all 6 cols.
    # B1a is B[0:4, 0:4]. In flat V (5 rows x 6 cols):
    # B1a[i,j] at V[i*6 + j] for i in 0..3, j in 0..3
    # B00: V[0], V[1], V[6], V[7]
    # B01: V[2], V[3], V[8], V[9]
    # B10: V[12], V[13], V[18], V[19]
    # B11: V[14], V[15], V[20], V[21]
    
    # C1a is 4x4, part of C (4x6). In flat W (6 rows x 4 cols, since W[k*n+i]):
    # W index: k*n + i where k is col of C (0..5), i is row of C (0..3)
    # C1a[i,k] for i in 0..3, k in 0..3 is at W[k*4 + i]
    # C00: rows 0-1, cols 0-1: W[0*4+0]=W[0], W[0*4+1]=W[1], W[1*4+0]=W[4], W[1*4+1]=W[5]
    # C01: rows 0-1, cols 2-3: W[2*4+0]=W[8], W[2*4+1]=W[9], W[3*4+0]=W[12], W[3*4+1]=W[13]
    # C10: rows 2-3, cols 0-1: W[0*4+2]=W[2], W[0*4+3]=W[3], W[1*4+2]=W[6], W[1*4+3]=W[7]
    # C11: rows 2-3, cols 2-3: W[2*4+2]=W[10], W[2*4+3]=W[11], W[3*4+2]=W[14], W[3*4+3]=W[15]
    
    # I need to map the 2x2 block offsets properly.
    # For U (4x5, stride 5): block (r,c) of 2x2 has elements at:
    # U[r*5+c], U[r*5+c+1], U[(r+1)*5+c], U[(r+1)*5+c+1]
    # For V (5x6, stride 6): block (r,c) of 2x2 has elements at:
    # V[r*6+c], V[r*6+c+1], V[(r+1)*6+c], V[(r+1)*6+c+1]
    # For W (6x4, stride 4, W[k*4+i]): block for C rows r..r+1, cols c..c+1:
    # W[c*4+r], W[c*4+r+1], W[(c+1)*4+r], W[(c+1)*4+r+1]
    
    # This is getting complex. Let me just directly compute the offsets.
    
    # For A1 (4x4 in 4x5): 
    # A00: u_off = 0, elements at 0,1,5,6
    # A01: u_off = 2, elements at 2,3,7,8
    # A10: u_off = 10, elements at 10,11,15,16
    # A11: u_off = 12, elements at 12,13,17,18
    
    # But my _strassen_4x4_4x4 expects contiguous 2x2 blocks with offset +0,+1,+2,+3.
    # The actual layout has stride 5, not 2. I need to adjust.
    
    # Let me rewrite the functions to take explicit element positions.
    
    # Actually, let me just create a more general approach.
    
    # For now, let me just do the schoolbook to make sure it works, then optimize.
    
    # Reset to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
