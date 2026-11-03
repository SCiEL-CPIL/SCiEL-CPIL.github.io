"""Program for finding rank-minimal exact tensor decomposition for 2x3x8 matrix multiplication."""

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Decompose the 2x3x8 matrix multiplication tensor.
    
    Strategy:
    1. Split B (3x8) into two 3x4 blocks: B1 (cols 0-3), B2 (cols 4-7).
    2. C = A * B = A * B1 + A * B2.
    3. Decompose each 2x3x4 multiplication.
    
    For 2x3x4:
    - Split A (2x3) into A_left (2x2) and A_right (2x1).
    - Split B (3x4) into B_top (2x4) and B_bottom (1x4).
    - C = A_left * B_top + A_right * B_bottom.
    
    For A_left * B_top (2x2 * 2x4):
    - Split B_top into two 2x2 blocks: B_tl, B_tr.
    - Use Strassen (rank 7) for A_left * B_tl.
    - Use Strassen (rank 7) for A_left * B_tr.
    - Total rank: 14.
    
    For A_right * B_bottom (2x1 * 1x4):
    - Schoolbook (rank 8).
    
    Total rank per 2x3x4 block: 14 + 8 = 22.
    Total rank for 2x3x8: 2 * 22 = 44.
    """
    n, m, p = 2, 3, 8
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, coeff=1):
        U[u_idx][rank - 1] = coeff
        V[v_idx][rank - 1] = coeff
        W[w_idx][rank - 1] = coeff
    
    # We will fill from index 0 to rank-1
    # Let's use a pointer
    ptr = 0
    
    # Process two blocks of B (cols 0-3 and 4-7)
    for block in range(2):
        # B block columns are 4*block to 4*block+3
        # A is same for both blocks (indices 0..5)
        
        # 1. A_left * B_top (2x2 * 2x4) -> 2x4 result
        # A_left indices: A[0,0], A[1,0], A[0,1], A[1,1] -> U indices: 0, 3, 1, 4
        # B_top indices: B[0, k], B[1, k] for k in 0..3 (relative to block)
        # Global B indices: B[0, 4*block + k], B[1, 4*block + k]
        # V indices: 0*8 + (4*block + k) = 4*block + k, 1*8 + (4*block + k) = 8 + 4*block + k
        # C result indices: C[k, 0], C[k, 1] for k in 0..3 (relative to block)
        # Global C indices: C[4*block + k, 0], C[4*block + k, 1]
        # W indices: (4*block + k)*2 + 0, (4*block + k)*2 + 1
        
        # Sub-block 1: B_top left (cols 0,1 of the 4)
        # B_tl: B[0, 4*block], B[1, 4*block], B[0, 4*block+1], B[1, 4*block+1]
        # V indices: v00 = 4*block, v10 = 8+4*block, v01 = 4*block+1, v11 = 8+4*block+1
        # C_tl: C[4*block, 0], C[4*block+1, 0], C[4*block, 1], C[4*block+1, 1]
        # W indices: w00 = (4*block)*2 + 0, w10 = (4*block+1)*2 + 0, w01 = (4*block)*2 + 1, w11 = (4*block+1)*2 + 1
        
        # Strassen for 2x2 * 2x2
        # A00 = A[0,0] (U0), A10 = A[1,0] (U3), A01 = A[0,1] (U1), A11 = A[1,1] (U4)
        # B00 = B[0, 4*block] (V 4*block), B10 = B[1, 4*block] (V 8+4*block)
        # B01 = B[0, 4*block+1] (V 4*block+1), B11 = B[1, 4*block+1] (V 8+4*block+1)
        
        # M1 = (A00 + A11) * (B00 + B11)
        # U: A00, A11. V: B00, B11. W: C00, C11
        # C00 += M1, C11 += M1
        # We need to add terms for each coefficient.
        
        def add_strassen_block(a00, a10, a01, a11, b00, b10, b01, b11, c00, c10, c01, c11):
            nonlocal ptr
            # M1
            # (A00 + A11) * (B00 + B11)
            # A00*B00, A00*B11, A11*B00, A11*B11
            # C00 += ..., C11 += ...
            # This is 4 multiplications for M1? No, M1 is one "multiplication" in the bilinear form sense?
            # No, the rank counts the number of bilinear terms.
            # M1 = (A00 + A11)(B00 + B11) is 1 term in the decomposition of the tensor?
            # Yes, if we treat the pre-combinations as part of the U/V vectors.
            # U vector has 1 at A00 and 1 at A11. V vector has 1 at B00 and 1 at B11.
            # W vector has 1 at C00 and 1 at C11.
            # So 1 rank unit.
            
            # M1
            U[a00][ptr] = 1; U[a11][ptr] = 1
            V[b00][ptr] = 1; V[b11][ptr] = 1
            W[c00][ptr] = 1; W[c11][ptr] = 1
            ptr += 1
            
            # M2 = (A10 + A11) * B00
            U[a10][ptr] = 1; U[a11][ptr] = 1
            V[b00][ptr] = 1
            W[c10][ptr] = 1; W[c11][ptr] = -1
            ptr += 1
            
            # M3 = A00 * (B10 - B11)
            U[a00][ptr] = 1
            V[b10][ptr] = 1; V[b11][ptr] = -1
            W[c01][ptr] = 1; W[c11][ptr] = 1
            ptr += 1
            
            # M4 = A11 * (B01 - B00)
            U[a11][ptr] = 1
            V[b01][ptr] = 1; V[b00][ptr] = -1
            W[c00][ptr] = 1; W[c10][ptr] = 1
            ptr += 1
            
            # M5 = (A00 + A01) * B11
            U[a00][ptr] = 1; U[a01][ptr] = 1
            V[b11][ptr] = 1
            W[c00][ptr] = -1; W[c01][ptr] = 1
            ptr += 1
            
            # M6 = (A10 - A00) * (B00 + B01)
            U[a10][ptr] = 1; U[a00][ptr] = -1
            V[b00][ptr] = 1; V[b01][ptr] = 1
            W[c11][ptr] = 1
            ptr += 1
            
            # M7 = (A01 - A11) * (B10 + B11)
            U[a01][ptr] = 1; U[a11][ptr] = -1
            V[b10][ptr] = 1; V[b11][ptr] = 1
            W[c00][ptr] = 1; W[c10][ptr] = 1
            ptr += 1
            
        # Block 1 indices
        a00, a10, a01, a11 = 0, 3, 1, 4
        b00 = 4*block; b10 = 8+4*block; b01 = 4*block+1; b11 = 8+4*block+1
        c00 = (4*block)*2 + 0; c10 = (4*block+1)*2 + 0; c01 = (4*block)*2 + 1; c11 = (4*block+1)*2 + 1
        add_strassen_block(a00, a10, a01, a11, b00, b10, b01, b11, c00, c10, c01, c11)
        
        # Sub-block 2: B_top right (cols 2,3 of the 4)
        # B_tr: B[0, 4*block+2], B[1, 4*block+2], B[0, 4*block+3], B[1, 4*block+3]
        # V indices: v00 = 4*block+2, v10 = 8+4*block+2, v01 = 4*block+3, v11 = 8+4*block+3
        # C_tr: C[4*block+2, 0], C[4*block+3, 0], C[4*block+2, 1], C[4*block+3, 1]
        # W indices: w00 = (4*block+2)*2 + 0, w10 = (4*block+3)*2 + 0, w01 = (4*block+2)*2 + 1, w11 = (4*block+3)*2 + 1
        
        b00 = 4*block+2; b10 = 8+4*block+2; b01 = 4*block+3; b11 = 8+4*block+3
        c00 = (4*block+2)*2 + 0; c10 = (4*block+3)*2 + 0; c01 = (4*block+2)*2 + 1; c11 = (4*block+3)*2 + 1
        add_strassen_block(a00, a10, a01, a11, b00, b10, b01, b11, c00, c10, c01, c11)
        
        # 2. A_right * B_bottom (2x1 * 1x4) -> 2x4 result
        # A_right: A[0,2], A[1,2] -> U indices: 2, 5
        # B_bottom: B[2, k] for k in 0..3 (relative to block)
        # Global B indices: B[2, 4*block + k]
        # V indices: 2*8 + (4*block + k) = 16 + 4*block + k
        # C result: C[4*block+k, 0] += A[0,2]*B[2, 4*block+k]
        #            C[4*block+k, 1] += A[1,2]*B[2, 4*block+k]
        # W indices: (4*block+k)*2 + 0, (4*block+k)*2 + 1
        
        for k in range(4):
            # Term 1: A[0,2] * B[2, 4*block+k] -> C[4*block+k, 0]
            U[2][ptr] = 1
            V[16 + 4*block + k][ptr] = 1
            W[(4*block+k)*2 + 0][ptr] = 1
            ptr += 1
            
            # Term 2: A[1,2] * B[2, 4*block+k] -> C[4*block+k, 1]
            U[5][ptr] = 1
            V[16 + 4*block + k][ptr] = 1
            W[(4*block+k)*2 + 1][ptr] = 1
            ptr += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
