"""Matrix multiplication 2x3x10 rank-minimal tensor decomposition."""
import time

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 10
    # Decompose p=10 into 5 blocks of size 2
    num_blocks = p // 2  # 5 blocks
    
    # Strassen decomposition for 2x2x2 (indices i,k in {0,1}, j in {0,1})
    # Standard Strassen: 7 products
    # S1 = (A00+A11)(B00+B11)
    # S2 = (A10+A11)B00
    # S3 = A00(B10-B11)
    # S4 = A11(B01-B00)
    # S5 = (A00+A01)B11
    # S6 = (A10-A00)(B00+B10)
    # S7 = (A01-A11)(B01+B11)
    # C00 = S1 + S4 - S5 + S7
    # C01 = S3 + S5
    # C10 = S2 + S4
    # C11 = S1 - S2 + S3 + S6
    
    # For 2x3x2: A is 2x3, B is 3x2, C is 2x2
    # Middle index j=0,1,2
    # Use Strassen on j=0,1 (treating as 2x2x2 with A' = A[:,0:2], B' = B[0:2,:])
    # Then add 4 terms for j=2
    # Total rank per block = 7 + 4 = 11
    
    rank = num_blocks * 11
    U = [[0] * rank for _ in range(n * m)]  # 6 x rank
    V = [[0] * rank for _ in range(m * p)]  # 30 x rank
    W = [[0] * rank for _ in range(p * n)]  # 20 x rank
    
    # Strassen coefficients for 2x2x2
    # Each term: (U_factor, V_factor, W_factor)
    # U indexes: i*m + j where i in {0,1}, j is middle index 0 or 1
    # V indexes: j*p + k where j is middle, k in {0,1}
    # W indexes: k*n + i where k in {0,1}, i in {0,1}
    
    # Strassen 2x2x2 terms (for middle indices 0,1):
    # Term structure: (u_coeff_by_row, v_coeff_by_col, w_coeff_by_pos)
    # We need to express each rank-1 term as products
    
    # Let me define the 7 Strassen terms explicitly
    # For 2x2x2 with A[i,j], B[j,k], C[i,k]:
    # i in {0,1}, j in {0,1}, k in {0,1}
    
    strassen_terms = []
    # S1 = (A00+A11)(B00+B11) -> C00 += 1, C11 += 1
    # U: row0(i=0,j=0)+1, row1(i=1,j=1)+1 => U[0]=1, U[3]=1
    # V: col0(j=0,k=0)+1, col1(j=0,k=1)+1 => V[0]=1, V[1]=1
    # Wait, V index is j*p+k = j*2+k for p=2
    # V[0*2+0]=V[0], V[0*2+1]=V[1] -> these are B[0,0] and B[0,1]
    # Hmm, I need to be more careful.
    
    # Let me redefine. For the sub-problem (2x2x2):
    # A_sub[i][j] for i in 0,1 and j in 0,1
    # B_sub[j][k] for j in 0,1 and k in 0,1
    # C_sub[i][k] for i in 0,1 and k in 0,1
    #
    # In the full tensor:
    # U index for A_sub[i][j] is i*m + j = i*3 + j (but j is 0 or 1 in sub)
    # Actually in the block, middle index 0 and 1 map to j=0 and j=1
    # U[i*3 + j] for the block
    #
    # V index for B_sub[j][k] is j*2 + k
    # W index for C_sub[i][k] is k*2 + i
    
    # Let me just directly construct the 11 terms for a 2x3x2 block
    
    # I'll define each term as a list of (row_idx, col_idx) for U, V, W
    # with coefficient 1 (we'll use coefficients if needed)
    
    def add_term(u_entries, v_entries, w_entries, col_offset):
        """Add a rank-1 term. u_entries: dict {row: coeff}, etc."""
        for row, coeff in u_entries.items():
            U[row][col_offset] += coeff
        for row, coeff in v_entries.items():
            V[row][col_offset] += coeff
        for row, coeff in w_entries.items():
            W[row][col_offset] += coeff
    
    for block in range(num_blocks):
        base_col = block * 11
        # Middle indices in this block: j=0,1,2
        # k indices: 0,1
        # i indices: 0,1
        
        # U row indices (i*3 + j): 
        # i=0,j=0 -> 0; i=0,j=1 -> 1; i=0,j=2 -> 2
        # i=1,j=0 -> 3; i=1,j=1 -> 4; i=1,j=2 -> 5
        
        # V row indices (j*2 + k) for this block's k=0,1:
        # j=0,k=0 -> block*2+0; j=0,k=1 -> block*2+1
        # j=1,k=0 -> block*2+2; j=1,k=1 -> block*2+3
        # j=2,k=0 -> block*2+4; j=2,k=1 -> block*2+5
        
        # W row indices (k*2 + i):
        # k=0,i=0 -> block*2+0; k=0,i=1 -> block*2+1
        # k=1,i=0 -> block*2+2; k=1,i=1 -> block*2+3
        
        # Strassen terms for middle j=0,1:
        # S1: (A[0,0]+A[1,1])*(B[0,0]+B[0,1]) -> contributes to C[0,0] and C[1,1]
        # Actually let me use the standard Strassen formula.
        # For 2x2x2: C00=S1+S4-S5+S7, C01=S3+S5, C10=S2+S4, C11=S1-S2+S3+S6
        
        # S1 = (A00+A11)*(B00+B11)
        # A00 = U[0], A11 = U[4] (i=1,j=1)
        # B00 = V[block*2+0], B11 = V[block*2+3] (j=1,k=1)
        # C contribution: C00 += 1, C11 += 1
        # C00 = W[block*2+0], C11 = W[block*2+3]
        add_term({0: 1, 4: 1}, {block*2+0: 1, block*2+3: 1}, {block*2+0: 1, block*2+3: 1}, base_col)
        
        # S2 = (A10+A11)*B00
        # A10 = U[3], A11 = U[4]
        # B00 = V[block*2+0]
        # C10 += 1 => W[block*2+1] += 1
        add_term({3: 1, 4: 1}, {block*2+0: 1}, {block*2+1: 1}, base_col+1)
        
        # S3 = A00*(B10-B11)
        # A00 = U[0]
        # B10 = V[block*2+2], B11 = V[block*2+3]
        # C01 += 1, C11 += -1 => W[block*2+2] += 1, W[block*2+3] += -1
        add_term({0: 1}, {block*2+2: 1, block*2+3: -1}, {block*2+2: 1, block*2+3: -1}, base_col+2)
        
        # S4 = A11*(B01-B00)
        # A11 = U[4]
        # B01 = V[block*2+1], B00 = V[block*2+0]
        # C00 += 1, C10 += 1 => W[block*2+0] += 1, W[block*2+1] += 1
        add_term({4: 1}, {block*2+1: 1, block*2+0: -1}, {block*2+0: 1, block*2+1: 1}, base_col+3)
        
        # S5 = (A00+A01)*B11
        # A00 = U[0], A01 = U[1]
        # B11 = V[block*2+3]
        # C01 += 1, C00 += -1 => W[block*2+2] += 1, W[block*2+0] += -1
        add_term({0: 1, 1: 1}, {block*2+3: 1}, {block*2+2: 1, block*2+0: -1}, base_col+4)
        
        # S6 = (A10-A00)*(B00+B10)
        # A10 = U[3], A00 = U[0]
        # B00 = V[block*2+0], B10 = V[block*2+2]
        # C11 += 1 => W[block*2+3] += 1
        add_term({3: 1, 0: -1}, {block*2+0: 1, block*2+2: 1}, {block*2+3: 1}, base_col+5)
        
        # S7 = (A01-A11)*(B01+B11)
        # A01 = U[1], A11 = U[4]
        # B01 = V[block*2+1], B11 = V[block*2+3]
        # C00 += 1 => W[block*2+0] += 1
        add_term({1: 1, 4: -1}, {block*2+1: 1, block*2+3: 1}, {block*2+0: 1}, base_col+6)
        
        # Now 4 terms for middle j=2:
        # C[0,0] += A[0,2]*B[2,0]
        add_term({2: 1}, {block*2+4: 1}, {block*2+0: 1}, base_col+7)
        # C[0,1] += A[0,2]*B[2,1]
        add_term({2: 1}, {block*2+5: 1}, {block*2+2: 1}, base_col+8)
        # C[1,0] += A[1,2]*B[2,0]
        add_term({5: 1}, {block*2+4: 1}, {block*2+1: 1}, base_col+9)
        # C[1,1] += A[1,2]*B[2,1]
        add_term({5: 1}, {block*2+5: 1}, {block*2+3: 1}, base_col+10)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
