"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 15 decomposition
    # 14 terms for 2x2x4 part (A[:,0:2] * B[0:2,:])
    # 1 term for 2x1x4 part (A[:,2] * B[2,:])
    
    rank = 15
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add Strassen term for 2x2x2 block
    # local i, j in 0..1
    # local k, l in 0..1 (k for B row, l for B col? No, B is j x k)
    # B indices: j (row) in 0..1, k (col) in offset..offset+1
    # A indices: i (row) in 0..1, j (col) in 0..1
    # C indices: i (row) in 0..1, k (col) in offset..offset+1
    # W index: k_global * n + i_global
    
    def add_strassen(block_col_offset, term_col):
        # A rows: i=0,1. Cols: j=0,1
        # B rows: j=0,1. Cols: k=block_col_offset, block_col_offset+1
        # C rows: i=0,1. Cols: k=block_col_offset, block_col_offset+1
        
        # M1 = (A00+A10)(B00+B11)
        U[0][term_col] += 1; U[4][term_col] += 1  # A00, A10 (idx i*m+j: 0*3+0=0, 1*3+0=3? No. U idx is i*m+j. i=0,j=0->0. i=1,j=0->3. Correct.)
        # Wait, U index is i*m + j.
        # i=0, j=0 -> 0
        # i=0, j=1 -> 1
        # i=1, j=0 -> 3
        # i=1, j=1 -> 4
        
        # B indices: j*p + k.
        # j=0, k=offset -> 0*4+offset = offset
        # j=0, k=offset+1 -> offset+1
        # j=1, k=offset -> 1*4+offset = 4+offset
        # j=1, k=offset+1 -> 5+offset
        
        V[offset][term_col] += 1      # B0, offset (j=0, k=off) -> B00
        V[4 + offset + 1][term_col] += 1 # B1, offset+1 (j=1, k=off+1) -> B11
        
        # W indices: k*p? No, k*n + i.
        # k=offset, i=0 -> offset*2+0
        # k=offset, i=1 -> offset*2+1
        # k=offset+1, i=0 -> (offset+1)*2+0
        # k=offset+1, i=1 -> (offset+1)*2+1
        
        W[offset * 2 + 0][term_col] += 1  # C0, off
        W[offset * 2 + 1][term_col] += 1  # C1, off
        W[(offset + 1) * 2 + 0][term_col] += 1 # C0, off+1
        W[(offset + 1) * 2 + 1][term_col] += 1 # C1, off+1
        
        # M2 = (A10+A11)B00
        U[3][term_col] += 1; U[4][term_col] += 1
        V[offset][term_col] += 1
        W[offset * 2 + 1][term_col] += 1 # C1, off
        W[(offset + 1) * 2 + 1][term_col] -= 1 # C1, off+1
        
        # M3 = A00(B01-B11)
        U[0][term_col] += 1
        V[offset + 1][term_col] += 1
        V[4 + offset + 1][term_col] -= 1
        W[offset * 2 + 0][term_col] += 1 # C0, off
        W[(offset + 1) * 2 + 0][term_col] += 1 # C0, off+1
        
        # M4 = A11(B10-B00)
        U[4][term_col] += 1
        V[4 + offset][term_col] += 1
        V[offset][term_col] -= 1
        W[offset * 2 + 0][term_col] += 1 # C0, off
        W[offset * 2 + 1][term_col] += 1 # C1, off
        
        # M5 = (A00+A01)B11
        U[0][term_col] += 1; U[1][term_col] += 1
        V[4 + offset + 1][term_col] += 1
        W[offset * 2 + 0][term_col] -= 1 # C0, off
        W[(offset + 1) * 2 + 0][term_col] += 1 # C0, off+1
        
        # M6 = (A10-A00)(B00+B01)
        U[3][term_col] += 1; U[0][term_col] -= 1
        V[offset][term_col] += 1; V[offset + 1][term_col] += 1
        W[(offset + 1) * 2 + 1][term_col] += 1 # C1, off+1
        
        # M7 = (A01-A11)(B10+B11)
        U[1][term_col] += 1; U[4][term_col] -= 1
        V[4 + offset][term_col] += 1; V[4 + offset + 1][term_col] += 1
        W[offset * 2 + 0][term_col] += 1 # C0, off
    
    # Block 1: k=0,1
    for t in range(7):
        add_strassen(0, t)
        
    # Block 2: k=2,3
    for t in range(7):
        add_strassen(2, 7 + t)
        
    # Term 14: Outer product A[:,2] * B[2,:]
    col = 14
    # U: i=0,1; j=2
    U[2][col] = 1 # i=0, j=2 -> 0*3+2=2
    U[5][col] = 1 # i=1, j=2 -> 1*3+2=5
    
    # V: j=2; k=0..3
    V[2 * 4 + 0][col] = 1
    V[2 * 4 + 1][col] = 1
    V[2 * 4 + 2][col] = 1
    V[2 * 4 + 3][col] = 1
    
    # W: k=0..3; i=0,1
    for k in range(4):
        W[k * 2 + 0][col] = 1
        W[k * 2 + 1][col] = 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
