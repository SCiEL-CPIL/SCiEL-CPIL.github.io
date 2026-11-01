"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-18 tensor decomposition for multiplying a 2x2 matrix by a 2x5 matrix.
    This uses two independent Strassen decompositions (rank 7 each) for the first 4 columns
    of the second matrix, and a standard decomposition (rank 4) for the 5th column.
    Total rank = 7 + 7 + 4 = 18.
    """
    n, m, p = 2, 2, 5
    rank = 18

    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    # Helper to add a Strassen block for 2x2x2
    # offset: starting column index in U, V, W
    def add_strassen(offset):
        # Strassen's 7 products for 2x2 matrix multiplication
        # M1 = (A11)(B11+B22)
        # M2 = (A11+A22)(B22)
        # M3 = (A22)(B11+B12)
        # M4 = (A11)(B22-B21)
        # M5 = (A11+A22)(B11)
        # M6 = (A22-A11)(B12+B22)
        # M7 = (A11-A22)(B21+B11)
        
        # U rows: 0:(1,1), 1:(1,2), 2:(2,1), 3:(2,2)
        # V rows: 0:(1,1), 1:(1,2), 2:(2,1), 3:(2,2)
        # W rows: 0:(1,1), 1:(1,2), 2:(2,1), 3:(2,2)
        
        # Column offset in U, V, W
        base = offset
        
        # M1
        U[0][base] = 1; U[1][base] = 0; U[2][base] = 0; U[3][base] = 0
        V[0][base] = 1; V[1][base] = 0; V[2][base] = 1; V[3][base] = 1
        W[0][base] = 1; W[1][base] = 0; W[2][base] = 0; W[3][base] = -1
        
        # M2
        U[0][base+1] = 1; U[1][base+1] = 0; U[2][base+1] = 1; U[3][base+1] = 0
        V[0][base+1] = 0; V[1][base+1] = 0; V[2][base+1] = 0; V[3][base+1] = 1
        W[0][base+1] = 0; W[1][base+1] = 0; W[2][base+1] = 0; W[3][base+1] = 1
        
        # M3
        U[0][base+2] = 0; U[1][base+2] = 0; U[2][base+2] = 0; U[3][base+2] = 1
        V[0][base+2] = 1; V[1][base+2] = 1; V[2][base+2] = 0; V[3][base+2] = 0
        W[0][base+2] = 0; W[1][base+2] = 0; W[2][base+2] = 1; W[3][base+2] = 0
        
        # M4
        U[0][base+3] = 1; U[1][base+3] = 0; U[2][base+3] = 0; U[3][base+3] = 0
        V[0][base+3] = 0; V[1][base+3] = 0; V[2][base+3] = -1; V[3][base+3] = 1
        W[0][base+3] = 0; W[1][base+3] = 0; W[2][base+3] = 0; W[3][base+3] = 1
        
        # M5
        U[0][base+4] = 1; U[1][base+4] = 0; U[2][base+4] = 1; U[3][base+4] = 0
        V[0][base+4] = 1; V[1][base+4] = 0; V[2][base+4] = 0; V[3][base+4] = 0
        W[0][base+4] = 1; W[1][base+4] = 0; W[2][base+4] = 0; W[3][base+4] = 0
        
        # M6
        U[0][base+5] = -1; U[1][base+5] = 0; U[2][base+5] = 1; U[3][base+5] = 0
        V[0][base+5] = 0; V[1][base+5] = 1; V[2][base+5] = 1; V[3][base+5] = 1
        W[0][base+5] = 0; W[1][base+5] = 0; W[2][base+5] = 0; W[3][base+5] = 1
        
        # M7
        U[0][base+6] = 1; U[1][base+6] = 0; U[2][base+6] = -1; U[3][base+6] = 0
        V[0][base+6] = 1; V[1][base+6] = 0; V[2][base+6] = 1; V[3][base+6] = 0
        W[0][base+6] = 0; W[1][base+6] = 1; W[2][base+6] = 0; W[3][base+6] = 0

    # Standard decomposition for 2x2x1 (rank 4)
    def add_standard(offset, col_idx):
        # For column col_idx of B, we compute 2x2 * 2x1
        # Result is 2x1.
        # C11 = A11*B11 + A12*B21
        # C21 = A21*B11 + A22*B21
        base = offset
        
        # Product 1: A11 * B1
        U[0][base] = 1; U[1][base] = 0; U[2][base] = 0; U[3][base] = 0
        V[0][base] = 1; V[1][base] = 0; V[2][base] = 0; V[3][base] = 0
        W[col_idx*2+0][base] = 1; W[col_idx*2+1][base] = 0 # W index: k*n+i. k is col in B. i is row in C.
        # Wait, W index is k*n+i. Here k is the column index of B.
        # B is 2x5. So k is 0..4.
        # C is 2x5. So row i is 0..1.
        # W[k*n+i].
        # For col 0: W[0*2+0]=W[0], W[0*2+1]=W[1].
        # For col 1: W[1*2+0]=W[2], W[1*2+1]=W[3].
        
        # Correction: V rows are j*p+k. j is row in B (0,1). k is col in B.
        # V[j*p+k].
        # For standard on col_idx:
        # V[0*2+col_idx] = B[0, col_idx]
        # V[1*2+col_idx] = B[1, col_idx]
        
        # Let's re-implement add_standard carefully.
        # U is 2x2 (A). V is 2x1 (col of B). W is 1x2 (col of C).
        # Actually W is p*n = 5*2 = 10 rows.
        # W[k*n+i]. k is col of C/B, i is row of C.
        
        # P1: A11 * B1
        U[0][base] = 1
        U[1][base] = 0
        U[2][base] = 0
        U[3][base] = 0
        
        V[0*2+col_idx][base] = 1
        V[1*2+col_idx][base] = 0
        
        W[col_idx*2+0][base] = 1
        W[col_idx*2+1][base] = 0
        
        # P2: A12 * B2
        U[0][base+1] = 0
        U[1][base+1] = 1
        U[2][base+1] = 0
        U[3][base+1] = 0
        
        V[0*2+col_idx][base+1] = 0
        V[1*2+col_idx][base+1] = 1
        
        W[col_idx*2+0][base+1] = 1
        W[col_idx*2+1][base+1] = 0
        
        # P3: A21 * B1
        U[0][base+2] = 0
        U[1][base+2] = 0
        U[2][base+2] = 1
        U[3][base+2] = 0
        
        V[0*2+col_idx][base+2] = 1
        V[1*2+col_idx][base+2] = 0
        
        W[col_idx*2+0][base+2] = 0
        W[col_idx*2+1][base+2] = 1
        
        # P4: A22 * B2
        U[0][base+3] = 0
        U[1][base+3] = 0
        U[2][base+3] = 0
        U[3][base+3] = 1
        
        V[0*2+col_idx][base+3] = 0
        V[1*2+col_idx][base+3] = 1
        
        W[col_idx*2+0][base+3] = 0
        W[col_idx*2+1][base+3] = 1

    # Block 1: Strassen for B columns 0, 1
    # V rows: 0:(1,0), 1:(1,1), 2:(2,0), 3:(2,1)
    # W rows: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # This maps to:
    # V[0] = B[0,0], V[1]=B[0,1], V[2]=B[1,0], V[3]=B[1,1]
    # W[0] = C[0,0], W[1]=C[0,1], W[2]=C[1,0], W[3]=C[1,1]
    # My add_strassen assumes:
    # V[0]=B11, V[1]=B12, V[2]=B21, V[3]=B22
    # W[0]=C11, W[1]=C12, W[2]=C21, W[3]=C22
    # This matches the index convention:
    # V[j*p+k]. j=row, k=col.
    # V[0*2+0]=V[0], V[0*2+1]=V[1], V[1*2+0]=V[2], V[1*2+1]=V[3].
    # W[k*n+i]. k=col, i=row.
    # W[0*2+0]=W[0], W[0*2+1]=W[1], W[1*2+0]=W[2], W[1*2+1]=W[3].
    
    add_strassen(0)
    
    # Block 2: Strassen for B columns 2, 3
    # We need to map the local 2x2x2 to global indices.
    # Local V indices 0,1,2,3 correspond to Global V indices 4,5,6,7.
    # Local W indices 0,1,2,3 correspond to Global W indices 4,5,6,7.
    # I will modify add_strassen to accept a row offset for V and W.
    
    # Let's rewrite add_strassen with offsets.
    
    # Redefine to be safe.
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    def add_strassen(u_offset, v_offset, w_offset):
        base = u_offset
        # M1
        U[0][base] = 1
        V[v_offset+0][base] = 1
        V[v_offset+2][base] = 1
        V[v_offset+3][base] = 1
        W[w_offset+0][base] = 1
        W[w_offset+3][base] = -1
        
        # M2
        U[0][base+1] = 1
        U[2][base+1] = 1
        V[v_offset+3][base+1] = 1
        W[w_offset+3][base+1] = 1
        
        # M3
        U[3][base+2] = 1
        V[v_offset+0][base+2] = 1
        V[v_offset+1][base+2] = 1
        W[w_offset+2][base+2] = 1
        
        # M4
        U[0][base+3] = 1
        V[v_offset+2][base+3] = -1
        V[v_offset+3][base+3] = 1
        W[w_offset+3][base+3] = 1
        
        # M5
        U[0][base+4] = 1
        U[2][base+4] = 1
        V[v_offset+0][base+4] = 1
        W[w_offset+0][base+4] = 1
        
        # M6
        U[0][base+5] = -1
        U[2][base+5] = 1
        V[v_offset+1][base+5] = 1
        V[v_offset+2][base+5] = 1
        V[v_offset+3][base+5] = 1
        W[w_offset+3][base+5] = 1
        
        # M7
        U[0][base+6] = 1
        U[2][base+6] = -1
        V[v_offset+0][base+6] = 1
        V[v_offset+2][base+6] = 1
        W[w_offset+1][base+6] = 1

    def add_standard(u_offset, v_offset, w_offset):
        # v_offset is the column index in B (k).
        # w_offset is the column index in C (k).
        # Standard 2x2x1.
        # U rows 0..3.
        # V rows v_offset, v_offset+2. (j=0,k and j=1,k).
        # W rows w_offset, w_offset+1. (k,i=0 and k,i=1).
        
        # P1: A11 * B1
        U[0][u_offset] = 1
        V[v_offset][u_offset] = 1
        W[w_offset][u_offset] = 1
        
        # P2: A12 * B2
        U[1][u_offset+1] = 1
        V[v_offset+2][u_offset+1] = 1
        W[w_offset][u_offset+1] = 1
        
        # P3: A21 * B1
        U[2][u_offset+2] = 1
        V[v_offset][u_offset+2] = 1
        W[w_offset+1][u_offset+2] = 1
        
        # P4: A22 * B2
        U[3][u_offset+3] = 1
        V[v_offset+2][u_offset+3] = 1
        W[w_offset+1][u_offset+3] = 1

    # Block 1: Cols 0,1 of B
    # V rows: 0,1,2,3. W rows: 0,1,2,3.
    add_strassen(0, 0, 0)
    
    # Block 2: Cols 2,3 of B
    # V rows: 4,5,6,7. W rows: 4,5,6,7.
    add_strassen(7, 4, 4)
    
    # Block 3: Col 4 of B
    # V rows: 8 (j=0,k=4), 10 (j=1,k=4).
    # V index j*p+k.
    # j=0, k=4 -> 0*2+4 = 4? No.
    # m=2, p=5.
    # V has shape (m*p, rank) = (10, rank).
    # V[j*p+k].
    # j=0, k=0 -> 0
    # j=0, k=1 -> 1
    # j=0, k=2 -> 2
    # j=0, k=3 -> 3
    # j=0, k=4 -> 4
    # j=1, k=0 -> 5
    # j=1, k=1 -> 6
    # j=1, k=2 -> 7
    # j=1, k=3 -> 8
    # j=1, k=4 -> 9
    
    # My previous mapping in add_strassen assumed V rows 0,1,2,3 correspond to B[0,0], B[0,1], B[1,0], B[1,1].
    # This matches V[j*p+k] if p=2.
    # But p=5.
    # So for Block 1 (cols 0,1):
    # B[0,0] is V[0*5+0] = V[0]
    # B[0,1] is V[0*5+1] = V[1]
    # B[1,0] is V[1*5+0] = V[5]
    # B[1,1] is V[1*5+1] = V[6]
    
    # So the V offsets for Block 1 are 0, 1, 5, 6.
    # The V offsets for Block 2 (cols 2,3):
    # B[0,2] is V[2]
    # B[0,3] is V[3]
    # B[1,2] is V[7]
    # B[1,3] is V[8]
    # So V offsets: 2, 3, 7, 8.
    
    # W index k*n+i. p=5, n=2.
    # W has shape (p*n, rank) = (10, rank).
    # Block 1 (cols 0,1):
    # C[0,0] is W[0*2+0] = W[0]
    # C[0,1] is W[0*2+1] = W[1]
    # C[1,0] is W[1*2+0] = W[2]
    # C[1,1] is W[1*2+1] = W[3]
    # W offsets: 0, 1, 2, 3.
    
    # Block 2 (cols 2,3):
    # C[0,2] is W[2*2+0] = W[4]
    # C[0,3] is W[2*2+1] = W[5]
    # C[1,2] is W[3*2+0] = W[6]
    # C[1,3] is W[3*2+1] = W[7]
    # W offsets: 4, 5, 6, 7.
    
    # Block 3 (col 4):
    # B[0,4] is V[4]
    # B[1,4] is V[9]
    # C[0,4] is W[8]
    # C[1,4] is W[9]
    # V offsets: 4, 9.
    # W offsets: 8, 9.
    
    # Let's re-run the additions with correct offsets.
    
    # Reset matrices
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    def add_strassen_full(u_off, v0, v1, v2, v3, w0, w1, w2, w3):
        base = u_off
        # M1
        U[0][base] = 1
        V[v0][base] = 1
        V[v2][base] = 1
        V[v3][base] = 1
        W[w0][base] = 1
        W[w3][base] = -1
        
        # M2
        U[0][base+1] = 1
        U[2][base+1] = 1
        V[v3][base+1] = 1
        W[w3][base+1] = 1
        
        # M3
        U[3][base+2] = 1
        V[v0][base+2] = 1
        V[v1][base+2] = 1
        W[w2][base+2] = 1
        
        # M4
        U[0][base+3] = 1
        V[v2][base+3] = -1
        V[v3][base+3] = 1
        W[w3][base+3] = 1
        
        # M5
        U[0][base+4] = 1
        U[2][base+4] = 1
        V[v0][base+4] = 1
        W[w0][base+4] = 1
        
        # M6
        U[0][base+5] = -1
        U[2][base+5] = 1
        V[v1][base+5] = 1
        V[v2][base+5] = 1
        V[v3][base+5] = 1
        W[w3][base+5] = 1
        
        # M7
        U[0][base+6] = 1
        U[2][base+6] = -1
        V[v0][base+6] = 1
        V[v2][base+6] = 1
        W[w1][base+6] = 1

    def add_standard_full(u_off, v0, v1, w0, w1):
        base = u_off
        # P1: A11 * B1
        U[0][base] = 1
        V[v0][base] = 1
        W[w0][base] = 1
        
        # P2: A12 * B2
        U[1][base+1] = 1
        V[v1][base+1] = 1
        W[w0][base+1] = 1
        
        # P3: A21 * B1
        U[2][base+2] = 1
        V[v0][base+2] = 1
        W[w1][base+2] = 1
        
        # P4: A22 * B2
        U[3][base+3] = 1
        V[v1][base+3] = 1
        W[w1][base+3] = 1

    # Block 1: Cols 0,1
    # V: 0, 1, 5, 6
    # W: 0, 1, 2, 3
    add_strassen_full(0, 0, 1, 5, 6, 0, 1, 2, 3)
    
    # Block 2: Cols 2,3
    # V: 2, 3, 7, 8
    # W: 4, 5, 6, 7
    add_strassen_full(7, 2, 3, 7, 8, 4, 5, 6, 7)
    
    # Block 3: Col 4
    # V: 4, 9
    # W: 8, 9
    add_standard_full(14, 4, 9, 8, 9)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
