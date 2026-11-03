"""Program that uses a split-row block to reduce rank below schoolbook 48."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=3, m=4, p=4
    # Rank 44 construction:
    # Top 2 rows of A (2x4) * B (4x4) using 4 Strassen blocks (rank 7 each) -> 28
    # Bottom 1 row of A (1x4) * B (4x4) using schoolbook (rank 16) -> 16
    # Total rank = 44
    
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen_block(a_idx, b_idx, c_idx, col):
        # a_idx: [a00, a01, a10, a11] global U indices
        # b_idx: [b00, b01, b10, b11] global V indices
        # c_idx: [c00, c01, c10, c11] global W indices
        # col: current column index
        
        # M1 = (A00+A11)(B00+B11)
        U[a_idx[0]][col] += 1
        U[a_idx[3]][col] += 1
        V[b_idx[0]][col] += 1
        V[b_idx[3]][col] += 1
        W[c_idx[0]][col] += 1
        W[c_idx[3]][col] += 1
        col += 1
        
        # M2 = (A10+A11)B00
        U[a_idx[2]][col] += 1
        U[a_idx[3]][col] += 1
        V[b_idx[0]][col] += 1
        W[c_idx[2]][col] += 1
        W[c_idx[3]][col] -= 1
        col += 1
        
        # M3 = A00(B10-B11)
        U[a_idx[0]][col] += 1
        V[b_idx[2]][col] += 1
        V[b_idx[3]][col] -= 1
        W[c_idx[1]][col] += 1
        W[c_idx[3]][col] += 1
        col += 1
        
        # M4 = A11(B01-B00)
        U[a_idx[3]][col] += 1
        V[b_idx[1]][col] += 1
        V[b_idx[0]][col] -= 1
        W[c_idx[0]][col] += 1
        W[c_idx[2]][col] += 1
        col += 1
        
        # M5 = (A00+A11)B11
        U[a_idx[0]][col] += 1
        U[a_idx[3]][col] += 1
        V[b_idx[3]][col] += 1
        W[c_idx[0]][col] -= 1
        W[c_idx[1]][col] += 1
        col += 1
        
        # M6 = (A01-A00)(B00+B10)
        U[a_idx[1]][col] += 1
        U[a_idx[0]][col] -= 1
        V[b_idx[0]][col] += 1
        V[b_idx[2]][col] += 1
        W[c_idx[3]][col] += 1
        col += 1
        
        # M7 = (A10-A11)(B01+B11)
        U[a_idx[2]][col] += 1
        U[a_idx[3]][col] -= 1
        V[b_idx[1]][col] += 1
        V[b_idx[3]][col] += 1
        W[c_idx[0]][col] += 1
        col += 1
        
        return col

    # Define indices for A (3x4), B (4x4), C (3x4)
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    a1 = [0*4+0, 0*4+1, 1*4+0, 1*4+1] # A00, A01, A10, A11
    b1 = [0*4+0, 0*4+1, 1*4+0, 1*4+1] # B00, B01, B10, B11
    c1 = [0*3+0, 1*3+0, 0*3+1, 1*3+1] # C00, C01, C10, C11
    col = add_strassen_block(a1, b1, c1, 0)
    
    # Block 2: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2] (added to previous)
    # Note: C indices are the same as Block 1 because it adds to the same output block.
    a2 = [0*4+2, 0*4+3, 1*4+2, 1*4+3]
    b2 = [2*4+0, 2*4+1, 3*4+0, 3*4+1]
    c2 = c1 # Same output block C[0:2, 0:2]
    col = add_strassen_block(a2, b2, c2, col)
    
    # Block 3: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    a3 = a1
    b3 = [0*4+2, 0*4+3, 1*4+2, 1*4+3]
    c3 = [2*3+0, 3*3+0, 2*3+1, 3*3+1] # C[0,2], C[0,3], C[1,2], C[1,3]
    col = add_strassen_block(a3, b3, c3, col)
    
    # Block 4: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4] (added to previous)
    a4 = a2
    b4 = [2*4+2, 2*4+3, 3*4+2, 3*4+3]
    c4 = c3 # Same output block C[0:2, 2:4]
    col = add_strassen_block(a4, b4, c4, col)
    
    # Bottom row: A[2, 0:4] * B[0:4, 0:4] -> C[2, 0:4]
    # Schoolbook for 1x4 * 4x4
    for j in range(4):
        for k in range(4):
            U[2*4 + j][col] += 1
            V[j*4 + k][col] += 1
            W[k*3 + 2][col] += 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
