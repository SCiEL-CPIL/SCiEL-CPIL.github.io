"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Use 7 Strassen blocks (2x2) and 1 Schoolbook block (2x1)
    # Total rank: 7 * 7 + 4 = 53
    
    # Strassen decomposition for 2x2
    # U, V, W for M_2,2,2
    # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    
    # We will build the global U, V, W
    # U shape (4, R), V shape (30, R), W shape (30, R)
    
    rank = 53
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(30)]
    W = [[0] * rank for _ in range(30)]
    
    col = 0
    # 7 Strassen blocks
    for block_idx in range(7):
        # Columns 2*block_idx, 2*block_idx+1
        k_offset = 2 * block_idx
        
        # Strassen products
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B01
        # M3 = A00(B10-B11)
        # M4 = A11(B00-B10)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # U entries (A part)
        # U indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        
        # M1
        U[0][col] = 1
        U[3][col] = 1
        V[0*15 + k_offset][col] = 1
        V[1*15 + k_offset+1][col] = 1
        # W entries (C part)
        # C00 = M1+M4-M5+M7
        # C01 = M2+M3
        # C10 = M6+M7
        # C11 = M1-M2+M3+M5
        # W indices: k*2+i
        # C00: k=0, i=0 -> idx 0
        # C11: k=1, i=1 -> idx 3
        W[0*2 + 0][col] = 1
        W[1*2 + 1][col] = 1
        col += 1
        
        # M2
        U[2][col] = 1
        U[3][col] = 1
        V[0*15 + k_offset+1][col] = 1
        # C01: k=1, i=0 -> idx 2
        W[1*2 + 0][col] = 1
        col += 1
        
        # M3
        U[0][col] = 1
        V[1*15 + k_offset][col] = 1
        V[1*15 + k_offset+1][col] = -1
        # C01: k=1, i=0 -> idx 2
        # C11: k=1, i=1 -> idx 3
        W[1*2 + 0][col] = 1
        W[1*2 + 1][col] = 1
        col += 1
        
        # M4
        U[3][col] = 1
        V[0*15 + k_offset][col] = 1
        V[1*15 + k_offset][col] = -1
        # C00: k=0, i=0 -> idx 0
        W[0*2 + 0][col] = 1
        col += 1
        
        # M5
        U[0][col] = 1
        U[1][col] = 1
        V[1*15 + k_offset+1][col] = 1
        # C00: k=0, i=0 -> idx 0
        # C11: k=1, i=1 -> idx 3
        W[0*2 + 0][col] = -1
        W[1*2 + 1][col] = 1
        col += 1
        
        # M6
        U[2][col] = 1
        U[0][col] = -1
        V[0*15 + k_offset][col] = 1
        V[0*15 + k_offset+1][col] = 1
        # C10: k=0, i=1 -> idx 1
        W[0*2 + 1][col] = 1
        col += 1
        
        # M7
        U[1][col] = 1
        U[3][col] = -1
        V[1*15 + k_offset][col] = 1
        V[1*15 + k_offset+1][col] = 1
        # C00: k=0, i=0 -> idx 0
        # C10: k=0, i=1 -> idx 1
        W[0*2 + 0][col] = 1
        W[0*2 + 1][col] = 1
        col += 1

    # 1 Schoolbook block for last column (k=14)
    k_offset = 14
    # C00 = A00 B0k + A01 B1k ? No, C[i,k] = sum_j A[i,j] B[j,k]
    # C0,14 = A00 B0,14 + A01 B1,14
    # C1,14 = A10 B0,14 + A11 B1,14
    
    # Term 1: A00 B0,14 -> C0,14
    U[0][col] = 1
    V[0*15 + k_offset][col] = 1
    W[k_offset*2 + 0][col] = 1
    col += 1
    
    # Term 2: A01 B1,14 -> C0,14
    U[1][col] = 1
    V[1*15 + k_offset][col] = 1
    W[k_offset*2 + 0][col] = 1
    col += 1
    
    # Term 3: A10 B0,14 -> C1,14
    U[2][col] = 1
    V[0*15 + k_offset][col] = 1
    W[k_offset*2 + 1][col] = 1
    col += 1
    
    # Term 4: A11 B1,14 -> C1,14
    U[3][col] = 1
    V[1*15 + k_offset][col] = 1
    W[k_offset*2 + 1][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
