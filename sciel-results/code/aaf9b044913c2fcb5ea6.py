"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # For 3x3x7: decompose as two 3x3x3 blocks + one 3x3x1 block
    # Use Strassen for the 2x2x2 core of each 3x3x3 block
    # Cross terms use schoolbook
    # 3x3x3: 7 (Strassen) + 4 (a*f^T) + 4 (A11*e) + 2 (a*d) + 4 (b^T*B11) + 2 (c*f^T) + 2 (b^T*e) + 1 (c*d) = 26
    # 3x3x1: 9 (schoolbook, optimal)
    # Total: 26 + 26 + 9 = 61
    
    rank = 61
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    
    def add_block_3x3x3(k_offset):
        """Add 26 terms for a 3x3x3 block with column offset k_offset."""
        nonlocal col
        # A indices: A[i,j] -> U[i*m + j] = U[i*3 + j]
        # B indices: B[j,k] -> V[j*p + k] = V[j*3 + (k + k_offset)]
        # C indices: C[i,k] -> W[k*n + i] = W[(k + k_offset)*3 + i]
        
        # Strassen 2x2x2 for A[0:2,0:2] * B[0:2, 0:2+k_offset]
        a00, a01, a10, a11 = 0, 1, 3, 4
        b00 = 0*3 + k_offset      # B[0, k_offset]
        b01 = 0*3 + k_offset + 1  # B[0, k_offset+1]
        b10 = 1*3 + k_offset      # B[1, k_offset]
        b11 = 1*3 + k_offset + 1  # B[1, k_offset+1]
        c00 = k_offset*3 + 0      # C[0, k_offset]
        c01 = (k_offset+1)*3 + 0  # C[0, k_offset+1]
        c10 = k_offset*3 + 1      # C[1, k_offset]
        c11 = (k_offset+1)*3 + 1  # C[1, k_offset+1]
        
        # P1 = (A00+A11)(B00+B11) -> C00 += P1, C11 += P1
        U[a00][col] = 1; U[a11][col] = 1
        V[b00][col] = 1; V[b11][col] = 1
        W[c00][col] = 1; W[c11][col] = 1
        col += 1
        
        # P2 = (A10+A11)(B00) -> C10 += P2
        U[a10][col] = 1; U[a11][col] = 1
        V[b00][col] = 1
        W[c10][col] = 1
        col += 1
        
        # P3 = A00(B10-B11) -> C01 += P3
        U[a00][col] = 1
        V[b10][col] = 1; V[b11][col] = -1
        W[c01][col] = 1
        col += 1
        
        # P4 = A11(B01-B00) -> C00 += P4, C10 += P4
        U[a11][col] = 1
        V[b01][col] = 1; V[b00][col] = -1
        W[c00][col] = 1; W[c10][col] = 1
        col += 1
        
        # P5 = (A00+A01)(B11) -> C01 += P5, C11 -= P5
        U[a00][col] = 1; U[a01][col] = 1
        V[b11][col] = 1
        W[c01][col] = 1; W[c11][col] = -1
        col += 1
        
        # P6 = (A10-A00)(B00+B01) -> C11 += P6
        U[a10][col] = 1; U[a00][col] = -1
        V[b00][col] = 1; V[b01][col] = 1
        W[c11][col] = 1
        col += 1
        
        # P7 = (A01-A11)(B10+B11) -> C00 += P7
        U[a01][col] = 1; U[a11][col] = -1
        V[b10][col] = 1; V[b11][col] = 1
        W[c00][col] = 1
        col += 1
        
        # Cross terms:
        # a = A[0:2, 2] (2x1): U indices 2, 5
        # f = B[2, 0:2+k_offset] (1x2): V indices 6+k_offset, 6+k_offset+1
        a_idx = [2, 5]
        f_idx = [6 + k_offset, 6 + k_offset + 1]
        
        # a*f^T: 4 terms
        for i in range(2):
            for j in range(2):
                U[a_idx[i]][col] = 1
                V[f_idx[j]][col] = 1
                W[(k_offset + j)*3 + i][col] = 1
                col += 1
        
        # A11*e: 4 terms, e = B[0:2, 2+k_offset]
        e_idx = [2 + k_offset, 3 + 2 + k_offset]
        for i in range(2):
            for j in range(2):
                U[i*3 + j][col] = 1
                V[e_idx[j]][col] = 1
                W[(2 + k_offset)*3 + i][col] = 1
                col += 1
        
        # a*d: 2 terms, d = B[2, 2+k_offset]
        d_idx = 8 + k_offset
        for i in range(2):
            U[a_idx[i]][col] = 1
            V[d_idx][col] = 1
            W[(2 + k_offset)*3 + i][col] = 1
            col += 1
        
        # b^T*B11: 4 terms, b = A[2, 0:2]
        b_idx = [6, 7]
        for i in range(2):
            for j in range(2):
                U[b_idx[i]][col] = 1
                V[j*3 + k_offset][col] = 1
                W[(k_offset + j)*3 + 2][col] = 1
                col += 1
        
        # c*f^T: 2 terms, c = A[2,2]
        c_idx = 8
        for j in range(2):
            U[c_idx][col] = 1
            V[f_idx[j]][col] = 1
            W[(k_offset + j)*3 + 2][col] = 1
            col += 1
        
        # b^T*e: 2 terms
        for i in range(2):
            U[b_idx[i]][col] = 1
            V[e_idx[i]][col] = 1
            W[(2 + k_offset)*3 + 2][col] = 1
            col += 1
        
        # c*d: 1 term
        U[c_idx][col] = 1
        V[d_idx][col] = 1
        W[(2 + k_offset)*3 + 2][col] = 1
        col += 1
    
    # Add two 3x3x3 blocks
    add_block_3x3x3(0)
    add_block_3x3x3(3)
    
    # Add one 3x3x1 block (column k=6)
    for i in range(n):
        for j in range(m):
            U[i*m + j][col] = 1
            V[j*p + 6][col] = 1
            W[6*n + i][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
