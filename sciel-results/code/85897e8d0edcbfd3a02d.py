"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 7
    # U: (n*m, rank) = (16, rank), U[i*m+j, c] = U[i*4+j, c] for A[i,j]
    # V: (m*p, rank) = (28, rank), V[j*p+k, c] = V[j*7+k, c] for B[j,k]
    # W: (p*n, rank) = (28, rank), W[k*n+i, c] = W[k*4+i, c] for C[i,k]
    
    # Strategy: 4x4*4x4 (Strassen, rank 49) + 4x4*4x3 (schoolbook, rank 48) = 97
    
    rank = 97
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(28)]
    W = [[0] * rank for _ in range(28)]
    col = 0
    
    # Part 1: 4x4 * 4x4 -> 4x4 (columns 0-3 of B, rows 0-3 of C)
    # A[i,j] -> U[i*4+j]
    # B[j,k] for k=0..3 -> V[j*7+k]
    # C[i,k] for k=0..3 -> W[k*4+i]
    
    # Strassen for 4x4*4x4:
    # Split into 2x2 blocks
    # A11 = A[0:2,0:2], A12 = A[0:2,2:4], A21 = A[2:4,0:2], A22 = A[2:4,2:4]
    # B11 = B[0:2,0:2], B12 = B[0:2,2:4], B21 = B[2:4,0:2], B22 = B[2:4,2:4]
    # C11 = C[0:2,0:2], C12 = C[0:2,2:4], C21 = C[2:4,0:2], C22 = C[2:4,2:4]
    
    # U indices:
    A11_U = [0, 1, 4, 5]   # A[0,0],A[0,1],A[1,0],A[1,1]
    A12_U = [2, 3, 6, 7]   # A[0,2],A[0,3],A[1,2],A[1,3]
    A21_U = [8, 9, 12, 13] # A[2,0],A[2,1],A[3,0],A[3,1]
    A22_U = [10, 11, 14, 15] # A[2,2],A[2,3],A[3,2],A[3,3]
    
    # V indices for B[0:4, 0:4]:
    B11_V = [0, 1, 7, 8]   # B[0,0],B[0,1],B[1,0],B[1,1]
    B12_V = [2, 3, 9, 10]  # B[0,2],B[0,3],B[1,2],B[1,3]
    B21_V = [14, 15, 21, 22] # B[2,0],B[2,1],B[3,0],B[3,1]
    B22_V = [16, 17, 23, 24] # B[2,2],B[2,3],B[3,2],B[3,3]
    
    # W indices for C[0:4, 0:4]: C[i,k] -> W[k*4+i]
    C11_W = [0, 1, 4, 5]   # C[0,0],C[1,0],C[0,1],C[1,1] -> W[0],W[1],W[4],W[5]
    C12_W = [2, 3, 6, 7]   # C[0,2],C[1,2],C[0,3],C[1,3] -> W[2],W[3],W[6],W[7]
    C21_W = [8, 9, 12, 13] # C[2,0],C[3,0],C[2,1],C[3,1] -> W[8],W[9],W[12],W[13]
    C22_W = [10, 11, 14, 15] # C[2,2],C[3,2],C[2,3],C[3,3] -> W[10],W[11],W[14],W[15]
    
    # Strassen M1 = (A11+A22)(B11+B22)
    for idx_a, idx_v, idx_w in [(A11_U[0], B11_V[0], C11_W[0]),
                                  (A11_U[1], B11_V[1], C11_W[1]),
                                  (A11_U[2], B11_V[2], C11_W[2]),
                                  (A11_U[3], B11_V[3], C11_W[3]),
                                  (A22_U[0], B22_V[0], C22_W[0]),
                                  (A22_U[1], B22_V[1], C22_W[1]),
                                  (A22_U[2], B22_V[2], C22_W[2]),
                                  (A22_U[3], B22_V[3], C22_W[3])]:
        pass  # This approach is getting messy. Let me use a cleaner method.
    
    # Let me just directly code each Strassen term.
    # For a 2x2*2x2 Strassen, the 7 terms involve sums/differences of 4 elements each.
    # I'll add contributions to U, V, W for each term.
    
    def add_term(u_indices, v_indices, w_indices, col_idx):
        """Add a rank-1 term: sum of U entries * sum of V entries -> sum of W entries"""
        for u in u_indices:
            U[u][col_idx] += 1
        for v in v_indices:
            V[v][col_idx] += 1
        for w in w_indices:
            W[w][col_idx] += 1
    
    def add_term_with_signs(u_signs, v_signs, w_signs, col_idx):
        """Add a rank-1 term with signs"""
        for u, s in u_signs:
            U[u][col_idx] += s
        for v, s in v_signs:
            V[v][col_idx] += s
        for w, s in w_signs:
            W[w][col_idx] += s
    
    # M1 = (A11+A22)(B11+B22) -> contributes to C11 and C22
    # (A11+A22) has entries at A11_U + A22_U with +1
    # (B11+B22) has entries at B11_V + B22_V with +1
    # Result: C11 = M1+M4-M5+M7, C22 = M1-M2+M3+M6
    # So M1 contributes +1 to C11 and +1 to C22
    u_m1 = [(i, 1) for i in A11_U + A22_U]
    v_m1 = [(i, 1) for i in B11_V + B22_V]
    w_m1 = [(i, 1) for i in C11_W + C22_W]
    add_term_with_signs(u_m1, v_m1, w_m1, col)
    col += 1
    
    # M2 = (A21+A22)B11 -> contributes to C21
    u_m2 = [(i, 1) for i in A21_U + A22_U]
    v_m2 = [(i, 1) for i in B11_V]
    w_m2 = [(i, 1) for i in C21_W]
    add_term_with_signs(u_m2, v_m2, w_m2, col)
    col += 1
    
    # M3 = A11(B12-B22) -> contributes to C12
    u_m3 = [(i, 1) for i in A11_U]
    v_m3 = [(i, 1) for i in B12_V] + [(i, -1) for i in B22_V]
    w_m3 = [(i, 1) for i in C12_W]
    add_term_with_signs(u_m3, v_m3, w_m3, col)
    col += 1
    
    # M4 = A22(B21-B11) -> contributes to C11 and C21
    u_m4 = [(i, 1) for i in A22_U]
    v_m4 = [(i, 1) for i in B21_V] + [(i, -1) for i in B11_V]
    w_m4 = [(i, 1) for i in C11_W] + [(i, 1) for i in C21_W]
    add_term_with_signs(u_m4, v_m4, w_m4, col)
    col += 1
    
    # M5 = (A11+A12)B22 -> contributes to C11 and C22
    u_m5 = [(i, 1) for i in A11_U + A12_U]
    v_m5 = [(i, 1) for i in B22_V]
    w_m5 = [(i, -1) for i in C11_W] + [(i, 1) for i in C22_W]
    add_term_with_signs(u_m5, v_m5, w_m5, col)
    col += 1
    
    # M6 = (A21-A11)(B11+B12) -> contributes to C12 and C22
    u_m6 = [(i, 1) for i in A21_U] + [(i, -1) for i in A11_U]
    v_m6 = [(i, 1) for i in B11_V + B12_V]
    w_m6 = [(i, 1) for i in C12_W] + [(i, 1) for i in C22_W]
    add_term_with_signs(u_m6, v_m6, w_m6, col)
    col += 1
    
    # M7 = (A12-A22)(B21+B22) -> contributes to C11 and C21
    u_m7 = [(i, 1) for i in A12_U] + [(i, -1) for i in A22_U]
    v_m7 = [(i, 1) for i in B21_V + B22_V]
    w_m7 = [(i, 1) for i in C11_W] + [(i, 1) for i in C21_W]
    add_term_with_signs(u_m7, v_m7, w_m7, col)
    col += 1
    
    # Part 2: 4x4 * 4x3 -> 4x3 (columns 4-6 of B, rows 4-6 of C)
    # Schoolbook: 4*4*3 = 48 terms
    for i in range(4):
        for j in range(4):
            for k in range(3):
                U[i*4+j][col] += 1
                V[j*7+(4+k)][col] += 1
                W[(4+k)*4+i][col] += 1
                col += 1
    
    assert col == rank, f"Expected {rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

