"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We decompose M_{3,3,5} into M_{3,3,3} and M_{3,3,2}
    # Rank of M_{3,3,3} is 19 (7 Strassen + 12 Schoolbook border)
    # Rank of M_{3,3,2} is 17 (7 Strassen + 10 Schoolbook border)
    # Total rank = 36
    
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # --- Part 1: M_{3,3,3} (columns 0-2 of B, rows 0-2 of C) ---
    # Indices: i,j,k in {0,1,2}
    # Core: i,j,k in {0,1}
    
    # Strassen Tensors for 2x2x2
    # P1 = (A00+A11)(B00+B11) -> C00 += P1, C11 += P1
    # P2 = (A10+A11)B00 -> C10 += P2
    # P3 = A00(B01-B11) -> C01 += P3
    # P4 = A11(B10-B00) -> C10 += P4, C00 += P4
    # P5 = (A00+A01)B11 -> C01 += P5, C00 -= P5
    # P6 = (A10-A00)(B00+B01) -> C11 += P6
    # P7 = (A01-A11)(B10+B11) -> C11 += P7, C00 += P7
    
    strassen_coeffs = [
        # (U_indices, V_indices, W_indices, W_coeff)
        # U is (i,j), V is (j,k), W is (k,i)
        # Note: W index is k*n + i. Here n=3.
        
        # P1
        ([(0,0), (1,1)], [(0,0), (1,1)], [(0,0,1), (1,1,1)]),
        # P2
        ([(1,0), (1,1)], [(0,0)], [(0,1,1)]),
        # P3
        ([(0,0)], [(0,1), (1,1)], [(1,0,1), (1,0,-1)]), # Wait, W is (k,i). P3 contributes to C01. k=1, i=0.
        # Let's be precise.
    ]
    
    # I will hardcode the 7 Strassen terms for the core.
    # U, V are 1s at specified positions. W is 1s at specified positions with signs.
    
    def add_strassen_core(offset_col, p_offset=0):
        # p_offset is for B's column offset (0 for first 2 cols, 2 for next 2? No, M_{3,3,2} uses first 2 cols of the 2-col block)
        # For M_{3,3,3}, B cols are 0,1.
        # For M_{3,3,2}, B cols are 3,4? No, B is 3x5.
        # We split B into B1 (3x3) and B2 (3x2).
        # B1 uses cols 0,1,2. B2 uses cols 3,4.
        # The Strassen core for M_{3,3,3} uses B cols 0,1.
        # The Strassen core for M_{3,3,2} uses B cols 3,4.
        
        nonlocal col
        # P1
        U[(0)*m+0][col] = 1; U[(1)*m+1][col] = 1
        V[(0)*p+0+p_offset][col] = 1; V[(1)*p+1+p_offset][col] = 1
        W[(0)*n+0][col] = 1; W[(1)*n+1][col] = 1
        col += 1
        
        # P2
        U[(1)*m+0][col] = 1; U[(1)*m+1][col] = 1
        V[(0)*p+0+p_offset][col] = 1
        W[(0)*n+1][col] = 1
        col += 1
        
        # P3
        U[(0)*m+0][col] = 1
        V[(0)*p+1+p_offset][col] = 1; V[(1)*p+1+p_offset][col] = -1
        W[(1)*n+0][col] = 1
        col += 1
        
        # P4
        U[(1)*m+1][col] = 1
        V[(1)*p+0+p_offset][col] = 1; V[(0)*p+0+p_offset][col] = -1
        W[(0)*n+1][col] = 1; W[(0)*n+0][col] = 1
        col += 1
        
        # P5
        U[(0)*m+0][col] = 1; U[(0)*m+1][col] = 1
        V[(1)*p+1+p_offset][col] = 1
        W[(1)*n+0][col] = 1; W[(0)*n+0][col] = -1
        col += 1
        
        # P6
        U[(1)*m+0][col] = 1; U[(0)*m+0][col] = -1
        V[(0)*p+0+p_offset][col] = 1; V[(0)*p+1+p_offset][col] = 1
        W[(1)*n+1][col] = 1
        col += 1
        
        # P7
        U[(0)*m+1][col] = 1; U[(1)*m+1][col] = -1
        V[(1)*p+0+p_offset][col] = 1; V[(1)*p+1+p_offset][col] = 1
        W[(1)*n+1][col] = 1; W[(0)*n+0][col] = 1
        col += 1

    add_strassen_core(0, 0) # For M_{3,3,3}, B cols 0,1
    
    # Border for M_{3,3,3}
    # Terms where i=2 or j=2 or k=2
    for i in range(3):
        for j in range(3):
            for k in range(3):
                if i == 2 or j == 2 or k == 2:
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
                    
    # --- Part 2: M_{3,3,2} (columns 3-4 of B, rows 2-3 of C? No, C is 3x5. C cols 3,4) ---
    # Wait, C is 3x5. C_{i,k} for k=3,4.
    # The W index is k*n + i. So k=3,4.
    
    add_strassen_core(0, 3) # For M_{3,3,2}, B cols 3,4. Note: p_offset is for V index.
    
    # Border for M_{3,3,2}
    # i,j in {0,1,2}, k in {3,4}
    for i in range(3):
        for j in range(3):
            for k in range(3, 5):
                if i == 2 or j == 2: # Core is i,j in {0,1}, k in {3,4}
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
