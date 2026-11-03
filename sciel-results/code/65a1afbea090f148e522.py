"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target rank 54 based on block decomposition:
    # A (3x5) split into A1 (2x5) and A2 (1x5)
    # B (5x5) split into B1 (5x3) and B2 (5x2)
    # C1 = A1 B1 (2x3), C12 = A1 B2 (2x2), C2 = A2 B (1x5)
    # Further splits for C1 and C12 using 2x2 Strassen where applicable.
    
    rank = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_schoolbook(i_start, i_end, j_start, j_end, k_start, k_end):
        nonlocal col
        for i in range(i_start, i_end):
            for j in range(j_start, j_end):
                for k in range(k_start, k_end):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1

    def add_strassen_2x2(i_base, j_base, k_base):
        # Adds Strassen decomposition for 2x2x2 block
        # i_base, j_base, k_base are the starting indices
        # Local indices: i in {0,1}, j in {0,1}, k in {0,1}
        # Global indices: I = i_base + i, J = j_base + j, K = k_base + k
        nonlocal col
        
        # M1 = (a11 + a22)(b11 + b22)
        # a11: I=i_base, J=j_base
        # a22: I=i_base+1, J=j_base+1
        # b11: J=j_base, K=k_base
        # b22: J=j_base+1, K=k_base+1
        U[i_base * m + j_base][col] = 1
        U[(i_base + 1) * m + (j_base + 1)][col] = 1
        V[j_base * p + k_base][col] = 1
        V[(j_base + 1) * p + (k_base + 1)][col] = 1
        # c11 += M1
        W[k_base * n + i_base][col] = 1
        col += 1
        
        # M2 = (a21 + a22) b11
        # a21: I=i_base+1, J=j_base
        # a22: I=i_base+1, J=j_base+1
        # b11: J=j_base, K=k_base
        U[(i_base + 1) * m + j_base][col] = 1
        U[(i_base + 1) * m + (j_base + 1)][col] = 1
        V[j_base * p + k_base][col] = 1
        # c21 += M2
        W[k_base * n + (i_base + 1)][col] = 1
        col += 1
        
        # M3 = a11 (b21 - b22)
        # a11: I=i_base, J=j_base
        # b21: J=j_base+1, K=k_base
        # b22: J=j_base+1, K=k_base+1
        U[i_base * m + j_base][col] = 1
        V[(j_base + 1) * p + k_base][col] = 1
        V[(j_base + 1) * p + (k_base + 1)][col] = -1
        # c12 += M3
        W[(k_base + 1) * n + i_base][col] = 1
        col += 1
        
        # M4 = a22 (b12 - b11)
        # a22: I=i_base+1, J=j_base+1
        # b12: J=j_base, K=k_base+1
        # b11: J=j_base, K=k_base
        U[(i_base + 1) * m + (j_base + 1)][col] = 1
        V[j_base * p + (k_base + 1)][col] = 1
        V[j_base * p + k_base][col] = -1
        # c11 -= M4, c12 += M4
        W[k_base * n + i_base][col] = -1
        W[(k_base + 1) * n + i_base][col] = 1
        col += 1
        
        # M5 = (a11 + a12) b22
        # a11: I=i_base, J=j_base
        # a12: I=i_base, J=j_base+1
        # b22: J=j_base+1, K=k_base+1
        U[i_base * m + j_base][col] = 1
        U[i_base * m + (j_base + 1)][col] = 1
        V[(j_base + 1) * p + (k_base + 1)][col] = 1
        # c11 += M5, c22 += M5
        W[k_base * n + i_base][col] = 1
        W[(k_base + 1) * n + (i_base + 1)][col] = 1
        col += 1
        
        # M6 = (a21 - a11)(b11 + b12)
        # a21: I=i_base+1, J=j_base
        # a11: I=i_base, J=j_base
        # b11: J=j_base, K=k_base
        # b12: J=j_base, K=k_base+1
        U[(i_base + 1) * m + j_base][col] = 1
        U[i_base * m + j_base][col] = -1
        V[j_base * p + k_base][col] = 1
        V[j_base * p + (k_base + 1)][col] = 1
        # c22 += M6
        W[(k_base + 1) * n + (i_base + 1)][col] = 1
        col += 1
        
        # M7 = (a12 - a22)(b21 + b22)
        # a12: I=i_base, J=j_base+1
        # a22: I=i_base+1, J=j_base+1
        # b21: J=j_base+1, K=k_base
        # b22: J=j_base+1, K=k_base+1
        U[i_base * m + (j_base + 1)][col] = 1
        U[(i_base + 1) * m + (j_base + 1)][col] = -1
        V[(j_base + 1) * p + k_base][col] = 1
        V[(j_base + 1) * p + (k_base + 1)][col] = 1
        # c11 -= M7, c21 += M7
        W[k_base * n + i_base][col] = -1
        W[k_base * n + (i_base + 1)][col] = 1
        col += 1

    # 1. C2 = A2 B (1x5 by 5x5) -> 5 terms
    # i=2, j=0..4, k=0..4
    add_schoolbook(2, 3, 0, 5, 0, 5)
    
    # 2. C11 = A1 B1 (2x5 by 5x3)
    # A1: i=0..1, j=0..4
    # B1: j=0..4, k=0..2
    # Split A1 into A11 (j=0..2) and A12 (j=3..4)
    # Split B1 into B11 (j=0..2) and B12 (j=3..4)
    
    # A11 B11 (2x3 by 3x3) -> 18 terms
    add_schoolbook(0, 2, 0, 3, 0, 3)
    
    # A12 B12 (2x2 by 2x3) -> 12 terms
    add_schoolbook(0, 2, 3, 5, 0, 3)
    
    # 3. C12 = A1 B2 (2x5 by 5x2)
    # A1: i=0..1, j=0..4
    # B2: j=0..4, k=3..4
    # Split A1 into A11' (j=0..2) and A12' (j=3..4)
    # Split B2 into B21 (j=0..2) and B22 (j=3..4)
    
    # A11' B21 (2x3 by 3x2) -> 12 terms
    add_schoolbook(0, 2, 0, 3, 3, 5)
    
    # A12' B22 (2x2 by 2x2) -> 7 terms (Strassen)
    # i_base=0, j_base=3, k_base=3
    add_strassen_2x2(0, 3, 3)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
