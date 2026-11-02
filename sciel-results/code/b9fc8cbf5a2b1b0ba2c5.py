"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Strategy: Split p=8 into two p=4 blocks.
    # For each 2x3x4 block:
    #   - Use Strassen (rank 7) for the 2x2x2 sub-blocks (two of them)
    #   - Use schoolbook (rank 8) for the 2x1x4 part
    # Total rank: 2 * (7 + 7 + 8) = 44
    
    rank = 44
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen(i_off, j_off, k_off):
        nonlocal col
        # m1: (a+d)(e+h)
        U[i_off*m + j_off][col] += 1
        U[(i_off+1)*m + (j_off+1)][col] += 1
        V[j_off*p + k_off][col] += 1
        V[(j_off+1)*p + (k_off+1)][col] += 1
        W[k_off*n + i_off][col] += 1
        W[(k_off+1)*n + (i_off+1)][col] += 1
        col += 1
        
        # m2: (c+d)e
        U[(i_off+1)*m + j_off][col] += 1
        U[(i_off+1)*m + (j_off+1)][col] += 1
        V[j_off*p + k_off][col] += 1
        W[k_off*n + (i_off+1)][col] += 1
        W[(k_off+1)*n + (i_off+1)][col] += -1
        col += 1
        
        # m3: a(f-h)
        U[i_off*m + j_off][col] += 1
        V[j_off*p + (k_off+1)][col] += 1
        V[(j_off+1)*p + (k_off+1)][col] += -1
        W[(k_off+1)*n + i_off][col] += 1
        W[(k_off+1)*n + (i_off+1)][col] += 1
        col += 1
        
        # m4: d(g-e)
        U[(i_off+1)*m + (j_off+1)][col] += 1
        V[(j_off+1)*p + k_off][col] += 1
        V[j_off*p + k_off][col] += -1
        W[k_off*n + i_off][col] += 1
        W[k_off*n + (i_off+1)][col] += 1
        col += 1
        
        # m5: (a+b)h
        U[i_off*m + j_off][col] += 1
        U[i_off*m + (j_off+1)][col] += 1
        V[(j_off+1)*p + (k_off+1)][col] += 1
        W[k_off*n + i_off][col] += -1
        W[(k_off+1)*n + i_off][col] += 1
        col += 1
        
        # m6: (c-a)(e+f)
        U[(i_off+1)*m + j_off][col] += 1
        U[i_off*m + j_off][col] += -1
        V[j_off*p + k_off][col] += 1
        V[j_off*p + (k_off+1)][col] += 1
        W[(k_off+1)*n + (i_off+1)][col] += 1
        col += 1
        
        # m7: (b-d)(g+h)
        U[i_off*m + (j_off+1)][col] += 1
        U[(i_off+1)*m + (j_off+1)][col] += -1
        V[(j_off+1)*p + k_off][col] += 1
        V[(j_off+1)*p + (k_off+1)][col] += 1
        W[k_off*n + i_off][col] += 1
        col += 1

    def add_schoolbook_2x1x4(i_off, j_off, k_off):
        nonlocal col
        for i in range(2):
            for k in range(4):
                U[(i_off+i)*m + j_off][col] = 1
                V[j_off*p + (k_off+k)][col] = 1
                W[(k_off+k)*n + (i_off+i)][col] = 1
                col += 1

    # Block 1: k in 0..3
    add_strassen(0, 0, 0)      # j=0,1; k=0,1
    add_strassen(0, 0, 2)      # j=0,1; k=2,3
    add_schoolbook_2x1x4(0, 2, 0)  # j=2; k=0..3

    # Block 2: k in 4..7
    add_strassen(0, 0, 4)      # j=0,1; k=4,5
    add_strassen(0, 0, 6)      # j=0,1; k=6,7
    add_schoolbook_2x1x4(0, 2, 4)  # j=2; k=4..7

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
