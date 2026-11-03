"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Rank 44 decomposition for 2x3x8 matrix multiplication.
    # Strategy: Split p=8 into two p=4 blocks.
    # For each p=4 block, split m=3 into m=2 and m=1.
    # For m=2, p=4 part: split into two 2x2x2 blocks, each rank 7 (Strassen) -> 14
    # For m=1, p=4 part: schoolbook 2x1x4, rank 8
    # Total per p=4 block: 22, total: 44
    
    rank = 44
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen(i_off, j_off, k_off):
        nonlocal col
        # m1: (a+d)(e+h) -> contributes to C[0,0] and C[1,1]
        U[i_off*m + j_off][col] += 1
        U[(i_off+1)*m + (j_off+1)][col] += 1
        V[j_off*p + k_off][col] += 1
        V[(j_off+1)*p + (k_off+1)][col] += 1
        W[k_off*n + i_off][col] += 1
        W[(k_off+1)*n + (i_off+1)][col] += 1
        col += 1
        
        # m2: (c+d)e -> contributes to C[1,0] and C[1,1]
        U[(i_off+1)*m + j_off][col] += 1
        U[(i_off+1)*m + (j_off+1)][col] += 1
        V[j_off*p + k_off][col] += 1
        W[k_off*n + (i_off+1)][col] += 1
        W[(k_off+1)*n + (i_off+1)][col] += -1
        col += 1
        
        # m3: a(f-h) -> contributes to C[0,1] and C[1,1]
        U[i_off*m + j_off][col] += 1
        V[j_off*p + (k_off+1)][col] += 1
        V[(j_off+1)*p + (k_off+1)][col] += -1
        W[(k_off+1)*n + i_off][col] += 1
        W[(k_off+1)*n + (i_off+1)][col] += 1
        col += 1
        
        # m4: d(g-e) -> contributes to C[0,0] and C[1,0]
        U[(i_off+1)*m + (j_off+1)][col] += 1
        V[(j_off+1)*p + k_off][col] += 1
        V[j_off*p + k_off][col] += -1
        W[k_off*n + i_off][col] += 1
        W[k_off*n + (i_off+1)][col] += 1
        col += 1
        
        # m5: (a+b)h -> contributes to C[0,0] and C[0,1]
        U[i_off*m + j_off][col] += 1
        U[i_off*m + (j_off+1)][col] += 1
        V[(j_off+1)*p + (k_off+1)][col] += 1
        W[k_off*n + i_off][col] += -1
        W[(k_off+1)*n + i_off][col] += 1
        col += 1
        
        # m6: (c-a)(e+f) -> contributes to C[1,1]
        U[(i_off+1)*m + j_off][col] += 1
        U[i_off*m + j_off][col] += -1
        V[j_off*p + k_off][col] += 1
        V[j_off*p + (k_off+1)][col] += 1
        W[(k_off+1)*n + (i_off+1)][col] += 1
        col += 1
        
        # m7: (b-d)(g+h) -> contributes to C[0,0]
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
                U[(i_off+i)*m + (j_off+0)][col] = 1
                V[(j_off+0)*p + (k_off+k)][col] = 1
                W[(k_off+k)*n + (i_off+i)][col] = 1
                col += 1

    # Block 1: k in 0..3
    add_strassen(0, 0, 0)       # j=0,1; k=0,1
    add_strassen(0, 0, 2)       # j=0,1; k=2,3
    add_schoolbook_2x1x4(0, 2, 0)  # j=2; k=0..3

    # Block 2: k in 4..7
    add_strassen(0, 0, 4)       # j=0,1; k=4,5
    add_strassen(0, 0, 6)       # j=0,1; k=6,7
    add_schoolbook_2x1x4(0, 2, 4)  # j=2; k=4..7

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
