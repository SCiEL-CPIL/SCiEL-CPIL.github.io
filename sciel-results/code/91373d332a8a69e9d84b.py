"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 28 decomposition
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen block
    # i_start, i_end: rows of A (0, 1)
    # j_start, j_end: cols of A / rows of B (0, 1)
    # k_start, k_end: cols of B (0, 1)
    def add_strassen(i_s, i_e, j_s, j_e, k_s, k_e):
        nonlocal col
        # A indices: (i, j) -> i in [i_s, i_e], j in [j_s, j_e]
        # B indices: (j, k) -> j in [j_s, j_e], k in [k_s, k_e]
        # Local indices 0,1
        # a00 = A[i_s, j_s], a01 = A[i_s, j_s+1], a10 = A[i_s+1, j_s], a11 = A[i_s+1, j_s+1]
        # b00 = B[j_s, k_s], b01 = B[j_s, k_s+1], b10 = B[j_s+1, k_s], b11 = B[j_s+1, k_s+1]
        
        # P1 = (a00 + a11)(b00 + b11)
        U[i_s*m + j_s][col] = 1
        U[(i_s+1)*m + (j_s+1)][col] = 1
        V[j_s*p + k_s][col] = 1
        V[(j_s+1)*p + (k_s+1)][col] = 1
        W[k_s*n + i_s][col] = 1 # M11
        W[(k_s+1)*n + (i_s+1)][col] = 1 # M22
        col += 1
        
        # P2 = (a10 + a11)b00
        U[(i_s+1)*m + j_s][col] = 1
        U[(i_s+1)*m + (j_s+1)][col] = 1
        V[j_s*p + k_s][col] = 1
        W[k_s*n + (i_s+1)][col] = 1 # M21
        col += 1
        
        # P3 = a00(b01 - b11)
        U[i_s*m + j_s][col] = 1
        V[j_s*p + (k_s+1)][col] = 1
        V[(j_s+1)*p + (k_s+1)][col] = -1
        W[(k_s+1)*n + i_s][col] = 1 # M12
        W[(k_s+1)*n + (i_s+1)][col] = 1 # M22
        col += 1
        
        # P4 = a11(b10 - b00)
        U[(i_s+1)*m + (j_s+1)][col] = 1
        V[(j_s+1)*p + k_s][col] = 1
        V[j_s*p + k_s][col] = -1
        W[k_s*n + i_s][col] = 1 # M11
        W[k_s*n + (i_s+1)][col] = 1 # M21
        col += 1
        
        # P5 = (a00 + a01)b11
        U[i_s*m + j_s][col] = 1
        U[i_s*m + (j_s+1)][col] = 1
        V[(j_s+1)*p + (k_s+1)][col] = 1
        W[k_s*n + i_s][col] = -1 # M11
        W[(k_s+1)*n + i_s][col] = 1 # M12
        col += 1
        
        # P6 = (a10 - a00)(b00 + b01)
        U[(i_s+1)*m + j_s][col] = 1
        U[i_s*m + j_s][col] = -1
        V[j_s*p + k_s][col] = 1
        V[j_s*p + (k_s+1)][col] = 1
        W[(k_s+1)*n + (i_s+1)][col] = 1 # M22
        col += 1
        
        # P7 = (a01 - a11)(b10 + b11)
        U[i_s*m + (j_s+1)][col] = 1
        U[(i_s+1)*m + (j_s+1)][col] = -1
        V[(j_s+1)*p + k_s][col] = 1
        V[(j_s+1)*p + (k_s+1)][col] = 1
        W[k_s*n + i_s][col] = 1 # M11
        W[(k_s+1)*n + (i_s+1)][col] = -1 # M22
        col += 1

    # Helper to add 2x2x1 block
    def add_2x2x1(i_s, i_e, j_s, j_e, k_s):
        nonlocal col
        # C[i_s, k_s] = A[i_s, j_s]B[j_s, k_s] + A[i_s, j_s+1]B[j_s+1, k_s]
        # C[i_s+1, k_s] = A[i_s+1, j_s]B[j_s, k_s] + A[i_s+1, j_s+1]B[j_s+1, k_s]
        
        # T1 = A[i_s, j_s] B[j_s, k_s]
        U[i_s*m + j_s][col] = 1
        V[j_s*p + k_s][col] = 1
        W[k_s*n + i_s][col] = 1
        col += 1
        
        # T2 = A[i_s, j_s+1] B[j_s+1, k_s]
        U[i_s*m + (j_s+1)][col] = 1
        V[(j_s+1)*p + k_s][col] = 1
        W[k_s*n + i_s][col] = 1
        col += 1
        
        # T3 = A[i_s+1, j_s] B[j_s, k_s]
        U[(i_s+1)*m + j_s][col] = 1
        V[j_s*p + k_s][col] = 1
        W[k_s*n + (i_s+1)][col] = 1
        col += 1
        
        # T4 = A[i_s+1, j_s+1] B[j_s+1, k_s]
        U[(i_s+1)*m + (j_s+1)][col] = 1
        V[(j_s+1)*p + k_s][col] = 1
        W[k_s*n + (i_s+1)][col] = 1
        col += 1

    # Block 1: Strassen for A[:2, :2] * B[:2, :2]
    add_strassen(0, 1, 0, 1, 0, 1)
    
    # Block 2: Strassen for A[:2, :2] * B[:2, 2:4]
    add_strassen(0, 1, 0, 1, 2, 3)
    
    # Block 3: 2x2x1 for A[:2, :2] * B[:2, 4]
    add_2x2x1(0, 1, 0, 1, 4)
    
    # Block 4: Schoolbook for A[:2, 2] * B[2, :5]
    # A col 2, B row 2
    for i in range(2):
        for k in range(5):
            U[i*m + 2][col] = 1
            V[2*p + k][col] = 1
            W[k*n + i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
