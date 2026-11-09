"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We construct a decomposition of rank 24.
    # Split j into {0,1} and {2}.
    # Split k into {0,1}, {2,3}, {4,5}, {6}.
    
    rank = 24
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen 2x2x2 decomposition for specific i,j,k ranges
    # i in {i0, i0+1}, j in {j0, j0+1}, k in {k0, k0+1}
    def add_strassen(i0, j0, k0):
        nonlocal col
        # M1 = (A[i0,j0]+A[i0+1,j0+1])(B[j0+1,k0]-B[j0+1,k0+1])
        # Note: Standard Strassen indices might vary. 
        # Let's use a known valid rank 7 decomposition for 2x2x2.
        # T_ijk = sum_c U_ijk,c V_jk,c W_ki,c
        
        # 1
        U[i0*m + j0][col] = 1
        U[(i0+1)*m + (j0+1)][col] = 1
        V[(j0+1)*p + k0][col] = -1
        V[(j0+1)*p + (k0+1)][col] = 1
        W[k0*n + i0][col] = 1
        W[(k0+1)*n + (i0+1)][col] = 1
        col += 1
        
        # 2
        U[i0*m + j0][col] = 1
        U[i0*m + (j0+1)][col] = 1
        V[(j0+1)*p + (k0+1)][col] = 1
        W[k0*n + i0][col] = 1
        W[(k0+1)*n + i0][col] = 1
        col += 1
        
        # 3
        U[i0*m + j0][col] = 1
        V[j0*p + k0][col] = 1
        W[k0*n + i0][col] = 1
        W[(k0+1)*n + i0][col] = -1
        col += 1
        
        # 4
        U[(i0+1)*m + (j0+1)][col] = 1
        V[(j0+1)*p + (k0+1)][col] = 1
        W[k0*n + (i0+1)][col] = 1
        W[(k0+1)*n + (i0+1)][col] = -1
        col += 1
        
        # 5
        U[(i0+1)*m + j0][col] = 1
        U[i0*m + j0][col] = 1
        V[j0*p + k0][col] = 1
        W[k0*n + (i0+1)][col] = 1
        W[(k0+1)*n + i0][col] = 1
        col += 1
        
        # 6
        U[(i0+1)*m + (j0+1)][col] = 1
        U[(i0+1)*m + j0][col] = -1
        V[(j0+1)*p + (k0+1)][col] = 1
        V[(j0+1)*p + k0][col] = 1
        W[k0*n + (i0+1)][col] = 1
        W[(k0+1)*n + (i0+1)][col] = 1
        col += 1
        
        # 7
        U[i0*m + (j0+1)][col] = 1
        U[(i0+1)*m + (j0+1)][col] = -1
        V[j0*p + k0][col] = 1
        V[j0*p + (k0+1)][col] = 1
        W[k0*n + i0][col] = 1
        W[(k0+1)*n + i0][col] = 1
        col += 1

    # Helper to add a 2x2x1 decomposition (rank 2)
    # i in {i0, i0+1}, j in {j0, j0+1}, k = k0
    def add_2x2x1(i0, j0, k0):
        nonlocal col
        # C_ik = A_i0 B_j0k + A_i1 B_j1k
        # Term 1: (A_i0,j0 + A_i1,j0) * B_j0,k0
        U[i0*m + j0][col] = 1
        U[(i0+1)*m + j0][col] = 1
        V[j0*p + k0][col] = 1
        W[k0*n + i0][col] = 1
        W[k0*n + (i0+1)][col] = 1
        col += 1
        
        # Term 2: (A_i0,j1 + A_i1,j1) * B_j1,k0
        U[i0*m + (j0+1)][col] = 1
        U[(i0+1)*m + (j0+1)][col] = 1
        V[(j0+1)*p + k0][col] = 1
        W[k0*n + i0][col] = 1
        W[k0*n + (i0+1)][col] = 1
        col += 1

    # Helper to add a 2x1x1 decomposition (rank 1)
    # i in {i0, i0+1}, j = j0, k = k0
    def add_2x1x1(i0, j0, k0):
        nonlocal col
        U[i0*m + j0][col] = 1
        U[(i0+1)*m + j0][col] = 1
        V[j0*p + k0][col] = 1
        W[k0*n + i0][col] = 1
        W[k0*n + (i0+1)][col] = 1
        col += 1

    # Part 1: j in {0,1}
    # k blocks: {0,1}, {2,3}, {4,5}, {6}
    add_strassen(0, 0, 0)
    add_strassen(0, 0, 2)
    add_strassen(0, 0, 4)
    add_2x2x1(0, 0, 6)
    
    # Part 2: j = 2
    # k blocks: {0,1}, {2,3}, {4,5}, {6}
    # For each k, add 2x1x1
    add_2x1x1(0, 2, 0)
    add_2x1x1(0, 2, 1)
    add_2x1x1(0, 2, 2)
    add_2x1x1(0, 2, 3)
    add_2x1x1(0, 2, 4)
    add_2x1x1(0, 2, 5)
    add_2x1x1(0, 2, 6)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
