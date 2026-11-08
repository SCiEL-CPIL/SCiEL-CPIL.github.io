import numpy as np

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 15
    rank = 15
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a Strassen block
    # i_offset, j_offset, k_offset are global offsets
    # s_offset is the column offset in U, V, W
    def add_strassen(i_off, j_off, k_off, s_off):
        # Strassen products for 2x2x2
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00B10
        # M4 = A11B11
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # Coefficients for A (U)
        # Rows are (i, j)
        # Local i in {0,1}, j in {0,1}
        # Global i = i_off + i_loc, j = j_off + j_loc
        # U index = (i_off + i_loc)*m + (j_off + j_loc)
        
        # Coefficients for B (V)
        # Rows are (j, k)
        # Local j in {0,1}, k in {0,1}
        # Global j = j_off + j_loc, k = k_off + k_loc
        # V index = (j_off + j_loc)*p + (k_off + k_loc)
        
        # Coefficients for C (W)
        # Rows are (k, i)
        # Local k in {0,1}, i in {0,1}
        # Global k = k_off + k_loc, i = i_off + i_loc
        # W index = (k_off + k_loc)*n + (i_off + i_loc)

        # M1
        s = s_off
        U[(i_off+0)*m + (j_off+0)][s] += 1
        U[(i_off+1)*m + (j_off+1)][s] += 1
        V[(j_off+0)*p + (k_off+0)][s] += 1
        V[(j_off+1)*p + (k_off+1)][s] += 1
        W[(k_off+0)*n + (i_off+0)][s] += 1
        W[(k_off+1)*n + (i_off+1)][s] += 1

        # M2
        s = s_off + 1
        U[(i_off+1)*m + (j_off+0)][s] += 1
        U[(i_off+1)*m + (j_off+1)][s] += 1
        V[(j_off+0)*p + (k_off+0)][s] += 1
        W[(k_off+0)*n + (i_off+1)][s] += 1
        W[(k_off+1)*n + (i_off+1)][s] += -1

        # M3
        s = s_off + 2
        U[(i_off+0)*m + (j_off+0)][s] += 1
        V[(j_off+1)*p + (k_off+0)][s] += 1
        W[(k_off+0)*n + (i_off+0)][s] += 1
        W[(k_off+1)*n + (i_off+1)][s] += 1

        # M4
        s = s_off + 3
        U[(i_off+1)*m + (j_off+1)][s] += 1
        V[(j_off+1)*p + (k_off+1)][s] += 1
        W[(k_off+0)*n + (i_off+0)][s] += 1
        W[(k_off+0)*n + (i_off+1)][s] += 1

        # M5
        s = s_off + 4
        U[(i_off+0)*m + (j_off+0)][s] += 1
        U[(i_off+0)*m + (j_off+1)][s] += 1
        V[(j_off+1)*p + (k_off+1)][s] += 1
        W[(k_off+0)*n + (i_off+0)][s] += -1
        W[(k_off+0)*n + (i_off+1)][s] += 1

        # M6
        s = s_off + 5
        U[(i_off+1)*m + (j_off+0)][s] += 1
        U[(i_off+0)*m + (j_off+0)][s] += -1
        V[(j_off+0)*p + (k_off+0)][s] += 1
        V[(j_off+0)*p + (k_off+1)][s] += 1
        W[(k_off+1)*n + (i_off+1)][s] += 1

        # M7
        s = s_off + 6
        U[(i_off+0)*m + (j_off+1)][s] += 1
        U[(i_off+1)*m + (j_off+1)][s] += -1
        V[(j_off+1)*p + (k_off+0)][s] += 1
        V[(j_off+1)*p + (k_off+1)][s] += 1
        W[(k_off+0)*n + (i_off+0)][s] += 1

    # Part 1: A[:,0] B[0,:]
    # i in {0,1}, j=0, k in {0,1,2,3}
    # Rank 1
    s = 0
    U[0][s] = 1
    U[1][s] = 1
    V[0][s] = 1
    V[1][s] = 1
    V[2][s] = 1
    V[3][s] = 1
    W[0][s] = 1
    W[1][s] = 1
    W[2][s] = 1
    W[3][s] = 1
    W[4][s] = 1
    W[5][s] = 1
    W[6][s] = 1
    W[7][s] = 1

    # Part 2a: A[:,1:3] B[1:3, 0:2]
    # i_off=0, j_off=1, k_off=0
    # s_off=1
    add_strassen(0, 1, 0, 1)

    # Part 2b: A[:,1:3] B[1:3, 2:4]
    # i_off=0, j_off=1, k_off=2
    # s_off=8
    add_strassen(0, 1, 2, 8)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
