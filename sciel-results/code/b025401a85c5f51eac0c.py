"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    # Rank 18 decomposition
    # 2x2x2 (Strassen, rank 7) + 2x2x2 (Strassen, rank 7) + 2x2x1 (Schoolbook, rank 4)
    rank = 18
    
    U = [[0] * rank for _ in range(n * m)] # 4 x 18
    V = [[0] * rank for _ in range(m * p)] # 10 x 18
    W = [[0] * rank for _ in range(p * n)] # 10 x 18

    # Helper to add a Strassen block
    # u_base: index of first row in U (0)
    # v_base: index of first row in V (0 or 4)
    # w_base: index of first row in W (0 or 4)
    # col_offset: starting column index in U, V, W
    
    def add_strassen(u_base, v_base, w_base, col_offset):
        # M1 = (a+d)(e+h)
        U[u_base + 0][col_offset + 0] = 1
        U[u_base + 3][col_offset + 0] = 1
        V[v_base + 0][col_offset + 0] = 1
        V[v_base + 3][col_offset + 0] = 1
        W[w_base + 0][col_offset + 0] = 1 # C11
        W[w_base + 3][col_offset + 0] = 1 # C22

        # M2 = (c+d)e
        U[u_base + 2][col_offset + 1] = 1
        U[u_base + 3][col_offset + 1] = 1
        V[v_base + 0][col_offset + 1] = 1
        W[w_base + 1][col_offset + 1] = 1 # C21
        W[w_base + 3][col_offset + 1] = -1 # C22

        # M3 = a(f-h)
        U[u_base + 0][col_offset + 2] = 1
        V[v_base + 1][col_offset + 2] = 1
        V[v_base + 3][col_offset + 2] = -1
        W[w_base + 2][col_offset + 2] = 1 # C12
        W[w_base + 3][col_offset + 2] = 1 # C22

        # M4 = d(g-e)
        U[u_base + 3][col_offset + 3] = 1
        V[v_base + 2][col_offset + 3] = 1
        V[v_base + 0][col_offset + 3] = -1
        W[w_base + 0][col_offset + 3] = 1 # C11
        W[w_base + 1][col_offset + 3] = 1 # C21

        # M5 = (a+b)h
        U[u_base + 0][col_offset + 4] = 1
        U[u_base + 1][col_offset + 4] = 1
        V[v_base + 3][col_offset + 4] = 1
        W[w_base + 0][col_offset + 4] = -1 # C11
        W[w_base + 2][col_offset + 4] = 1 # C12

        # M6 = (c-a)(e+f)
        U[u_base + 2][col_offset + 5] = 1
        U[u_base + 0][col_offset + 5] = -1
        V[v_base + 0][col_offset + 5] = 1
        V[v_base + 1][col_offset + 5] = 1
        W[w_base + 3][col_offset + 5] = 1 # C22

        # M7 = (b-d)(g+h)
        U[u_base + 1][col_offset + 6] = 1
        U[u_base + 3][col_offset + 6] = -1
        V[v_base + 2][col_offset + 6] = 1
        V[v_base + 3][col_offset + 6] = 1
        W[w_base + 0][col_offset + 6] = 1 # C11

    # Block 1: Cols 0,1 of B -> Cols 0,1 of C
    add_strassen(0, 0, 0, 0)
    
    # Block 2: Cols 2,3 of B -> Cols 2,3 of C
    add_strassen(0, 4, 4, 7)
    
    # Block 3: Col 4 of B -> Col 4 of C
    # C04 = A00*B04 + A01*B14
    # C14 = A10*B04 + A11*B14
    # U indices: 0:A00, 1:A01, 2:A10, 3:A11
    # V indices: 8:B04, 9:B14
    # W indices: 8:C04, 9:C14
    
    # Term 1: A00 * B04 -> C04
    U[0][14] = 1
    V[8][14] = 1
    W[8][14] = 1
    
    # Term 2: A01 * B14 -> C04
    U[1][15] = 1
    V[9][15] = 1
    W[8][15] = 1
    
    # Term 3: A10 * B04 -> C14
    U[2][16] = 1
    V[8][16] = 1
    W[9][16] = 1
    
    # Term 4: A11 * B14 -> C14
    U[3][17] = 1
    V[9][17] = 1
    W[9][17] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
