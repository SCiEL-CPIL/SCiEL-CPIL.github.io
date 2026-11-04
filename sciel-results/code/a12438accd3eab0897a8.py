"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import time

def construct():
    n, m, p = DIMENSIONS
    rank = 68
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen blocks for A[0:2, 0:2] * B[0:2, :]
    # 4 blocks of 2x2x2
    strassen_terms = [
        # M1: (A00+A11)(B00+B11)
        # U: A00=1, A11=1. V: B00=1, B11=1. W: C00=1, C01=-1, C10=-1, C11=1
        # Indices:
        # U: i,j. V: j,k. W: k,i.
        # Block c handles B cols 2c, 2c+1.
        # A cols 0,1. B rows 0,1.
        #
        # Term 1 (M1)
        { 'U': [(0,0,1), (1,1,1)], 'V': [(0,0,1), (1,1,1)], 'W': [(0,0,1), (1,0,-1), (0,1,-1), (1,1,1)] },
        # Term 2 (M2): (A10+A11)B00
        { 'U': [(1,0,1), (1,1,1)], 'V': [(0,0,1)], 'W': [(0,1,1), (1,1,-1)] }, # C10 += M2, C11 -= M2
        # Term 3 (M3): A00(B10-B11)
        { 'U': [(0,0,1)], 'V': [(1,0,1), (1,1,-1)], 'W': [(0,0,1), (1,0,1)] }, # C00 += M3, C01 += M3
        # Term 4 (M4): A11(B01-B00)
        { 'U': [(1,1,1)], 'V': [(0,1,1), (0,0,-1)], 'W': [(0,0,-1), (1,0,1)] }, # C00 -= M4, C10 += M4
        # Term 5 (M5): (A00+A01)B11
        { 'U': [(0,0,1), (0,1,1)], 'V': [(1,1,1)], 'W': [(0,0,1), (0,1,1)] }, # C00 += M5, C01 += M5
        # Term 6 (M6): (A10-A00)(B00+B01)
        { 'U': [(1,0,1), (0,0,-1)], 'V': [(0,0,1), (0,1,1)], 'W': [(1,1,1)] }, # C11 += M6
        # Term 7 (M7): (A01-A11)(B10+B11)
        { 'U': [(0,1,1), (1,1,-1)], 'V': [(1,0,1), (1,1,1)], 'W': [(0,0,-1), (0,1,1)] }, # C00 -= M7, C01 += M7
    ]
    
    for c in range(4):
        for term in strassen_terms:
            # Apply offsets
            # A cols are 0,1. B rows 0,1.
            # B cols are 2c, 2c+1.
            # C cols are 2c, 2c+1.
            
            for i, j, val in term['U']:
                U[i * m + j][col] += val
            for j, k, val in term['V']:
                # k is local col index 0 or 1
                global_k = 2 * c + k
                V[j * p + global_k][col] += val
            for k, i, val in term['W']:
                # k is local col index 0 or 1
                global_k = 2 * c + k
                W[global_k * n + i][col] += val
            col += 1

    # Schoolbook for A[0:2, 2] * B[2, :]
    # i in 0,1; k in 0..7
    for i in range(2):
        for k in range(8):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1

    # Schoolbook for A[2, 0:2] * B[0:2, :]
    # j in 0,1; k in 0..7
    for j in range(2):
        for k in range(8):
            U[2 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 2][col] = 1
            col += 1

    # Schoolbook for A[2, 2] * B[2, :]
    # k in 0..7
    for k in range(8):
        U[2 * m + 2][col] = 1
        V[2 * p + k][col] = 1
        W[k * n + 2][col] = 1
        col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
