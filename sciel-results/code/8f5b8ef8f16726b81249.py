import time

DIMENSIONS = [2, 2, 6]

def construct():
    n, m, p = DIMENSIONS
    # Rank 21 decomposition using 3 independent Strassen blocks
    rank = 21
    U = [[0] * rank for _ in range(n * m)]  # 4 x 21
    V = [[0] * rank for _ in range(m * p)]  # 12 x 21
    W = [[0] * rank for _ in range(p * n)]  # 12 x 21
    
    # Strassen indices for A (2x2):
    # A00 -> 0, A01 -> 1, A10 -> 2, A11 -> 3
    
    # For each of the 3 blocks of B (2x2 each)
    for t in range(3):
        # B indices for this block:
        # B00 -> 2t
        # B01 -> 2t + 1
        # B10 -> 6 + 2t
        # B11 -> 6 + 2t + 1
        
        # C indices for this block:
        # C00 (i=0, l=2t) -> 4t
        # C01 (i=0, l=2t+1) -> 4t + 1
        # C10 (i=1, l=2t) -> 4t + 2
        # C11 (i=1, l=2t+1) -> 4t + 3
        
        base_col = 7 * t
        
        # M1 = (A00 + A11)(B00 + B11)
        c = base_col + 0
        U[0][c] = 1
        U[3][c] = 1
        V[2*t][c] = 1
        V[6 + 2*t + 1][c] = 1
        W[4*t][c] = 1
        W[4*t + 3][c] = 1

        # M2 = (A10 + A11)B00
        c = base_col + 1
        U[2][c] = 1
        U[3][c] = 1
        V[2*t][c] = 1
        W[4*t + 2][c] = 1
        W[4*t + 3][c] = -1

        # M3 = A00(B01 - B11)
        c = base_col + 2
        U[0][c] = 1
        V[2*t + 1][c] = 1
        V[6 + 2*t + 1][c] = -1
        W[4*t + 1][c] = 1
        W[4*t + 3][c] = 1

        # M4 = A11(B10 - B00)
        c = base_col + 3
        U[3][c] = 1
        V[6 + 2*t][c] = 1
        V[2*t][c] = -1
        W[4*t][c] = 1
        W[4*t + 2][c] = 1

        # M5 = (A00 + A01)B11
        c = base_col + 4
        U[0][c] = 1
        U[1][c] = 1
        V[6 + 2*t + 1][c] = 1
        W[4*t][c] = -1
        W[4*t + 1][c] = 1

        # M6 = (A10 - A00)(B00 + B01)
        c = base_col + 5
        U[2][c] = 1
        U[0][c] = -1
        V[2*t][c] = 1
        V[2*t + 1][c] = 1
        W[4*t + 3][c] = 1

        # M7 = (A01 - A11)(B10 + B11)
        c = base_col + 6
        U[1][c] = 1
        U[3][c] = -1
        V[6 + 2*t][c] = 1
        V[6 + 2*t + 1][c] = 1
        W[4*t][c] = 1

    return {"U": U, "V": V, "W": W}
