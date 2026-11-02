"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = 2, 2, 16
    nb = p // m  # 8 column-pairs in B
    r = nb * 7   # rank = 56

    U = [[0] * r for _ in range(n * m)]   # 4 x 56
    V = [[0] * r for _ in range(m * p)]   # 32 x 56
    W = [[0] * r for _ in range(p * n)]   # 32 x 56

    for t in range(nb):
        base = t * 7
        # B row-0 cols 2t, 2t+1 -> indices 2t, 2t+1
        # B row-1 cols 2t, 2t+1 -> indices 16+2t, 16+2t+1
        bA = 2 * t          # B[0, 2t]
        bB = 2 * t + 1      # B[0, 2t+1]
        bC = 16 + 2 * t     # B[1, 2t]
        bD = 16 + 2 * t + 1 # B[1, 2t+1]

        # C row 0 col 2t -> index 2t*2+0 = 4t
        # C row 0 col 2t+1 -> index 4t+1
        # C row 1 col 2t -> index 4t+2
        # C row 1 col 2t+1 -> index 4t+3
        c00 = 4 * t
        c01 = 4 * t + 1
        c10 = 4 * t + 2
        c11 = 4 * t + 3

        # A indices: 0=A[0,0], 1=A[0,1], 2=A[1,0], 3=A[1,1]

        # M1 = (A00+A11)*(B00+B11)
        U[0][base] = 1; U[3][base] = 1
        V[bA][base] = 1; V[bD][base] = 1
        W[c00][base] = 1; W[c11][base] = 1

        # M2 = (A10+A11)*B00
        U[2][base+1] = 1; U[3][base+1] = 1
        V[bA][base+1] = 1
        W[c01][base+1] = 1; W[c11][base+1] = -1

        # M3 = A00*(B01-B11)
        U[0][base+2] = 1
        V[bB][base+2] = 1; V[bD][base+2] = -1
        W[c10][base+2] = 1; W[c11][base+2] = 1

        # M4 = A11*(B10-B00)
        U[3][base+3] = 1
        V[bC][base+3] = 1; V[bA][base+3] = -1
        W[c00][base+3] = 1; W[c01][base+3] = 1

        # M5 = (A00+A01)*B11
        U[0][base+4] = 1; U[1][base+4] = 1
        V[bD][base+4] = 1
        W[c00][base+4] = -1; W[c10][base+4] = 1

        # M6 = (A10-A00)*(B00+B01)
        U[2][base+5] = 1; U[0][base+5] = -1
        V[bA][base+5] = 1; V[bB][base+5] = 1
        W[c11][base+5] = 1

        # M7 = (A01-A11)*(B10+B11)
        U[1][base+6] = 1; U[3][base+6] = -1
        V[bC][base+6] = 1; V[bD][base+6] = 1
        W[c00][base+6] = 1

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
