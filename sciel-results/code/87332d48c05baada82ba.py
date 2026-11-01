import numpy as np

DIMENSIONS = [2, 2, 6]

def construct():
    n, m, p = DIMENSIONS
    # Rank 21: 3 blocks of Strassen
    rank = 21
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # M1: (a00+a11)(b00+b11)
    # M2: (a10+a11)b00
    # M3: a00(b01-b11)
    # M4: a11(b10-b00)
    # M5: (a00+a01)b11
    # M6: (a10-a00)(b00+b01)
    # M7: (a01-a11)(b10+b11)
    
    # A indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # B indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # C indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    
    # Block 0: B cols 0,1. C cols 0,1.
    # Block 1: B cols 2,3. C cols 2,3.
    # Block 2: B cols 4,5. C cols 4,5.
    
    for b in range(3):
        base = b * 7
        # M1
        U[0][base] = 1; U[3][base] = 1
        V[b*4 + 0][base] = 1; V[b*4 + 3][base] = 1
        W[b*2 + 0][base] = 1; W[b*2 + 1][base] = 1; W[b*2 + 2][base] = -1; W[b*2 + 3][base] = 1
        # M2
        U[2][base+1] = 1; U[3][base+1] = 1
        V[b*4 + 0][base+1] = 1
        W[b*2 + 2][base+1] = 1; W[b*2 + 3][base+1] = -1
        # M3
        U[0][base+2] = 1
        V[b*4 + 1][base+2] = 1; V[b*4 + 3][base+2] = -1
        W[b*2 + 1][base+2] = 1; W[b*2 + 3][base+2] = 1
        # M4
        U[3][base+3] = 1
        V[b*4 + 2][base+3] = 1; V[b*4 + 0][base+3] = -1
        W[b*2 + 0][base+3] = 1; W[b*2 + 2][base+3] = 1
        # M5
        U[0][base+4] = 1; U[1][base+4] = 1
        V[b*4 + 3][base+4] = 1
        W[b*2 + 0][base+4] = -1; W[b*2 + 1][base+4] = 1
        # M6
        U[2][base+5] = 1; U[0][base+5] = -1
        V[b*4 + 0][base+5] = 1; V[b*4 + 1][base+5] = 1
        W[b*2 + 3][base+5] = 1
        # M7
        U[1][base+6] = 1; U[3][base+6] = -1
        V[b*4 + 2][base+6] = 1; V[b*4 + 3][base+6] = 1
        W[b*2 + 0][base+6] = 1; W[b*2 + 1][base+6] = -1

    return {"U": U, "V": V, "W": W}
