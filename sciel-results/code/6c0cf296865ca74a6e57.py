import numpy as np

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 6 blocks of 2x2x2 Strassen (rank 7 each)
    rank = 42
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # U indices: 0:00, 1:01, 2:10, 3:11
    # V indices for block t: 
    #   b00: 12*0 + 2t = 2t
    #   b01: 12*0 + 2t+1 = 2t+1
    #   b10: 12*1 + 2t = 12+2t
    #   b11: 12*1 + 2t+1 = 13+2t
    # W indices for block t:
    #   c00: 2*(2t) + 0 = 4t
    #   c01: 2*(2t+1) + 0 = 4t+2
    #   c10: 2*(2t) + 1 = 4t+1
    #   c11: 2*(2t+1) + 1 = 4t+3
    
    for t in range(6):
        base_col = t * 7
        k0 = 2 * t
        k1 = 2 * t + 1
        
        # Term 0: M1 = (a00+a11)(b00+b11)
        col = base_col
        U[0][col] += 1
        U[3][col] += 1
        V[2*t][col] += 1
        V[13+2*t][col] += 1
        W[4*t][col] += 1
        W[4*t+3][col] += 1
        
        # Term 1: M2 = (a10+a11)b00
        col = base_col + 1
        U[2][col] += 1
        U[3][col] += 1
        V[2*t][col] += 1
        W[4*t+1][col] += 1
        W[4*t+3][col] += -1
        
        # Term 2: M3 = a00(b01-b11)
        col = base_col + 2
        U[0][col] += 1
        V[2*t+1][col] += 1
        V[14+2*t][col] += -1
        W[4*t+2][col] += 1
        W[4*t+3][col] += 1
        
        # Term 3: M4 = a11(b10-b00)
        col = base_col + 3
        U[3][col] += 1
        V[12+2*t][col] += 1
        V[2*t][col] += -1
        W[4*t][col] += 1
        W[4*t+1][col] += 1
        
        # Term 4: M5 = (a00+a01)b11
        col = base_col + 4
        U[0][col] += 1
        U[1][col] += 1
        V[13+2*t][col] += 1
        W[4*t][col] += -1
        W[4*t+2][col] += 1
        
        # Term 5: M6 = (a10-a00)(b00+b01)
        col = base_col + 5
        U[2][col] += 1
        U[0][col] += -1
        V[2*t][col] += 1
        V[2*t+1][col] += 1
        W[4*t+3][col] += 1
        
        # Term 6: M7 = (a01-a11)(b10+b11)
        col = base_col + 6
        U[1][col] += 1
        U[3][col] += -1
        V[12+2*t][col] += 1
        V[13+2*t][col] += 1
        W[4*t][col] += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
