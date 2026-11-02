import time
import sys

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 37
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen decomposition for the 2x2 block part (j in {0,1}, k in blocks of 2)
    # There are 5 blocks of size 2 in the k dimension (p=10)
    # Each block uses 7 Strassen products
    for l in range(5):
        k_base = 2 * l
        for r in range(7):
            R = l * 7 + r
            
            if r == 0:
                # P0 = (A00 + A11)(B00 + B11)
                U[0 * m + 0][R] += 1
                U[1 * m + 1][R] += 1
                V[0 * p + k_base][R] += 1
                V[1 * p + k_base + 1][R] += 1
                W[k_base * n + 0][R] += 1
                W[(k_base + 1) * n + 1][R] += 1
            elif r == 1:
                # P1 = (A10 + A11)B00
                U[1 * m + 0][R] += 1
                U[1 * m + 1][R] += 1
                V[0 * p + k_base][R] += 1
                W[k_base * n + 1][R] += 1
                W[(k_base + 1) * n + 1][R] += -1
            elif r == 2:
                # P2 = A00(B01 - B11)
                U[0 * m + 0][R] += 1
                V[0 * p + k_base + 1][R] += 1
                V[1 * p + k_base + 1][R] += -1
                W[(k_base + 1) * n + 0][R] += 1
                W[(k_base + 1) * n + 1][R] += 1
            elif r == 3:
                # P3 = A11(B10 - B00)
                U[1 * m + 1][R] += 1
                V[1 * p + k_base][R] += 1
                V[0 * p + k_base][R] += -1
                W[k_base * n + 0][R] += 1
                W[k_base * n + 1][R] += 1
            elif r == 4:
                # P4 = (A00 + A01)B11
                U[0 * m + 0][R] += 1
                U[0 * m + 1][R] += 1
                V[1 * p + k_base + 1][R] += 1
                W[k_base * n + 0][R] += -1
                W[(k_base + 1) * n + 0][R] += 1
            elif r == 5:
                # P5 = (A10 - A00)(B00 + B01)
                U[1 * m + 0][R] += 1
                U[0 * m + 0][R] += -1
                V[0 * p + k_base][R] += 1
                V[0 * p + k_base + 1][R] += 1
                W[(k_base + 1) * n + 1][R] += 1
            elif r == 6:
                # P6 = (A01 - A11)(B10 + B11)
                U[0 * m + 1][R] += 1
                U[1 * m + 1][R] += -1
                V[1 * p + k_base][R] += 1
                V[1 * p + k_base + 1][R] += 1
                W[k_base * n + 0][R] += 1

    # Handle the remaining j=2 part (A is 2x1, B is 1x10)
    # This is rank 2
    # R=35 for i=0
    R = 35
    U[0 * m + 2][R] = 1
    for k in range(p):
        V[2 * p + k][R] = 1
        W[k * n + 0][R] = 1
        
    # R=36 for i=1
    R = 36
    U[1 * m + 2][R] = 1
    for k in range(p):
        V[2 * p + k][R] = 1
        W[k * n + 1][R] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
