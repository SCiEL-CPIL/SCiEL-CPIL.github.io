"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145  # 5s safety margin
    
    # Build the known rank-22 decomposition
    U22 = np.zeros((n * m, 22))
    V22 = np.zeros((m * p, 22))
    W22 = np.zeros((p * n, 22))
    
    # Block 1: A_L B_T1 (2x2x2 Strassen)
    col = 0
    # M1: (A00+A11)(B00+B11) -> C00, C11
    U22[0, col] = 1; U22[4, col] = 1
    V22[0, col] = 1; V22[5, col] = 1
    W22[0, col] = 1; W22[3, col] = 1
    col += 1
    # M2: (A10+A11)B00 -> C10, -C11
    U22[3, col] = 1; U22[4, col] = 1
    V22[0, col] = 1
    W22[1, col] = 1; W22[3, col] = -1
    col += 1
    # M3: A00(B01-B11) -> C01, C11
    U22[0, col] = 1
    V22[1, col] = 1; V22[5, col] = -1
    W22[2, col] = 1; W22[3, col] = 1
    col += 1
    # M4: A11(B10-B00) -> C00, C10
    U22[4, col] = 1
    V22[4, col] = 1; V22[0, col] = -1
    W22[0, col] = 1; W22[1, col] = 1
    col += 1
    # M5: (A00+A01)B11 -> -C00, C01
    U22[0, col] = 1; U22[1, col] = 1
    V22[5, col] = 1
    W22[0, col] = -1; W22[2, col] = 1
    col += 1
    # M6: (A10-A00)(B00+B01) -> C11
    U22[3, col] = 1; U22[0, col] = -1
    V22[0, col] = 1; V22[1, col] = 1
    W22[3, col] = 1
    col += 1
    # M7: (A01-A11)(B10+B11) -> C00
    U22[1, col] = 1; U22[4, col] = -1
    V22[4, col] = 1; V22[5, col] = 1
    W22[0, col] = 1
    col += 1
    
    # Block 2: A_L B_T2 (2x2x2 Strassen)
    # M1: (A00+A11)(B02+B13) -> C02, C13
    U22[0, col] = 1; U22[4, col] = 1
    V22[2, col] = 1; V22[7, col] = 1
    W22[4, col] = 1; W22[7, col] = 1
    col += 1
    # M2: (A10+A11)B02 -> C12, -C13
    U22[3, col] = 1; U22[4, col] = 1
    V22[2, col] = 1
    W22[5, col] = 1; W22[7, col] = -1
    col += 1
    # M3: A00(B03-B13) -> C03, C13
    U22[0, col] = 1
    V22[3, col] = 1; V22[7, col] = -1
    W22[6, col] = 1; W22[7, col] = 1
    col += 1
    # M4: A11(B12-B02) -> C02, C12
    U22[4, col] = 1
    V22[6, col] = 1; V22[2, col] = -1
    W22[4, col] = 1; W22[5, col] = 1
    col += 1
    # M5: (A00+A01)B13 -> -C02, C03
    U22[0, col] = 1; U22[1, col] = 1
    V22[7, col] = 1
    W22[4, col] = -1; W22[6, col] = 1
    col += 1
    # M6: (A10-A00)(B02+B03) -> C13
    U22[3, col] = 1; U22[0, col] = -1
    V22[2, col] = 1; V22[3, col] = 1
    W22[7, col] = 1
    col += 1
    # M7: (A01-A11)(B12+B13) -> C02
    U22[1, col] = 1; U22[4, col] = -1
    V22[6, col] = 1; V22[7, col] = 1
    W22[4, col] = 1
    col += 1
    
    # Block 3: A_R B_B (2x1x4 schoolbook)
    # Z1: A02 B20 -> C00
    U22[2, col] = 1; V22[8, col] = 1; W22[0, col] = 1
    col += 1
    # Z2: A02 B21 -> C01
    U22[2, col] = 1; V22[9, col] = 1; W22[2, col] = 1
    col += 1
    # Z3: A02 B22 -> C02
    U22[2, col] = 1; V22[10, col] = 1; W22[4, col] = 1
    col += 1
    # Z4: A02 B23 -> C03
    U22[2, col] = 1; V22[11, col] = 1; W22[6, col] = 1
    col += 1
    # Z5: A12 B20 -> C10
    U22[5, col] = 1; V22[8, col] = 1; W22[1, col] = 1
    col += 1
    # Z6: A12 B21 -> C11
    U22[5, col] = 1; V22[9, col] = 1; W22[3, col] = 1
    col += 1
    # Z7: A12 B22 -> C12
    U22[5, col] = 1; V22[10, col] = 1; W22[5, col] = 1
    col += 1
    # Z8: A12 B23 -> C13
    U22[5, col] = 1; V22[11, col] = 1; W22[7, col] = 1
    col += 1
    
    # Verify rank-22 decomposition
    def verify_decomposition(U, V, W, rank):
        U_3d = U.reshape(n, m, rank)
        V_3d = V.reshape(m, p, rank)
        W_3d = W.reshape(p, n, rank)
        T = np.zeros((n, m, p))
        for r in range(rank):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i,j,k] += U_3d[i,j,r] * V_3d[j,k,r] * W_3d[k,i,r]
        return np.allclose(T, 1.0, atol=1e-10)
    
    if not verify_decomposition(U22, V22, W22, 22):
        # Fallback to schoolbook
        rank_school = n * m * p
        U_school = np.zeros((n * m, rank_school))
        V_school = np.zeros((m * p, rank_school))
        W_school = np.zeros((p * n, rank_school))
        col_school = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i * m + j, col_school] = 1
                    V_school[j * p + k, col_school] = 1
                    W_school[k * n + i, col_school] = 1
                    col_school += 1
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    
    # Return the verified rank-22 decomposition
    return {"U": U22.tolist(), "V": V22.tolist(), "W": W22.tolist()}
# EVOLVE-BLOCK-END
