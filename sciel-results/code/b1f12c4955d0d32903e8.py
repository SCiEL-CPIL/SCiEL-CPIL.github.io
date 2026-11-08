import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use Strassen's algorithm on 2x2 blocks.
    # 6x6 matrix is divided into 3x3 blocks of 2x2 matrices.
    # Block indices I, J, L in {0, 1, 2}
    # Local indices r in {0, 1}
    
    rank = 189
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for I in range(3):
        for J in range(3):
            for L in range(3):
                # Global indices for A block (I, J)
                # A[k, j] where k = 2I + rk, j = 2J + rj
                a_00 = (2 * I) * m + (2 * J)
                a_01 = (2 * I) * m + (2 * J + 1)
                a_10 = (2 * I + 1) * m + (2 * J)
                a_11 = (2 * I + 1) * m + (2 * J + 1)
                
                # Global indices for B block (J, L)
                # B[j, i] where j = 2J + rj, i = 2L + ri
                b_00 = (2 * J) * p + (2 * L)
                b_01 = (2 * J) * p + (2 * L + 1)
                b_10 = (2 * J + 1) * p + (2 * L)
                b_11 = (2 * J + 1) * p + (2 * L + 1)
                
                # Global indices for C block (I, L)
                # C[k, i] where k = 2I + rk, i = 2L + ri
                c_00 = (2 * I) * n + (2 * L)
                c_01 = (2 * I) * n + (2 * L + 1)
                c_10 = (2 * I + 1) * n + (2 * L)
                c_11 = (2 * I + 1) * n + (2 * L + 1)
                
                # Strassen's 7 products
                # M1 = (A00 + A11)(B00 + B11)
                U[a_00, col] += 1; U[a_11, col] += 1
                V[b_00, col] += 1; V[b_11, col] += 1
                W[c_00, col] += 1; W[c_11, col] += 1
                col += 1
                
                # M2 = (A10 + A11)B00
                U[a_10, col] += 1; U[a_11, col] += 1
                V[b_00, col] += 1
                W[c_10, col] += 1; W[c_11, col] -= 1
                col += 1
                
                # M3 = A00(B01 - B11)
                U[a_00, col] += 1
                V[b_01, col] += 1; V[b_11, col] -= 1
                W[c_01, col] += 1; W[c_11, col] += 1
                col += 1
                
                # M4 = A11(B10 - B00)
                U[a_11, col] += 1
                V[b_10, col] += 1; V[b_00, col] -= 1
                W[c_00, col] += 1; W[c_10, col] += 1
                col += 1
                
                # M5 = (A00 + A01)B11
                U[a_00, col] += 1; U[a_01, col] += 1
                V[b_11, col] += 1
                W[c_00, col] -= 1; W[c_01, col] += 1
                col += 1
                
                # M6 = (A10 - A00)(B00 + B01)
                U[a_10, col] += 1; U[a_00, col] -= 1
                V[b_00, col] += 1; V[b_01, col] += 1
                W[c_11, col] += 1
                col += 1
                
                # M7 = (A01 - A11)(B10 + B11)
                U[a_01, col] += 1; U[a_11, col] -= 1
                V[b_10, col] += 1; V[b_11, col] += 1
                W[c_00, col] += 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
