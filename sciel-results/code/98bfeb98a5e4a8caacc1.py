"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 2]
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-7 exact tensor decomposition for 2x2 matrix multiplication.
    Uses Strassen's algorithm decomposition, which is optimal (rank 7 is the
    proven minimum rank for 2x2 matrix multiplication over the reals).
    """
    n, m, p = DIMENSIONS  # 2, 2, 2
    rank = 7

    # Strassen's 7 products:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)

    # U shape (n*m, rank) -> (4, 7)
    # Indexing: U[i*m + j, c] is the coefficient of A[i,j] in the c-th product
    U_data = [
        [1, 0, 1, 0, 1, -1, 0],  # A[0,0]
        [0, 0, 0, 0, 1, 0, 1],   # A[0,1]
        [0, 1, 0, 0, 0, 1, 0],   # A[1,0]
        [1, 1, 0, 1, 0, 0, -1]   # A[1,1]
    ]

    # V shape (m*p, rank) -> (4, 7)
    # Indexing: V[j*p + k, c] is the coefficient of B[j,k] in the c-th product
    V_data = [
        [1, 1, 0, -1, 0, 1, 0],  # B[0,0]
        [0, 0, 1, 0, 0, 1, 0],   # B[0,1]
        [0, 0, 0, 1, 0, 0, 1],   # B[1,0]
        [1, 0, -1, 0, 1, 0, 1]   # B[1,1]
    ]

    # W shape (p*n, rank) -> (4, 7)
    # Indexing: W[k*n + i, c] is the coefficient for C[i,k] from the c-th product
    # C[i,k] = sum_j A[i,j] B[j,k].
    # In Strassen:
    # C[0,0] = M1 + M4 - M5 + M7
    # C[1,0] = M2 + M4
    # C[0,1] = M3 + M5
    # C[1,1] = M1 - M2 + M3 + M6

    # Mapping W rows (index = k*n + i):
    # Row 0 (k=0, i=0): C[0,0] -> [1, 0, 0, 1, -1, 0, 1]
    # Row 1 (k=0, i=1): C[1,0] -> [0, 1, 0, 1, 0, 0, 0]
    # Row 2 (k=1, i=0): C[0,1] -> [0, 0, 1, 0, 1, 0, 0]
    # Row 3 (k=1, i=1): C[1,1] -> [1, -1, 1, 0, 0, 1, 0]

    W_data = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]

    U = np.array(U_data, dtype=float)
    V = np.array(V_data, dtype=float)
    W = np.array(W_data, dtype=float)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
