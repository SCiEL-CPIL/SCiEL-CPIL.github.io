import time
import numpy as np

DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal (rank 11) exact tensor decomposition for
    multiplying a 2x2 matrix by a 2x3 matrix.
    
    Uses Strassen-style bilinear forms for the 2x2 block (rank 7)
    combined with schoolbook multiplication for the third column (rank 4),
    totaling rank 11 which is the known minimum.
    
    Output factor W follows the convention W[k*n + i, r].
    """
    start_time = time.time()
    
    n, m, p = DIMENSIONS
    rank = 11
    
    # Initialize matrices with zeros
    # U: (n*m, rank) -> 4x11
    # V: (m*p, rank) -> 6x11
    # W: (p*n, rank) -> 6x11
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Helper functions: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    def set_u(i, j, r, val):
        U[i * m + j, r] = val

    def set_v(j, k, r, val):
        V[j * p + k, r] = val

    def set_w(k, i, r, val):
        W[k * n + i, r] = val

    # --- Strassen-style decomposition for the 2x2 x 2x2 block (terms 0-6) ---
    # These 7 terms handle C[0..1][0..1] = A[0..1][0..1] * B[0..1][0..1]
    
    # Term 0: (A00+A11)(B00+B11)
    set_u(0, 0, 0, 1); set_u(1, 1, 0, 1)
    set_v(0, 0, 0, 1); set_v(1, 1, 0, 1)
    set_w(0, 0, 0, 1)   # C00 += 1
    set_w(1, 1, 0, 1)   # C11 += 1

    # Term 1: (A10+A11)B00
    set_u(1, 0, 1, 1); set_u(1, 1, 1, 1)
    set_v(0, 0, 1, 1)
    set_w(0, 1, 1, 1)   # C10 += 1
    set_w(1, 1, 1, -1)  # C11 -= 1

    # Term 2: A00(B01-B11)
    set_u(0, 0, 2, 1)
    set_v(0, 1, 2, 1); set_v(1, 1, 2, -1)
    set_w(1, 0, 2, 1)   # C01 += 1
    set_w(1, 1, 2, 1)   # C11 += 1

    # Term 3: A11(B10-B00)
    set_u(1, 1, 3, 1)
    set_v(1, 0, 3, 1); set_v(0, 0, 3, -1)
    set_w(0, 0, 3, 1)   # C00 += 1
    set_w(0, 1, 3, 1)   # C10 += 1

    # Term 4: (A00+A01)B11
    set_u(0, 0, 4, 1); set_u(0, 1, 4, 1)
    set_v(1, 1, 4, 1)
    set_w(0, 0, 4, -1)  # C00 -= 1
    set_w(1, 0, 4, 1)   # C01 += 1

    # Term 5: (A10-A00)(B00+B01)
    set_u(1, 0, 5, 1); set_u(0, 0, 5, -1)
    set_v(0, 0, 5, 1); set_v(0, 1, 5, 1)
    set_w(1, 1, 5, 1)   # C11 += 1

    # Term 6: (A01-A11)(B10+B11)
    set_u(0, 1, 6, 1); set_u(1, 1, 6, -1)
    set_v(1, 0, 6, 1); set_v(1, 1, 6, 1)
    set_w(0, 0, 6, 1)   # C00 += 1

    # --- Schoolbook terms for the third column (terms 7-10) ---
    # These 4 terms handle C[0..1][2] = A[0..1][0..1] * B[0..1][2]
    
    # Term 7: A00 * B02
    set_u(0, 0, 7, 1)
    set_v(0, 2, 7, 1)
    set_w(2, 0, 7, 1)   # C02 += 1

    # Term 8: A01 * B12
    set_u(0, 1, 8, 1)
    set_v(1, 2, 8, 1)
    set_w(2, 0, 8, 1)   # C02 += 1

    # Term 9: A10 * B02
    set_u(1, 0, 9, 1)
    set_v(0, 2, 9, 1)
    set_w(2, 1, 9, 1)   # C12 += 1

    # Term 10: A11 * B12
    set_u(1, 1, 10, 1)
    set_v(1, 2, 10, 1)
    set_w(2, 1, 10, 1)  # C12 += 1

    # Deadline guard (construction is O(1), this is purely safety)
    if time.time() - start_time > 140.0:
        pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
