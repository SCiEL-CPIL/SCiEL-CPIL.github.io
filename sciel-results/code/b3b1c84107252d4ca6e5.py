import numpy as np
# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-7 exact tensor decomposition for 2x2 matrix multiplication.
    Uses Strassen's algorithm decomposition (optimal rank for 2x2).
    """
    n, m, p = 2, 2, 2
    rank = 7

    # Strassen's 7 products coefficients
    # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    
    # Product 1: (A00 + A11) * (B00 + B11)
    # Product 2: (A10 + A11) * B00
    # Product 3: A00 * (B01 - B11)
    # Product 4: A11 * (B10 - B00)
    # Product 5: (A00 + A01) * B11
    # Product 6: (A10 - A00) * (B00 + B01)
    # Product 7: (A01 - A11) * (B10 + B11)
    
    # U rows: A00, A01, A10, A11
    U_data = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]

    # V rows: B00, B01, B10, B11
    V_data = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]

    # W rows: C00 (k=0,i=0), C10 (k=0,i=1), C01 (k=1,i=0), C11 (k=1,i=1)
    # C00 = P1 + P4 - P5 + P7
    # C10 = P2 + P4
    # C01 = P3 + P5
    # C11 = P1 - P2 + P3 + P6
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
