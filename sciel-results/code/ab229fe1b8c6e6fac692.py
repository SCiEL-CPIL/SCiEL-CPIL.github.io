import numpy as np

def _schoolbook_decomposition():
    """Fallback rank-28 schoolbook decomposition."""
    n, m, p = 2, 2, 7
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, idx] = 1
                V[j*p + k, idx] = 1
                W[k*n + i, idx] = 1
                idx += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def construct():
    """
    Construct a rank-25 exact tensor decomposition for 2x2 * 2x7 matrix multiplication.
    Uses Strassen rank-7 for three 2x2 blocks and schoolbook rank-4 for the last column.
    """
    n, m, p = 2, 2, 7
    rank = 25

    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)

    def add_strassen_block(col_start, k_start):
        # Adds 7 terms for the 2x2 block starting at column k_start
        # M1 = (A00 + A11) * (B0,ks + B1,ks+1)
        U[0, col_start+0] += 1
        U[3, col_start+0] += 1
        V[7*0 + k_start, col_start+0] += 1
        V[7*1 + (k_start+1), col_start+0] += 1
        W[2*k_start + 0, col_start+0] += 1
        W[2*(k_start+1) + 1, col_start+0] += 1

        # M2 = (A10 + A11) * B0,ks
        U[2, col_start+1] += 1
        U[3, col_start+1] += 1
        V[7*0 + k_start, col_start+1] += 1
        W[2*k_start + 1, col_start+1] += 1
        W[2*(k_start+1) + 1, col_start+1] -= 1

        # M3 = A00 * (B0,ks+1 - B1,ks+1)
        U[0, col_start+2] += 1
        V[7*0 + (k_start+1), col_start+2] += 1
        V[7*1 + (k_start+1), col_start+2] -= 1
        W[2*(k_start+1) + 0, col_start+2] += 1
        W[2*(k_start+1) + 1, col_start+2] += 1

        # M4 = A11 * (B1,ks - B0,ks)
        U[3, col_start+3] += 1
        V[7*1 + k_start, col_start+3] += 1
        V[7*0 + k_start, col_start+3] -= 1
        W[2*k_start + 0, col_start+3] += 1
        W[2*k_start + 1, col_start+3] += 1

        # M5 = (A00 + A01) * B1,ks+1
        U[0, col_start+4] += 1
        U[1, col_start+4] += 1
        V[7*1 + (k_start+1), col_start+4] += 1
        W[2*k_start + 0, col_start+4] -= 1
        W[2*(k_start+1) + 0, col_start+4] += 1

        # M6 = (A10 - A00) * (B0,ks + B0,ks+1)
        U[2, col_start+5] += 1
        U[0, col_start+5] -= 1
        V[7*0 + k_start, col_start+5] += 1
        V[7*0 + (k_start+1), col_start+5] += 1
        W[2*(k_start+1) + 1, col_start+5] += 1

        # M7 = (A01 - A11) * (B1,ks + B1,ks+1)
        U[1, col_start+6] += 1
        U[3, col_start+6] -= 1
        V[7*1 + k_start, col_start+6] += 1
        V[7*1 + (k_start+1), col_start+6] += 1
        W[2*k_start + 0, col_start+6] += 1

    # Apply Strassen to columns 0-1
    add_strassen_block(0, 0)
    
    # Apply Strassen to columns 2-3
    add_strassen_block(7, 2)
    
    # Apply Strassen to columns 4-5
    add_strassen_block(14, 4)
    
    # Schoolbook for column 6
    col_start = 21
    k = 6

    # A00 * B0,6 -> C0,6
    U[0, col_start+0] += 1
    V[7*0 + k, col_start+0] += 1
    W[2*k + 0, col_start+0] += 1

    # A01 * B1,6 -> C0,6
    U[1, col_start+1] += 1
    V[7*1 + k, col_start+1] += 1
    W[2*k + 0, col_start+1] += 1

    # A10 * B0,6 -> C1,6
    U[2, col_start+2] += 1
    V[7*0 + k, col_start+2] += 1
    W[2*k + 1, col_start+2] += 1

    # A11 * B1,6 -> C1,6
    U[3, col_start+3] += 1
    V[7*1 + k, col_start+3] += 1
    W[2*k + 1, col_start+3] += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
