import numpy as np
import time

DIMENSIONS = [3, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 150s limit

    n, m, p = 3, 4, 8
    R = 88  # rank of hybrid decomposition
    U = np.zeros((n*m, R), dtype=np.float64)
    V = np.zeros((m*p, R), dtype=np.float64)
    W = np.zeros((p*n, R), dtype=np.float64)

    col = 0

    # Row 0: schoolbook decomposition (4 * 8 = 32 multiplications)
    for j in range(m):
        for k in range(p):
            U[0*m + j, col] = 1.0
            V[j*p + k, col] = 1.0
            W[k*n + 0, col] = 1.0
            col += 1

    # Rows 1-2: Strassen-like block decomposition
    # Split into 2x2 blocks for each pair of rows/cols/rows
    def strassen_block(i0, jA0, jB0, k0, base_col):
        c = base_col
        # M1
        U[i0*4 + jA0, c] += 1
        U[(i0+1)*4 + (jA0+1), c] += 1
        V[jB0*8 + k0, c] += 1
        V[(jB0+1)*8 + (k0+1), c] += 1
        W[k0*3 + i0, c] += 1
        W[(k0+1)*3 + (i0+1), c] += 1
        c += 1
        # M2
        U[(i0+1)*4 + jA0, c] += 1
        U[(i0+1)*4 + (jA0+1), c] += 1
        V[jB0*8 + k0, c] += 1
        W[k0*3 + (i0+1), c] += 1
        W[(k0+1)*3 + (i0+1), c] -= 1
        c += 1
        # M3
        U[i0*4 + jA0, c] += 1
        V[jB0*8 + (k0+1), c] += 1
        V[(jB0+1)*8 + (k0+1), c] -= 1
        W[(k0+1)*3 + i0, c] += 1
        W[(k0+1)*3 + (i0+1), c] += 1
        c += 1
        # M4
        U[(i0+1)*4 + (jA0+1), c] += 1
        V[(jB0+1)*8 + k0, c] += 1
        V[jB0*8 + k0, c] -= 1
        W[k0*3 + i0, c] += 1
        W[k0*3 + (i0+1), c] += 1
        c += 1
        # M5
        U[i0*4 + jA0, c] += 1
        U[i0*4 + (jA0+1), c] += 1
        V[(jB0+1)*8 + (k0+1), c] += 1
        W[k0*3 + i0, c] -= 1
        W[(k0+1)*3 + i0, c] += 1
        c += 1
        # M6
        U[(i0+1)*4 + jA0, c] += 1
        U[i0*4 + jA0, c] -= 1
        V[jB0*8 + k0, c] += 1
        V[jB0*8 + (k0+1), c] += 1
        W[(k0+1)*3 + (i0+1), c] += 1
        c += 1
        # M7
        U[i0*4 + (jA0+1), c] += 1
        U[(i0+1)*4 + (jA0+1), c] -= 1
        V[(jB0+1)*8 + k0, c] += 1
        V[(jB0+1)*8 + (k0+1), c] += 1
        W[k0*3 + i0, c] += 1
        c += 1
        return c

    # Apply Strassen blocks for the 2x4x8 subproblem
    for k0 in [0, 2, 4, 6]:
        col = strassen_block(1, 0, 0, k0, col)
        col = strassen_block(1, 2, 2, k0, col)

    assert col == 88, f"Expected 88 columns, got {col}"

    # Convert to lists
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
