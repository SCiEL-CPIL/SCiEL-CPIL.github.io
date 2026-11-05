import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110

    # Target tensor T[i*m+j, j*p+k, k*n+i] = 1
    # n=2, m=4, p=7
    # U shape (8, R)
    # V shape (28, R)
    # W shape (14, R)
    
    rank = 47
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0

    def add_strassen(i0, j0, k0):
        nonlocal col
        # M1: (a+d)(e+h) -> C00 + C11
        # A: i0,j0 (1); i0+1,j0+1 (1)
        # B: j0,k0 (1); j0+1,k0+1 (1)
        # C: k0,i0 (1); k0+1,i0+1 (1)
        U[i0 * m + j0, col] += 1
        U[(i0 + 1) * m + (j0 + 1), col] += 1
        V[j0 * p + k0, col] += 1
        V[(j0 + 1) * p + (k0 + 1), col] += 1
        W[k0 * n + i0, col] += 1
        W[(k0 + 1) * n + (i0 + 1), col] += 1
        col += 1

        # M2: (c+d)e -> C10 - C11
        # A: i0+1,j0 (1); i0+1,j0+1 (1)
        # B: j0,k0 (1)
        # C: k0,i0+1 (1); k0+1,i0+1 (-1)
        U[(i0 + 1) * m + j0, col] += 1
        U[(i0 + 1) * m + (j0 + 1), col] += 1
        V[j0 * p + k0, col] += 1
        W[k0 * n + (i0 + 1), col] += 1
        W[(k0 + 1) * n + (i0 + 1), col] -= 1
        col += 1

        # M3: a(f-h) -> C01 + C11
        # A: i0,j0 (1)
        # B: j0,k0+1 (1); j0+1,k0+1 (-1)
        # C: k0+1,i0 (1); k0+1,i0+1 (1)
        U[i0 * m + j0, col] += 1
        V[j0 * p + (k0 + 1), col] += 1
        V[(j0 + 1) * p + (k0 + 1), col] -= 1
        W[(k0 + 1) * n + i0, col] += 1
        W[(k0 + 1) * n + (i0 + 1), col] += 1
        col += 1

        # M4: d(g-e) -> C00 + C10
        # A: i0+1,j0+1 (1)
        # B: j0+1,k0 (1); j0,k0 (-1)
        # C: k0,i0 (1); k0,i0+1 (1)
        U[(i0 + 1) * m + (j0 + 1), col] += 1
        V[(j0 + 1) * p + k0, col] += 1
        V[j0 * p + k0, col] -= 1
        W[k0 * n + i0, col] += 1
        W[k0 * n + (i0 + 1), col] += 1
        col += 1

        # M5: (a+b)h -> -C00 + C01
        # A: i0,j0 (1); i0,j0+1 (1)
        # B: j0+1,k0+1 (1)
        # C: k0,i0 (-1); k0+1,i0 (1)
        U[i0 * m + j0, col] += 1
        U[i0 * m + (j0 + 1), col] += 1
        V[(j0 + 1) * p + (k0 + 1), col] += 1
        W[k0 * n + i0, col] -= 1
        W[(k0 + 1) * n + i0, col] += 1
        col += 1

        # M6: (c-a)(e+f) -> C11
        # A: i0+1,j0 (1); i0,j0 (-1)
        # B: j0,k0 (1); j0,k0+1 (1)
        # C: k0+1,i0+1 (1)
        U[(i0 + 1) * m + j0, col] += 1
        U[i0 * m + j0, col] -= 1
        V[j0 * p + k0, col] += 1
        V[j0 * p + (k0 + 1), col] += 1
        W[(k0 + 1) * n + (i0 + 1), col] += 1
        col += 1

        # M7: (b-d)(g+h) -> C00 + C10
        # A: i0,j0+1 (1); i0+1,j0+1 (-1)
        # B: j0+1,k0 (1); j0+1,k0+1 (1)
        # C: k0,i0 (1); k0,i0+1 (1)
        U[i0 * m + (j0 + 1), col] += 1
        U[(i0 + 1) * m + (j0 + 1), col] -= 1
        V[(j0 + 1) * p + k0, col] += 1
        V[(j0 + 1) * p + (k0 + 1), col] += 1
        W[k0 * n + i0, col] += 1
        W[k0 * n + (i0 + 1), col] += 1
        col += 1

    # Strassen blocks for j in {0,1}, {2,3} and k in {0,1}, {2,3}, {4,5}
    for j0 in [0, 2]:
        for k0 in [0, 2, 4]:
            add_strassen(0, j0, k0)

    # Handle k=6 (last column of B) with rank 5 terms
    # 1. A00 B06 -> C06
    U[0 * m + 0, col] = 1
    V[0 * p + 6, col] = 1
    W[6 * n + 0, col] = 1
    col += 1

    # 2. A10 B06 -> C16
    U[1 * m + 0, col] = 1
    V[0 * p + 6, col] = 1
    W[6 * n + 1, col] = 1
    col += 1

    # 3. (A01+A11) B16 -> C06 + C16
    U[0 * m + 1, col] = 1
    U[1 * m + 1, col] = 1
    V[1 * p + 6, col] = 1
    W[6 * n + 0, col] = 1
    W[6 * n + 1, col] = 1
    col += 1

    # 4. (A02+A12) B26 -> C06 + C16
    U[0 * m + 2, col] = 1
    U[1 * m + 2, col] = 1
    V[2 * p + 6, col] = 1
    W[6 * n + 0, col] = 1
    W[6 * n + 1, col] = 1
    col += 1

    # 5. (A03+A13) B36 -> C06 + C16
    U[0 * m + 3, col] = 1
    U[1 * m + 3, col] = 1
    V[3 * p + 6, col] = 1
    W[6 * n + 0, col] = 1
    W[6 * n + 1, col] = 1
    col += 1

    assert col == rank, f"Expected rank {rank}, got {col}"
    
    # Verification
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    recon = np.einsum('ar,br,cr->abc', U, V, W)
    if not np.allclose(recon, T, atol=1e-9):
        # Fallback to schoolbook if verification fails
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
