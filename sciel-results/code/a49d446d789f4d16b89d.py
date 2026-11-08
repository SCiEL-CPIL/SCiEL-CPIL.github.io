import time
import numpy as np

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    U_dim = n * m   # 10
    V_dim = m * p   # 25
    W_dim = p * n   # 10

    # Target tensor T[u,v,w] = 1 when u=i*m+j, v=j*p+k, w=k*n+i
    T = np.zeros((U_dim, V_dim, W_dim))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    def validate(U, V, W, atol=1e-8):
        prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        return np.allclose(prod, T, atol=atol)

    def snap_validate(U, V, W):
        # The construction uses only integers; snap to nearest integer to be safe.
        U_i = np.round(U).astype(float)
        V_i = np.round(V).astype(float)
        W_i = np.round(W).astype(float)
        if validate(U_i, V_i, W_i):
            return True, U_i, V_i, W_i
        return False, None, None, None

    def build_rank28():
        rank = 28
        U = np.zeros((U_dim, rank))
        V = np.zeros((V_dim, rank))
        W = np.zeros((W_dim, rank))

        def add_strassen(j0, k0, c0):
            # Block covers A rows j0..j0+1, B cols k0..k0+1, C cols k0..k0+1
            # P1 = (A00+A11)(B00+B11)
            U[0 * m + j0, c0] += 1
            U[1 * m + (j0 + 1), c0] += 1
            V[j0 * p + k0, c0] += 1
            V[(j0 + 1) * p + (k0 + 1), c0] += 1
            W[k0 * n + 0, c0] += 1
            W[(k0 + 1) * n + 1, c0] += 1
            # P2 = (A10+A11) B00
            U[1 * m + j0, c0 + 1] += 1
            U[1 * m + (j0 + 1), c0 + 1] += 1
            V[j0 * p + k0, c0 + 1] += 1
            W[k0 * n + 1, c0 + 1] += 1
            W[(k0 + 1) * n + 1, c0 + 1] -= 1
            # P3 = A00 (B01 - B11)
            U[0 * m + j0, c0 + 2] += 1
            V[j0 * p + (k0 + 1), c0 + 2] += 1
            V[(j0 + 1) * p + (k0 + 1), c0 + 2] -= 1
            W[(k0 + 1) * n + 0, c0 + 2] += 1
            W[(k0 + 1) * n + 1, c0 + 2] += 1
            # P4 = A11 (B10 - B00)
            U[1 * m + (j0 + 1), c0 + 3] += 1
            V[(j0 + 1) * p + k0, c0 + 3] += 1
            V[j0 * p + k0, c0 + 3] -= 1
            W[k0 * n + 0, c0 + 3] += 1
            W[k0 * n + 1, c0 + 3] += 1
            # P5 = (A00+A01) B11
            U[0 * m + j0, c0 + 4] += 1
            U[0 * m + (j0 + 1), c0 + 4] += 1
            V[(j0 + 1) * p + (k0 + 1), c0 + 4] += 1
            W[k0 * n + 0, c0 + 4] -= 1
            W[(k0 + 1) * n + 0, c0 + 4] += 1
            # P6 = (A10 - A00)(B00+B01)
            U[1 * m + j0, c0 + 5] += 1
            U[0 * m + j0, c0 + 5] -= 1
            V[j0 * p + k0, c0 + 5] += 1
            V[j0 * p + (k0 + 1), c0 + 5] += 1
            W[(k0 + 1) * n + 1, c0 + 5] += 1
            # P7 = (A01 - A11)(B10+B11)
            U[0 * m + (j0 + 1), c0 + 6] += 1
            U[1 * m + (j0 + 1), c0 + 6] -= 1
            V[(j0 + 1) * p + k0, c0 + 6] += 1
            V[(j0 + 1) * p + (k0 + 1), c0 + 6] += 1
            W[k0 * n + 0, c0 + 6] += 1

        # Four 2x2 blocks: (j0,k0) in {(0,0),(0,2),(2,0),(2,2)}
        add_strassen(0, 0, 0)
        add_strassen(0, 2, 7)
        add_strassen(2, 0, 14)
        add_strassen(2, 2, 21)

        # Cross terms for the 4th (index 4) row/col of the 5x5 partition:
        #   i * A[4] * B[k]  -> C[k,i]  for i in {0,1}, k in {0,1,2,3}
        c = 28
        for i in range(2):
            for k in range(4):
                U[i * m + 4, c] = 1
                V[4 * p + k, c] = 1
                W[k * n + i, c] = 1
                c += 1
        #   i * A[j] * B[4]  -> C[4,i]  for i in {0,1}, j in {0,1,2,3}
        for i in range(2):
            for j in range(4):
                U[i * m + j, c] = 1
                V[j * p + 4, c] = 1
                W[4 * n + i, c] = 1
                c += 1
        #   i * A[4] * B[4]  -> C[4,i]  for i in {0,1}
        for i in range(2):
            U[i * m + 4, c] = 1
            V[4 * p + 4, c] = 1
            W[4 * n + i, c] = 1
            c += 1

        return U, V, W

    # Build and verify the rank-28 construction.
    U28, V28, W28 = build_rank28()
    if validate(U28, V28, W28):
        return {
            "U": [[float(x) for x in row] for row in U28],
            "V": [[float(x) for x in row] for row in V28],
            "W": [[float(x) for x in row] for row in W28],
        }

    # Fallback: naive rank-50 (always valid).
    rank = n * m * p
    U = np.zeros((U_dim, rank))
    V = np.zeros((V_dim, rank))
    W = np.zeros((W_dim, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {
        "U": [[float(x) for x in row] for row in U],
        "V": [[float(x) for x in row] for row in V],
        "W": [[float(x) for x in row] for row in W],
    }
# EVOLVE-BLOCK-END
