"""Programmatic construction of the 49-term rank decomposition for 4x4x4
matrix multiplication via the Kronecker product of the 2x2x2 Strassen tensor."""

import time

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
# The 2x2x2 Strassen tensor.
# For A (2x2) * B (2x2) = C (2x2):
#   A = [[a, b], [c, d]]
#   B = [[e, f], [g, h]]
# Strassen's 7 products:
#   M1 = (a+d)(e+h)
#   M2 = (c+d)e
#   M3 = a(f-h)
#   M4 = d(g-f)
#   M5 = (a+b)h
#   M6 = (c-a)(e+g)
#   M7 = (b-d)(f+h)
# Output:
#   C00 = M1 + M4 - M5 + M7
#   C01 = M3 + M5
#   C10 = M2 + M4
#   C11 = M1 - M2 + M3 + M6
#
# Factor matrices for the 2x2x2 Strassen tensor (rank 7):
# U: rows = (i, j) flattened as i*2 + j, columns = term c. U[i,j,c] = coefficient of A[i,j]
# V: rows = (j, k) flattened as j*2 + k, columns = term c. V[j,k,c] = coefficient of B[j,k]
# W: rows = (k, i) flattened as k*2 + i (trace convention W[k*n+i]), columns = term c. W[k,i,c] = coefficient of C[i,k]

def _strassen_factors():
    """Return (U, V, W) for the 2x2x2 Strassen tensor as lists of lists of ints."""
    # U[i*2+j, c] = coeff of A[i,j] in term c
    U = [
        [1, 0, 1, 0, 1, 1, 0],  # a (i=0,j=0)
        [0, 0, 1, 0, 1, 0, 1],  # b (i=0,j=1)
        [0, 1, 0, 0, 0, -1, 0], # c (i=1,j=0)
        [1, 1, 0, 1, 0, 0, -1], # d (i=1,j=1)
    ]
    # V[j*2+k, c] = coeff of B[j,k] in term c
    V = [
        [1, 1, 0, 0, 0, 1, 0],  # e (j=0,k=0)
        [1, 0, 1, -1, 0, 0, 1], # f (j=0,k=1)
        [0, 0, 0, 1, 0, 1, 0],  # g (j=1,k=0)
        [1, 0, -1, 0, 1, 0, 1], # h (j=1,k=1)
    ]
    # W[k*2+i, c] = coeff of C[i,k] in term c  (trace convention)
    # C00 = M1 + M4 - M5 + M7  -> row k=0,i=0: [1, 0, 0, 1, -1, 0, 1]
    # C01 = M3 + M5            -> row k=1,i=0: [0, 0, 1, 0, 1, 0, 0]
    # C10 = M2 + M4            -> row k=0,i=1: [0, 1, 0, 1, 0, 0, 0]
    # C11 = M1 - M2 + M3 + M6  -> row k=1,i=1: [1, -1, 1, 0, 0, 1, 0]
    W = [
        [1, 0, 0, 1, -1, 0, 1],  # k=0,i=0 -> C00
        [0, 1, 0, 1, 0, 0, 0],   # k=0,i=1 -> C10
        [0, 0, 1, 0, 1, 0, 0],   # k=1,i=0 -> C01
        [1, -1, 1, 0, 0, 1, 0],  # k=1,i=1 -> C11
    ]
    return U, V, W


def construct():
    deadline = time.time() + 140.0  # safety margin before 150s kill
    t0 = time.time()

    U1, V1, W1 = _strassen_factors()

    # Build the 4x4x4 decomposition via Kronecker product of the Strassen tensor with itself.
    # The 4x4 matrix multiply tensor = S ⊗ S where S is the 2x2x2 Strassen tensor.
    # Rank = 7 * 7 = 49.
    #
    # Index mapping:
    #   Global A index: i = 2*i1 + i0, j = 2*j1 + j0
    #   Global B index: j = 2*j1 + j0, k = 2*k1 + k0
    #   Global C index: i = 2*i1 + i0, k = 2*k1 + k0
    #
    # U row (global) = i*4 + j = (2*i1+i0)*4 + (2*j1+j0) = 8*i1 + 4*i0 + 2*j1 + j0
    # V row (global) = j*4 + k = (2*j1+j0)*4 + (2*k1+k0) = 8*j1 + 4*j0 + 2*k1 + k0
    # W row (global) = k*4 + i = (2*k1+k0)*4 + (2*i1+i0) = 8*k1 + 4*k0 + 2*i1 + i0
    #
    # For Kronecker term (c1, c0) with global term index c = c1*7 + c0:
    #   U[global_row, c] = U1[i1*2+j1, c1] * U1[i0*2+j0, c0]
    #   V[global_row, c] = V1[j1*2+k1, c1] * V1[j0*2+k0, c0]
    #   W[global_row, c] = W1[k1*2+i1, c1] * W1[k0*2+i0, c0]

    rank = 49
    n = m = p = 4
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for c1 in range(7):
        for c0 in range(7):
            c = c1 * 7 + c0
            # For each (i1, j1) where U1 has nonzero at c1
            for i1 in range(2):
                for j1 in range(2):
                    u1 = U1[i1 * 2 + j1][c1]
                    if u1 == 0:
                        continue
                    for i0 in range(2):
                        for j0 in range(2):
                            u0 = U1[i0 * 2 + j0][c0]
                            if u0 == 0:
                                continue
                            val = u1 * u0
                            if val == 0:
                                continue
                            # Global U row
                            i = 2 * i1 + i0
                            j = 2 * j1 + j0
                            u_row = i * 4 + j
                            U[u_row][c] += val

            # For each (j1, k1) where V1 has nonzero at c1
            for j1 in range(2):
                for k1 in range(2):
                    v1 = V1[j1 * 2 + k1][c1]
                    if v1 == 0:
                        continue
                    for j0 in range(2):
                        for k0 in range(2):
                            v0 = V1[j0 * 2 + k0][c0]
                            if v0 == 0:
                                continue
                            val = v1 * v0
                            if val == 0:
                                continue
                            # Global V row
                            j = 2 * j1 + j0
                            k = 2 * k1 + k0
                            v_row = j * 4 + k
                            V[v_row][c] += val

            # For each (k1, i1) where W1 has nonzero at c1
            for k1 in range(2):
                for i1 in range(2):
                    w1 = W1[k1 * 2 + i1][c1]
                    if w1 == 0:
                        continue
                    for k0 in range(2):
                        for i0 in range(2):
                            w0 = W1[k0 * 2 + i0][c0]
                            if w0 == 0:
                                continue
                            val = w1 * w0
                            if val == 0:
                                continue
                            # Global W row
                            k = 2 * k1 + k0
                            i = 2 * i1 + i0
                            w_row = k * 4 + i
                            W[w_row][c] += val

    elapsed = time.time() - t0
    # Verify correctness with a simple test if time permits
    if time.time() < deadline - 5.0:
        _verify(U, V, W, n, m, p)

    return {"U": U, "V": V, "W": W}


def _verify(U, V, W, n, m, p):
    """Verify that sum_c U[:,c] outer V[:,c] outer W[:,c] equals the matrix multiply tensor."""
    import random
    # Test with a few random matrices
    for trial in range(5):
        A = [[random.randint(-5, 5) for _ in range(m)] for _ in range(n)]
        B = [[random.randint(-5, 5) for _ in range(p)] for _ in range(m)]
        # True product C = A * B
        C = [[0] * p for _ in range(n)]
        for i in range(n):
            for k in range(p):
                s = 0
                for j in range(m):
                    s += A[i][j] * B[j][k]
                C[i][k] = s
        # Reconstruct C from decomposition
        C_rec = [[0] * p for _ in range(n)]
        rank = len(U[0])
        for c in range(rank):
            # M = U_col * V_col (element-wise)
            # Then C[i,k] += W[k*n+i, c] * M
            for i in range(n):
                for k in range(p):
                    w_val = W[k * n + i][c]
                    if w_val == 0:
                        continue
                    # M = sum_j U[i*m+j, c] * V[j*p+k, c]
                    s = 0
                    for j in range(m):
                        s += U[i * m + j][c] * V[j * p + k][c]
                    C_rec[i][k] += w_val * s
        # Check
        for i in range(n):
            for k in range(p):
                if C[i][k] != C_rec[i][k]:
                    raise ValueError(f"Verification failed at ({i},{k}): {C[i][k]} != {C_rec[i][k]}")


# EVOLVE-BLOCK-END
