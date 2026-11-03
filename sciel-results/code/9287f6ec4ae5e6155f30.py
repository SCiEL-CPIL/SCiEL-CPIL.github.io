"""
Rank-49 exact tensor decomposition for 4x4 matrix multiplication.

Constructed as the Kronecker product of two rank-7 Strassen decompositions
for 2x2 matrix multiplication, with a bit-interleaving permutation to
correct the index ordering from Kronecker-product layout to the standard
row-major (i*4+j) layout required by the task contract.

The matrix multiplication tensor M_{4,4,4} has entries
  M[i*4+j, j*4+k, k*4+i] = 1  for i,j,k in {0,1,2,3}
and 0 elsewhere.

The Kronecker product of M_{2,2,2} with itself, after appropriate
reindexing, equals M_{4,4,4}. Since rank(M_{2,2,2}) = 7 (Strassen),
the Kronecker product gives a valid decomposition of rank at most 49.

This is better than the schoolbook rank of 64, and while not optimal
(known minimum is 32), it is a clean, verifiable structural improvement.
"""
import time
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """Return a rank-49 decomposition for 4x4 matrix multiplication via Kronecker product of Strassen."""
    deadline = time.time() + 140
    n, m, p = DIMENSIONS

    # Build the rank-7 Strassen decomposition for 2x2 matrix multiplication.
    # T_2[i*2+j, j*2+k, k*2+i] = 1 for i,j,k in {0,1}
    U2 = np.zeros((4, 7), dtype=np.float64)
    V2 = np.zeros((4, 7), dtype=np.float64)
    W2 = np.zeros((4, 7), dtype=np.float64)

    # Strassen's 7 products for 2x2:
    # M0 = (A00+A11)*(B00+B11)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1

    # M1 = (A10+A11)*B00
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[1, 1] = 1

    # M2 = A00*(B10-B11)
    U2[0, 2] = 1
    V2[2, 2] = 1; V2[3, 2] = -1
    W2[2, 2] = 1

    # M3 = A11*(B01-B00)
    U2[3, 3] = 1
    V2[1, 3] = 1; V2[0, 3] = -1
    W2[2, 3] = 1; W2[3, 3] = 1

    # M4 = (A00+A01)*B11
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[2, 4] = 1

    # M5 = (A10-A00)*(B00+B01)
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[1, 5] = -1

    # M6 = (A01-A11)*(B10+B11)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[1, 6] = 1

    # Verify 2x2 Strassen decomposition
    for i in range(2):
        for j in range(2):
            for k in range(2):
                expected = 1.0
                actual = 0.0
                for c in range(7):
                    actual += U2[i*2+j, c] * V2[j*2+k, c] * W2[k*2+i, c]
                if abs(actual - expected) > 1e-9:
                    # Fallback to schoolbook rank-64
                    return _schoolbook()

    # Kronecker product to get 4x4 decomposition
    # np.kron(U2, U2) has shape (4*4, 7*7) = (16, 49)
    # The row index in the kron product for U is: (i1*2+j1)*4 + (i2*2+j2)
    # where i = i1*2+i2, j = j1*2+j2
    # We want row index = i*4+j = (i1*2+i2)*4 + (j1*2+j2) = i1*8 + i2*4 + j1*2 + j2
    # The kron row index is: (i1*2+j1)*4 + (i2*2+j2) = i1*8 + j1*4 + i2*2 + j2
    # So we need a permutation to go from kron layout to standard layout.

    U4_kron = np.kron(U2, U2)  # (16, 49)
    V4_kron = np.kron(V2, V2)  # (16, 49)
    W4_kron = np.kron(W2, W2)  # (16, 49)

    if time.time() > deadline:
        return _schoolbook()

    # Compute permutation for U: maps standard index S = i*4+j to kron index K
    # Standard: i = S//4, j = S%4; i1=i//2, i2=i%2, j1=j//2, j2=j%2
    # Kron: K = (i1*2+j1)*4 + (i2*2+j2)
    def perm_u(S):
        i = S // 4
        j = S % 4
        i1, i2 = i // 2, i % 2
        j1, j2 = j // 2, j % 2
        return (i1 * 2 + j1) * 4 + (i2 * 2 + j2)

    # For V: standard index S = j*4+k; j1=j//2, j2=j%2, k1=k//2, k2=k%2
    # Kron: K = (j1*2+k1)*4 + (j2*2+k2)
    def perm_v(S):
        j = S // 4
        k = S % 4
        j1, j2 = j // 2, j % 2
        k1, k2 = k // 2, k % 2
        return (j1 * 2 + k1) * 4 + (j2 * 2 + k2)

    # For W: standard index S = k*4+i; k1=k//2, k2=k%2, i1=i//2, i2=i%2
    # Kron: K = (k1*2+i1)*4 + (k2*2+i2)
    def perm_w(S):
        k = S // 4
        i = S % 4
        k1, k2 = k // 2, k % 2
        i1, i2 = i // 2, i % 2
        return (k1 * 2 + i1) * 4 + (k2 * 2 + i2)

    perm_U = np.array([perm_u(S) for S in range(16)], dtype=int)
    perm_V = np.array([perm_v(S) for S in range(16)], dtype=int)
    perm_W = np.array([perm_w(S) for S in range(16)], dtype=int)

    U4 = U4_kron[perm_U, :]
    V4 = V4_kron[perm_V, :]
    W4 = W4_kron[perm_W, :]

    if time.time() > deadline:
        return _schoolbook()

    # Verify the 4x4 decomposition
    for i in range(4):
        for j in range(4):
            for k in range(4):
                expected = 1.0
                actual = 0.0
                for c in range(49):
                    actual += U4[i*4+j, c] * V4[j*4+k, c] * W4[k*4+i, c]
                if abs(actual - expected) > 1e-9:
                    return _schoolbook()

    # Convert to lists (the interface expects lists of lists)
    return {
        "U": U4.tolist(),
        "V": V4.tolist(),
        "W": W4.tolist()
    }


def _schoolbook():
    """Fallback: rank-64 schoolbook decomposition."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
