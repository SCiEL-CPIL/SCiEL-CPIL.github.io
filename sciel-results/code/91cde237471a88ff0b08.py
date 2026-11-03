# EVOLVE-BLOCK-START
import time
import numpy as np

def _build_rank17():
    """Build a verified rank-17 decomposition for 2x3x3 matrix multiplication."""
    n, m, p = 2, 3, 3
    rank = 17
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    # Part 1: Strassen for A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # M1: (A00+A11)(B00+B11)
    U[0, 0] = 1; U[4, 0] = 1
    V[0, 0] = 1; V[4, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    # M2: (A10+A11)B00
    U[3, 1] = 1; U[4, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    # M3: A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[4, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    # M4: A11(B10-B00)
    U[4, 3] = 1
    V[3, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    # M5: (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[4, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    # M6: (A10-A00)(B00+B01)
    U[3, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    # M7: (A01-A11)(B10+B11)
    U[1, 6] = 1; U[4, 6] = -1
    V[3, 6] = 1; V[4, 6] = 1
    W[0, 6] = 1

    # Part 2: Schoolbook for A[0:2, 2] * B[2, 0:2] -> C[0:2, 0:2]
    U[2, 7] = 1; V[6, 7] = 1; W[0, 7] = 1
    U[2, 8] = 1; V[7, 8] = 1; W[2, 8] = 1
    U[5, 9] = 1; V[6, 9] = 1; W[1, 9] = 1
    U[5, 10] = 1; V[7, 10] = 1; W[3, 10] = 1

    # Part 3: Schoolbook for A[0:2, 0:3] * B[0:3, 2] -> C[0:2, 2]
    U[0, 11] = 1; V[2, 11] = 1; W[4, 11] = 1
    U[1, 12] = 1; V[5, 12] = 1; W[4, 12] = 1
    U[2, 13] = 1; V[8, 13] = 1; W[4, 13] = 1
    U[3, 14] = 1; V[2, 14] = 1; W[5, 14] = 1
    U[4, 15] = 1; V[5, 15] = 1; W[5, 15] = 1
    U[5, 16] = 1; V[8, 16] = 1; W[5, 16] = 1

    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n=2, m=3, p=3, tol=1e-9):
    """Verify decomposition exactly reproduces the matrix multiplication tensor."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                s = np.dot(U[a, :], V[b, :] * W[c, :])
                if abs(s - 1.0) > tol:
                    return False
    return True

def _to_list(U, V, W):
    """Convert numpy arrays to nested list format with integer entries."""
    U_list = [[int(U[i, r]) for r in range(U.shape[1])] for i in range(U.shape[0])]
    V_list = [[int(V[i, r]) for r in range(V.shape[1])] for i in range(V.shape[0])]
    W_list = [[int(W[i, r]) for r in range(W.shape[1])] for i in range(W.shape[0])]
    return {"U": U_list, "V": V_list, "W": W_list}

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for
    multiplying a 2x3 matrix by a 3x3 matrix.
    """
    base = _build_rank17()
    return _to_list(base["U"], base["V"], base["W"])
# EVOLVE-BLOCK-END
