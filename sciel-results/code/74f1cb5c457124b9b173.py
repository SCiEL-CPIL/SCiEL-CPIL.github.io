import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 120

    n, m, p = DIMENSIONS

    def _target_tensor_223():
        # Target tensor T of shape (6, 4, 6)
        # T[c_idx, a_idx, b_idx] = 1 if c_idx corresponds to (i,k) and a_idx to (i,j) and b_idx to (j,k)
        # c_idx = i*3 + k
        # a_idx = i*2 + j
        # b_idx = j*3 + k
        T = np.zeros((6, 4, 6))
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    c_idx = i * 3 + k
                    a_idx = i * 2 + j
                    b_idx = j * 3 + k
                    T[c_idx, a_idx, b_idx] = 1.0
        return T

    def _search_223_rank9():
        T = _target_tensor_223()
        R = 9
        # Random initialization
        for _ in range(50):
            U = np.random.randn(4, R)
            V = np.random.randn(6, R)
            W = np.zeros((6, R))
            # Simple ALS
            for it in range(100):
                # Update W
                for r in range(R):
                    # W[:, r] = (T * (U[:, r] \otimes V[:, r])) / norm
                    # T is (6, 4, 6). U[:, r] is (4,). V[:, r] is (6,).
                    # Product P = T * np.outer(U[:, r], V[:, r])
                    # P[c, a, b] = T[c, a, b] * U[a, r] * V[b, r]
                    # We want W[c, r] such that sum_r W[c, r] U[a, r] V[b, r] = T[c, a, b]
                    # This is not a simple projection.
                    pass
            # Check error
            # Error = || T - sum_r W[:, r] \otimes U[:, r] \otimes V[:, r] ||
            # This is too slow for 50 restarts.
            pass
        return None

    # Fallback to Rank 50
    rank = 50
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    col = 0

    def emit_strassen(i0, j0, k0):
        nonlocal col
        # M1
        U[i0*m + j0, col] = 1.0
        U[(i0+1)*m + (j0+1), col] = 1.0
        V[j0*p + k0, col] = 1.0
        V[(j0+1)*p + (k0+1), col] = 1.0
        W[k0*n + i0, col] = 1.0
        W[(k0+1)*n + (i0+1), col] = 1.0
        col += 1
        # M2
        U[(i0+1)*m + j0, col] = 1.0
        U[(i0+1)*m + (j0+1), col] = 1.0
        V[j0*p + k0, col] = 1.0
        W[k0*n + (i0+1), col] = 1.0
        W[(k0+1)*n + (i0+1), col] = -1.0
        col += 1
        # M3
        U[i0*m + j0, col] = 1.0
        V[j0*p + (k0+1), col] = 1.0
        V[(j0+1)*p + (k0+1), col] = -1.0
        W[(k0+1)*n + i0, col] = 1.0
        W[(k0+1)*n + (i0+1), col] = 1.0
        col += 1
        # M4
        U[(i0+1)*m + (j0+1), col] = 1.0
        V[(j0+1)*p + k0, col] = 1.0
        V[j0*p + k0, col] = -1.0
        W[k0*n + (i0+1), col] = 1.0
        W[k0*n + i0, col] = 1.0
        col += 1
        # M5
        U[i0*m + j0, col] = 1.0
        U[i0*m + (j0+1), col] = 1.0
        V[(j0+1)*p + (k0+1), col] = 1.0
        W[k0*n + i0, col] = -1.0
        W[(k0+1)*n + i0, col] = 1.0
        col += 1
        # M6
        U[(i0+1)*m + j0, col] = 1.0
        U[i0*m + j0, col] = -1.0
        V[j0*p + k0, col] = 1.0
        V[j0*p + (k0+1), col] = 1.0
        W[(k0+1)*n + (i0+1), col] = 1.0
        col += 1
        # M7
        U[i0*m + (j0+1), col] = 1.0
        U[(i0+1)*m + (j0+1), col] = -1.0
        V[(j0+1)*p + k0, col] = 1.0
        V[(j0+1)*p + (k0+1), col] = 1.0
        W[k0*n + i0, col] = 1.0
        col += 1

    def emit_schoolbook_2x2x1(j0, k0):
        nonlocal col
        i0 = 0
        U[i0*m + j0, col] = 1.0
        V[j0*p + k0, col] = 1.0
        W[k0*n + i0, col] = 1.0
        col += 1
        U[i0*m + (j0+1), col] = 1.0
        V[(j0+1)*p + k0, col] = 1.0
        W[k0*n + i0, col] = 1.0
        col += 1
        U[(i0+1)*m + j0, col] = 1.0
        V[j0*p + k0, col] = 1.0
        W[k0*n + (i0+1), col] = 1.0
        col += 1
        U[(i0+1)*m + (j0+1), col] = 1.0
        V[(j0+1)*p + k0, col] = 1.0
        W[k0*n + (i0+1), col] = 1.0
        col += 1

    emit_strassen(0, 0, 0)
    emit_strassen(0, 0, 2)
    emit_strassen(0, 0, 4)
    emit_schoolbook_2x2x1(0, 6)
    
    emit_strassen(0, 2, 0)
    emit_strassen(0, 2, 2)
    emit_strassen(0, 2, 4)
    emit_schoolbook_2x2x1(2, 6)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
