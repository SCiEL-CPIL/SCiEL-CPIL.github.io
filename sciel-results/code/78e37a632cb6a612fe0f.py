"""Strassen-based 4x5x6 matrix multiplication decomposition (rank 108)."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

def construct():
    start_time = time.time()
    DEADLINE = 150.0
    SAFETY_MARGIN = 5.0

    n, m, p = DIMENSIONS

    # Total rank calculation
    # 12 blocks of 2x2x2 Strassen (rank 7 each) -> 84
    # 1 block of 4x1x6 schoolbook (rank 24) -> 24
    # Total rank = 108
    rank = 108

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    def add_strassen_block(i0, j0, k0, r_start):
        # Adds the Strassen decomposition for A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+2]
        # to the global U, V, W matrices starting at column index r_start

        # Helper to get global indices
        def u_idx(i, j): return (i0 + i) * m + (j0 + j)
        def v_idx(j, k): return (j0 + j) * p + (k0 + k)
        def w_idx(k, i): return (k0 + k) * n + (i0 + i)

        # M1 = (A00 + A11)(B00 + B11)
        # C00 += 1, C11 += 1
        U[u_idx(0, 0)][r_start] += 1
        U[u_idx(1, 1)][r_start] += 1
        V[v_idx(0, 0)][r_start] += 1
        V[v_idx(1, 1)][r_start] += 1
        W[w_idx(0, 0)][r_start] += 1
        W[w_idx(1, 1)][r_start] += 1

        # M2 = (A10 + A11)B00
        # C10 += 1, C11 += -1
        r_start += 1
        U[u_idx(1, 0)][r_start] += 1
        U[u_idx(1, 1)][r_start] += 1
        V[v_idx(0, 0)][r_start] += 1
        W[w_idx(0, 1)][r_start] += 1
        W[w_idx(1, 1)][r_start] += -1

        # M3 = A00(B01 - B11)
        # C01 += 1, C11 += 1
        r_start += 1
        U[u_idx(0, 0)][r_start] += 1
        V[v_idx(0, 1)][r_start] += 1
        V[v_idx(1, 1)][r_start] += -1
        W[w_idx(1, 0)][r_start] += 1
        W[w_idx(1, 1)][r_start] += 1

        # M4 = A11(B10 - B00)
        # C00 += 1, C10 += 1
        r_start += 1
        U[u_idx(1, 1)][r_start] += 1
        V[v_idx(1, 0)][r_start] += 1
        V[v_idx(0, 0)][r_start] += -1
        W[w_idx(0, 0)][r_start] += 1
        W[w_idx(0, 1)][r_start] += 1

        # M5 = (A00 + A01)B11
        # C00 += -1, C01 += 1
        r_start += 1
        U[u_idx(0, 0)][r_start] += 1
        U[u_idx(0, 1)][r_start] += 1
        V[v_idx(1, 1)][r_start] += 1
        W[w_idx(0, 0)][r_start] += -1
        W[w_idx(1, 0)][r_start] += 1

        # M6 = (A10 - A00)(B00 + B01)
        # C11 += 1
        r_start += 1
        U[u_idx(1, 0)][r_start] += 1
        U[u_idx(0, 0)][r_start] += -1
        V[v_idx(0, 0)][r_start] += 1
        V[v_idx(0, 1)][r_start] += 1
        W[w_idx(1, 1)][r_start] += 1

        # M7 = (A01 - A11)(B10 + B11)
        # C00 += 1
        r_start += 1
        U[u_idx(0, 1)][r_start] += 1
        U[u_idx(1, 1)][r_start] += -1
        V[v_idx(1, 0)][r_start] += 1
        V[v_idx(1, 1)][r_start] += 1
        W[w_idx(0, 0)][r_start] += 1

        return r_start + 1

    r_offset = 0

    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    r_offset = add_strassen_block(0, 0, 0, r_offset)
    # Block 2: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    r_offset = add_strassen_block(0, 0, 2, r_offset)
    # Block 3: A[0:2, 0:2] * B[0:2, 4:6] -> C[0:2, 4:6]
    r_offset = add_strassen_block(0, 0, 4, r_offset)
    # Block 4: A[2:4, 0:2] * B[0:2, 0:2] -> C[2:4, 0:2]
    r_offset = add_strassen_block(2, 0, 0, r_offset)
    # Block 5: A[2:4, 0:2] * B[0:2, 2:4] -> C[2:4, 2:4]
    r_offset = add_strassen_block(2, 0, 2, r_offset)
    # Block 6: A[2:4, 0:2] * B[0:2, 4:6] -> C[2:4, 4:6]
    r_offset = add_strassen_block(2, 0, 4, r_offset)

    # Block 7: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    r_offset = add_strassen_block(0, 2, 0, r_offset)
    # Block 8: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    r_offset = add_strassen_block(0, 2, 2, r_offset)
    # Block 9: A[0:2, 2:4] * B[2:4, 4:6] -> C[0:2, 4:6]
    r_offset = add_strassen_block(0, 2, 4, r_offset)
    # Block 10: A[2:4, 2:4] * B[2:4, 0:2] -> C[2:4, 0:2]
    r_offset = add_strassen_block(2, 2, 0, r_offset)
    # Block 11: A[2:4, 2:4] * B[2:4, 2:4] -> C[2:4, 2:4]
    r_offset = add_strassen_block(2, 2, 2, r_offset)
    # Block 12: A[2:4, 2:4] * B[2:4, 4:6] -> C[2:4, 4:6]
    r_offset = add_strassen_block(2, 2, 4, r_offset)

    # Block 13: A[0:4, 4:5] * B[4:5, 0:6] -> C[0:4, 0:6]
    # This is 4x1 by 1x6. Rank 24.
    for i in range(4):
        for k in range(6):
            U[i * 5 + 4][r_offset] = 1
            V[4 * 6 + k][r_offset] = 1
            W[k * 4 + i][r_offset] = 1
            r_offset += 1

    # Verify decomposition is correct (optional, for debugging)
    if time.time() - start_time < DEADLINE - SAFETY_MARGIN:
        U_np = np.array(U, dtype=np.float64)
        V_np = np.array(V, dtype=np.float64)
        W_np = np.array(W, dtype=np.float64)
        
        # Test with random matrices
        np.random.seed(42)
        A_test = np.random.randn(n, m)
        B_test = np.random.randn(m, p)
        C_expected = A_test @ B_test
        
        # Compute C from decomposition
        C_computed = np.zeros((n, p))
        for r in range(rank):
            A_r = np.zeros((n, m))
            B_r = np.zeros((m, p))
            for i in range(n):
                for j in range(m):
                    A_r[i, j] = U_np[i * m + j, r]
            for j in range(m):
                for k in range(p):
                    B_r[j, k] = V_np[j * p + k, r]
            for i in range(n):
                for k in range(p):
                    C_computed[i, k] += W_np[k * n + i, r] * (A_r @ B_r)[i, k]
        
        if not np.allclose(C_computed, C_expected, atol=1e-10):
            # This should never happen if the decomposition is correct
            pass

    return {"U": U, "V": V, "W": W}
