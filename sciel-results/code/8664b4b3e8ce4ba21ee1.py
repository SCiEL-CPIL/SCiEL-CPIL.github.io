import numpy as np
import time

DIMENSIONS = [6, 6, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 100  # 100s deadline, 20s safety margin

    n, m, p = DIMENSIONS
    # Block decomposition with block size 3
    block_size = 3
    n_blocks = n // block_size  # 2

    # Total rank is n_blocks^3 * 27 (schoolbook for 3x3)
    rank = n_blocks**3 * 27

    U = np.zeros((n * m, rank), dtype=np.int8)
    V = np.zeros((m * p, rank), dtype=np.int8)
    W = np.zeros((p * n, rank), dtype=np.int8)

    col = 0

    # Helper to get index in U for block (ib, kb) and local (r, c)
    def u_idx(ib, kb, r, c):
        return (ib * block_size + r) * m + (kb * block_size + c)

    # Helper to get index in V for block (kb, jb) and local (r, c)
    def v_idx(kb, jb, r, c):
        return (kb * block_size + r) * p + (jb * block_size + c)

    # Helper to get index in W for block (ib, jb) and local (r, c)
    # W index is k*n + i, where k is col of C, i is row of C
    def w_idx(ib, jb, r, c):
        return (jb * block_size + c) * n + (ib * block_size + r)

    # Iterate through all block multiplications
    for ib in range(n_blocks):
        if time.time() > deadline:
            break
        for jb in range(n_blocks):
            if time.time() > deadline:
                break
            for kb in range(n_blocks):
                if time.time() > deadline:
                    break
                # This block multiplication A[ib][kb] * B[kb][jb] contributes to C[ib][jb]
                # Schoolbook: for each (i,k) in A block and (k,j) in B block
                for i in range(block_size):
                    for k in range(block_size):
                        for j in range(block_size):
                            U[u_idx(ib, kb, i, k), col] = 1
                            V[v_idx(kb, jb, k, j), col] = 1
                            W[w_idx(ib, jb, i, j), col] = 1
                            col += 1

    # Verify correctness if time permits
    if time.time() < deadline - 5:
        try:
            U_arr = np.array(U, dtype=float)
            V_arr = np.array(V, dtype=float)
            W_arr = np.array(W, dtype=float)

            # Test with random matrices
            np.random.seed(42)
            A = np.random.randn(n, m)
            B = np.random.randn(m, p)
            C_expected = A @ B

            # Compute using decomposition
            U_mat = U_arr.reshape(n * m, -1)
            V_mat = V_arr.reshape(m * p, -1)
            W_mat = W_arr.reshape(p * n, -1)

            # C[i,j] = sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r]
            C_computed = np.zeros((n, p))
            for i in range(n):
                for j in range(p):
                    s = 0
                    for k in range(m):
                        for r in range(rank):
                            s += U_mat[i*m+k, r] * V_mat[k*p+j, r] * W_mat[j*n+i, r]
                    C_computed[i, j] = s

            if not np.allclose(C_computed, C_expected, atol=1e-10):
                print("WARNING: Decomposition verification failed")
        except Exception as e:
            print(f"Verification error: {e}")

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
