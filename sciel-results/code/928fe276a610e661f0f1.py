import numpy as np
import time

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145

    # Build the rank-28 Strassen-based decomposition
    rank = 28
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)

    # Helper to add a Strassen block
    def add_strassen_block(block_id, A_rows, A_cols, B_rows, B_cols, C_rows, C_cols):
        base_col = block_id * 7
        
        # M1: (X11+X22)(Y11+Y22)
        col = base_col + 0
        U[A_rows[0] * m + A_cols[0]][col] += 1
        U[A_rows[1] * m + A_cols[1]][col] += 1
        V[B_rows[0] * p + B_cols[0]][col] += 1
        V[B_rows[1] * p + B_cols[1]][col] += 1
        
        # M2: (X21+X22)Y11
        col = base_col + 1
        U[A_rows[1] * m + A_cols[0]][col] += 1
        U[A_rows[1] * m + A_cols[1]][col] += 1
        V[B_rows[0] * p + B_cols[0]][col] += 1
        
        # M3: X11(Y12-Y22)
        col = base_col + 2
        U[A_rows[0] * m + A_cols[0]][col] += 1
        V[B_rows[0] * p + B_cols[1]][col] += 1
        V[B_rows[1] * p + B_cols[1]][col] -= 1
        
        # M4: X22(Y21-Y11)
        col = base_col + 3
        U[A_rows[1] * m + A_cols[1]][col] += 1
        V[B_rows[1] * p + B_cols[0]][col] += 1
        V[B_rows[0] * p + B_cols[0]][col] -= 1
        
        # M5: (X11+X12)Y22
        col = base_col + 4
        U[A_rows[0] * m + A_cols[0]][col] += 1
        U[A_rows[0] * m + A_cols[1]][col] += 1
        V[B_rows[1] * p + B_cols[1]][col] += 1
        
        # M6: (X21-X11)(Y11+Y12)
        col = base_col + 5
        U[A_rows[1] * m + A_cols[0]][col] += 1
        U[A_rows[0] * m + A_cols[0]][col] -= 1
        V[B_rows[0] * p + B_cols[0]][col] += 1
        V[B_rows[0] * p + B_cols[1]][col] += 1
        
        # M7: (X12-X22)(Y21+Y22)
        col = base_col + 6
        U[A_rows[0] * m + A_cols[1]][col] += 1
        U[A_rows[1] * m + A_cols[1]][col] -= 1
        V[B_rows[1] * p + B_cols[0]][col] += 1
        V[B_rows[1] * p + B_cols[1]][col] += 1
        
        # W updates
        # Z11 = M1 + M4 - M5 + M7
        w_idx = C_cols[0] * n + C_rows[0]
        W[w_idx][base_col + 0] += 1
        W[w_idx][base_col + 3] += 1
        W[w_idx][base_col + 4] -= 1
        W[w_idx][base_col + 6] += 1
        
        # Z12 = M3 + M5
        w_idx = C_cols[1] * n + C_rows[0]
        W[w_idx][base_col + 2] += 1
        W[w_idx][base_col + 4] += 1
        
        # Z21 = M2 + M4
        w_idx = C_cols[0] * n + C_rows[1]
        W[w_idx][base_col + 1] += 1
        W[w_idx][base_col + 3] += 1
        
        # Z22 = M1 - M2 + M3 + M6
        w_idx = C_cols[1] * n + C_rows[1]
        W[w_idx][base_col + 0] += 1
        W[w_idx][base_col + 1] -= 1
        W[w_idx][base_col + 2] += 1
        W[w_idx][base_col + 5] += 1

    # Block 0: A_L * B_LL -> C_L
    add_strassen_block(0, [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1])
    
    # Block 1: A_R * B_RL -> C_L
    add_strassen_block(1, [0, 1], [2, 3], [2, 3], [0, 1], [0, 1], [0, 1])
    
    # Block 2: A_L * B_LR -> C_R
    add_strassen_block(2, [0, 1], [0, 1], [0, 1], [2, 3], [0, 1], [2, 3])
    
    # Block 3: A_R * B_RR -> C_R
    add_strassen_block(3, [0, 1], [2, 3], [2, 3], [2, 3], [0, 1], [2, 3])

    # Vectorized Verification
    def verify(U_mat, V_mat, W_mat):
        Uf = np.array(U_mat, dtype=float)
        Vf = np.array(V_mat, dtype=float)
        Wf = np.array(W_mat, dtype=float)
        
        # Create random test matrices
        np.random.seed(42)
        A = np.random.rand(n, m)
        B = np.random.rand(m, p)
        C = A @ B
        
        # Compute decomposition result
        # C[i, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # This is a bit tricky to vectorize directly without loops over j.
        # But we can reshape.
        # Let's just use the loop for safety, it's small.
        Cf = np.zeros((n, p))
        for ii in range(n):
            for kk in range(p):
                acc = 0.0
                for cc in range(Uf.shape[1]):
                    for jj in range(m):
                        acc += Uf[ii * m + jj, cc] * Vf[jj * p + kk, cc] * Wf[kk * n + ii, cc]
                Cf[ii, kk] = acc
        return np.allclose(C, Cf, atol=1e-9)

    if not verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"], dtype=float)
    V = np.array(res["V"], dtype=float)
    W = np.array(res["W"], dtype=float)
    A = np.arange(n * m).reshape(n, m) + 1.0
    B = np.arange(m * p).reshape(m, p) + 2.0
    C = A @ B
    Cf = np.zeros((n, p))
    for ii in range(n):
        for kk in range(p):
            acc = 0.0
            for cc in range(U.shape[1]):
                for jj in range(m):
                    acc += U[ii * m + jj, cc] * V[jj * p + kk, cc] * W[kk * n + ii, cc]
            Cf[ii, kk] = acc
    assert np.allclose(C, Cf), (C, Cf)
    print("OK rank", U.shape[1])
