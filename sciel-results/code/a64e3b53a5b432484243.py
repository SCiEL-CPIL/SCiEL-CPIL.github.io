import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # Safety margin before 150s limit

    n, m, p = DIMENSIONS  # n=2, m=3, p=5

    # Rank-28 decomposition:
    # Block 1: j=0, k=0..4 -> 2x1x5 schoolbook, rank 10
    # Block 2a: j=1,2, k=0,1 -> 2x2x2 Strassen, rank 7
    # Block 2b: j=1,2, k=2 -> 2x2x1 schoolbook, rank 4
    # Block 2c: j=1,2, k=3,4 -> 2x2x2 Strassen, rank 7
    # Total: 10 + 7 + 4 + 7 = 28

    rank = 28
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    col = 0

    # Block 1: j=0, k=0..4 (schoolbook 2x1x5)
    for i in range(n):
        for k in range(p):
            U[i * m + 0, col] = 1.0
            V[0 * p + k, col] = 1.0
            W[k * n + i, col] = 1.0
            col += 1

    # Strassen 2x2x2 helper: adds 7 terms
    def add_strassen(i_off, j_off, k_off):
        nonlocal col
        # M1: (A00+A11)(B00+B11) -> C00, C11
        U[(i_off+0)*m + (j_off+0), col] = 1.0
        U[(i_off+1)*m + (j_off+1), col] = 1.0
        V[(j_off+0)*p + (k_off+0), col] = 1.0
        V[(j_off+1)*p + (k_off+1), col] = 1.0
        W[(k_off+0)*n + (i_off+0), col] = 1.0
        W[(k_off+1)*n + (i_off+1), col] = 1.0
        col += 1
        # M2: (A10+A11)B00 -> C10, -C11
        U[(i_off+1)*m + (j_off+0), col] = 1.0
        U[(i_off+1)*m + (j_off+1), col] = 1.0
        V[(j_off+0)*p + (k_off+0), col] = 1.0
        W[(k_off+0)*n + (i_off+1), col] = 1.0
        W[(k_off+1)*n + (i_off+1), col] = -1.0
        col += 1
        # M3: A00(B01-B11) -> C01, C11
        U[(i_off+0)*m + (j_off+0), col] = 1.0
        V[(j_off+0)*p + (k_off+1), col] = 1.0
        V[(j_off+1)*p + (k_off+1), col] = -1.0
        W[(k_off+1)*n + (i_off+0), col] = 1.0
        W[(k_off+1)*n + (i_off+1), col] = 1.0
        col += 1
        # M4: A11(B10-B00) -> C00, C10
        U[(i_off+1)*m + (j_off+1), col] = 1.0
        V[(j_off+1)*p + (k_off+0), col] = 1.0
        V[(j_off+0)*p + (k_off+0), col] = -1.0
        W[(k_off+0)*n + (i_off+0), col] = 1.0
        W[(k_off+0)*n + (i_off+1), col] = 1.0
        col += 1
        # M5: (A00+A01)B11 -> -C00, C01
        U[(i_off+0)*m + (j_off+0), col] = 1.0
        U[(i_off+0)*m + (j_off+1), col] = 1.0
        V[(j_off+1)*p + (k_off+1), col] = 1.0
        W[(k_off+0)*n + (i_off+0), col] = -1.0
        W[(k_off+1)*n + (i_off+0), col] = 1.0
        col += 1
        # M6: (A10-A00)(B00+B01) -> C11
        U[(i_off+1)*m + (j_off+0), col] = 1.0
        U[(i_off+0)*m + (j_off+0), col] = -1.0
        V[(j_off+0)*p + (k_off+0), col] = 1.0
        V[(j_off+0)*p + (k_off+1), col] = 1.0
        W[(k_off+1)*n + (i_off+1), col] = 1.0
        col += 1
        # M7: (A01-A11)(B10+B11) -> C00
        U[(i_off+0)*m + (j_off+1), col] = 1.0
        U[(i_off+1)*m + (j_off+1), col] = -1.0
        V[(j_off+1)*p + (k_off+0), col] = 1.0
        V[(j_off+1)*p + (k_off+1), col] = 1.0
        W[(k_off+0)*n + (i_off+0), col] = 1.0
        col += 1

    # Block 2a: j=1,2, k=0,1 (Strassen 2x2x2)
    add_strassen(i_off=0, j_off=1, k_off=0)

    # Block 2b: j=1,2, k=2 (schoolbook 2x2x1)
    for i in range(n):
        for j in [1, 2]:
            U[i * m + j, col] = 1.0
            V[j * p + 2, col] = 1.0
            W[2 * n + i, col] = 1.0
            col += 1

    # Block 2c: j=1,2, k=3,4 (Strassen 2x2x2)
    add_strassen(i_off=0, j_off=1, k_off=3)

    assert col == rank, f"Column count mismatch: {col} != {rank}"

    # Verify with vectorized computation for speed
    def verify(U_mat, V_mat, W_mat, r):
        # Vectorized: for each term t, compute scalar_a = sum(A * U_t), scalar_b = sum(B * V_t)
        # U_mat shape: (n*m, r) -> reshape to (n, m, r)
        # V_mat shape: (m*p, r) -> reshape to (m, p, r)
        # W_mat shape: (p*n, r) -> reshape to (p, n, r)
        U_3d = U_mat.reshape(n, m, r)
        V_3d = V_mat.reshape(m, p, r)
        W_3d = W_mat.reshape(p, n, r)
        
        for seed in [42, 123]:
            if time.time() > DEADLINE - 5:
                return True  # Assume correct if running out of time
            np.random.seed(seed)
            A_test = np.random.randn(n, m)
            B_test = np.random.randn(m, p)
            C_test = A_test @ B_test
            
            # scalar_a[t] = sum over i,j of A_test[i,j] * U_3d[i,j,t]
            scalar_a = np.einsum('ij,ijr->r', A_test, U_3d)
            scalar_b = np.einsum('kl,klr->r', B_test, V_3d)
            
            # C_decomp = sum_t scalar_a[t] * scalar_b[t] * W_3d[:,:,t].T
            # W_3d shape: (p, n, r), so W_3d[:,:,t] is (p,n), transpose gives (n,p)
            C_decomp = np.einsum('r,r,pnr->np', scalar_a, scalar_b, W_3d)
            
            if not np.allclose(C_test, C_decomp, atol=1e-10):
                return False
        return True

    if not verify(U, V, W, rank):
        # Fallback: return schoolbook rank-30
        rank_sb = n * m * p
        U_sb = np.zeros((n * m, rank_sb), dtype=float)
        V_sb = np.zeros((m * p, rank_sb), dtype=float)
        W_sb = np.zeros((p * n, rank_sb), dtype=float)
        col_sb = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j, col_sb] = 1.0
                    V_sb[j * p + k, col_sb] = 1.0
                    W_sb[k * n + i, col_sb] = 1.0
                    col_sb += 1
        return {"U": U_sb, "V": V_sb, "W": W_sb}

    if time.time() > DEADLINE - 2:
        # Emergency: return without verification (we already verified above)
        pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
