import time
import numpy as np

def construct():
    t0 = time.time()
    tmax = t0 + 115.0  # Safety margin before 120s limit
    
    n = m = p = 5
    U_cols = []
    V_cols = []
    W_cols = []
    
    # Block decomposition: split indices into {0..3} and {4}
    # Part 1: Core 4x4x4 naive multiplication (rank 16)
    for i in range(4):
        for j in range(4):
            for k in range(4):
                U_cols.append({i * m + j: 1})
                V_cols.append({j * p + k: 1})
                W_cols.append({k * n + i: 1})
    
    # Part 2: A[0..3,4] * B[4,0..3] -> C[0..3,0..3] (rank 16)
    for i in range(4):
        for k in range(4):
            U_cols.append({i * m + 4: 1})
            V_cols.append({4 * p + k: 1})
            W_cols.append({k * n + i: 1})
    
    # Part 3: A[0..3,0..3] * B[0..3,4] -> C[0..3,4] (rank 16)
    for i in range(4):
        for j in range(4):
            U_cols.append({i * m + j: 1})
            V_cols.append({j * p + 4: 1})
            W_cols.append({4 * n + i: 1})
    
    # Part 4: A[0..3,4] * B[4,4] -> C[0..3,4] (rank 4)
    for i in range(4):
        U_cols.append({i * m + 4: 1})
        V_cols.append({4 * p + 4: 1})
        W_cols.append({4 * n + i: 1})
    
    # Part 5: A[4,0..3] * B[0..3,0..3] -> C[4,0..3] (rank 16)
    for j in range(4):
        for k in range(4):
            U_cols.append({4 * m + j: 1})
            V_cols.append({j * p + k: 1})
            W_cols.append({k * n + 4: 1})
    
    # Part 6: A[4,4] * B[4,0..3] -> C[4,0..3] (rank 4)
    for k in range(4):
        U_cols.append({4 * m + 4: 1})
        V_cols.append({4 * p + k: 1})
        W_cols.append({k * n + 4: 1})
    
    # Part 7: A[4,0..3] * B[0..3,4] -> C[4,4] (rank 4)
    for j in range(4):
        U_cols.append({4 * m + j: 1})
        V_cols.append({j * p + 4: 1})
        W_cols.append({4 * n + 4: 1})
    
    # Part 8: A[4,4] * B[4,4] -> C[4,4] (rank 1)
    U_cols.append({4 * m + 4: 1})
    V_cols.append({4 * p + 4: 1})
    W_cols.append({4 * n + 4: 1})
    
    rank = len(U_cols)
    
    # Build dense matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r in range(rank):
        for idx, val in U_cols[r].items():
            U[idx][r] = val
        for idx, val in V_cols[r].items():
            V[idx][r] = val
        for idx, val in W_cols[r].items():
            W[idx][r] = val
    
    # Verify correctness using numpy
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    U_r = U_np.reshape(n, m, rank)
    V_r = V_np.reshape(m, p, rank)
    W_r = W_np.reshape(p, n, rank)
    
    # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r, optimize=True)
    
    # Check that T equals the matrix multiplication tensor
    # For matrix mult: C[i,k] = sum_j A[i,j]*B[j,k]
    # The structural tensor should satisfy: T[i,j,k] = 1 for all i,j,k
    # But actually we need to verify the bilinear identity
    
    # Quick sanity check: verify a few random matrices
    np.random.seed(42)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test
    
    # Compute C from decomposition
    C_decomp = np.zeros((n, p))
    for c in range(rank):
        a_part = U_r[:, :, c] @ A_test  # shape (n,)
        b_part = B_test @ V_r[:, :, c]  # shape (p,)
        C_decomp += W_r[:, :, c] * np.outer(a_part, b_part)
    
    if not np.allclose(C_decomp, C_expected, atol=1e-9):
        # Fallback: return schoolbook if verification fails
        rank_sb = n * m * p
        U_sb = [[0] * rank_sb for _ in range(n * m)]
        V_sb = [[0] * rank_sb for _ in range(m * p)]
        W_sb = [[0] * rank_sb for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i*m+j][col] = 1
                    V_sb[j*p+k][col] = 1
                    W_sb[k*n+i][col] = 1
                    col += 1
        return {"U": U_sb, "V": V_sb, "W": W_sb}
    
    return {"U": U, "V": V, "W": W}
