import numpy as np
import time

DIMENSIONS = [4, 4, 9]

def _build_4x4x3_direct():
    """
    Build direct decomposition for 4x4 x 4x3 -> 4x3.
    Rank = 4*3 = 12 (one term per (i,j) pair in output, summing over k)
    Actually: for each output column j (0..2) and each k (0..3), we have a term.
    Total rank = 3 * 4 = 12.
    U: (16, 12), V: (12, 12), W: (12, 12)
    """
    rank = 12
    U = np.zeros((16, rank))
    V = np.zeros((12, rank))
    W = np.zeros((12, rank))
    
    for j in range(3):
        for k in range(4):
            r = j * 4 + k
            for i in range(4):
                U[i * 4 + k, r] = 1
            V[k * 3 + j, r] = 1
            for i in range(4):
                W[j * 4 + i, r] = 1
    
    return U, V, W, rank

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    # Decompose 4x4x9 as three 4x4x3 blocks
    U3, V3, W3, rank3 = _build_4x4x3_direct()
    
    total_rank = 3 * rank3  # 36
    U = np.zeros((16, total_rank))
    V = np.zeros((36, total_rank))
    W = np.zeros((36, total_rank))
    
    for block in range(3):
        base_r = block * rank3
        U[:, base_r:base_r+rank3] = U3
        for k in range(4):
            for j in range(3):
                V[k * 9 + (block*3 + j), base_r:base_r+rank3] = V3[k * 3 + j, :]
        for j in range(3):
            for i in range(4):
                W[(block*3 + j) * 4 + i, base_r:base_r+rank3] = W3[j * 4 + i, :]
    
    # Verification
    U_reshaped = U.reshape(4, 4, total_rank)
    V_reshaped = V.reshape(4, 9, total_rank)
    W_reshaped = W.reshape(9, 4, total_rank)
    
    A_test = np.random.randn(4, 4)
    B_test = np.random.randn(4, 9)
    C_expected = A_test @ B_test
    
    a_vecs = np.einsum('ikr,ik->ir', U_reshaped, A_test)
    b_vecs = np.einsum('kjr,kj->jr', V_reshaped, B_test)
    C_computed = np.einsum('jir,ir,jr->ij', W_reshaped, a_vecs, b_vecs)
    
    if not np.allclose(C_computed, C_expected, atol=1e-10):
        # Fallback to direct
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        idx = 0
        for i in range(n):
            for j in range(p):
                for k in range(m):
                    U[i * m + k, idx] = 1
                    V[k * p + j, idx] = 1
                    W[j * n + i, idx] = 1
                    idx += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
