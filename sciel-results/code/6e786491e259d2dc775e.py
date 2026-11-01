import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 4, 4]
DEADLINE = 140.0

def _verify_decomposition(U, V, W, n=2, m=4, p=4):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    # Test with random matrices
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B
    C_computed = np.zeros((n, p))
    for r in range(U.shape[1]):
        u = U[:, r].reshape(n, m)
        v = V[:, r].reshape(m, p)
        w = W[:, r].reshape(p, n)  # W is indexed by [k*n + i], so reshape to (p, n)
        # C += u @ v * w element-wise? No...
        # Actually: C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # This is: C = sum_r (U_r outer V_r) weighted by W_r? No...
        # Let me re-derive: C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * V[j*p+k, r])
        # = sum_r W[k*n+i, r] * (U_r @ V_r)[i,k]
        # where U_r is n×m, V_r is m×p
        pass
    
    # Direct computation
    for i in range(n):
        for k in range(p):
            acc = 0.0
            for r in range(U.shape[1]):
                for j in range(m):
                    acc += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
            C_computed[i, k] = acc
    return np.allclose(C_expected, C_computed, atol=1e-10)

def _generate_strassen_decomposition():
    """Generate the rank-28 Strassen-based decomposition."""
    strassen_cols = [
        {0: [(0,1), (3,1)], 1: [(0,1), (3,1)], 2: [(0,1), (3,1)]},
        {0: [(2,1), (3,1)], 1: [(0,1)], 2: [(2,1), (3,-1)]},
        {0: [(0,1)], 1: [(1,1), (3,-1)], 2: [(1,1), (3,1)]},
        {0: [(3,1)], 1: [(2,1), (0,-1)], 2: [(0,1), (2,1)]},
        {0: [(0,1), (1,1)], 1: [(3,1)], 2: [(0,-1), (1,1)]},
        {0: [(2,1), (0,-1)], 1: [(0,1), (1,1)], 2: [(3,1)]},
        {0: [(1,1), (3,-1)], 1: [(2,1), (3,1)], 2: [(0,1)]}
    ]
    
    U = np.zeros((8, 28), dtype=int)
    V = np.zeros((16, 28), dtype=int)
    W = np.zeros((8, 28), dtype=int)
    
    def map_A(local_idx, offset):
        r = local_idx // 2
        c = local_idx % 2
        return r * 4 + (offset * 2 + c)
        
    def map_B(local_idx, row_off, col_off):
        r = local_idx // 2
        c = local_idx % 2
        return (row_off * 2 + r) * 4 + (col_off * 2 + c)
        
    def map_C(local_idx, col_off):
        r = local_idx // 2
        c = local_idx % 2
        return (col_off * 2 + c) * 2 + r

    blocks = [
        (0, 0, 0),   # A_L, B_TL, C_L
        (1, 1, 0),   # A_R, B_BL, C_L
        (0, 0, 1),   # A_L, B_TR, C_R
        (1, 1, 1),   # A_R, B_BR, C_R
    ]
    
    for block_idx, (a_off, b_row_off, c_col_off) in enumerate(blocks):
        b_col_off = 0 if c_col_off == 0 else 1
        for j, col_def in enumerate(strassen_cols):
            col_idx = block_idx * 7 + j
            for loc_idx, coeff in col_def[0]:
                U[map_A(loc_idx, a_off), col_idx] += coeff
            for loc_idx, coeff in col_def[1]:
                V[map_B(loc_idx, b_row_off, b_col_off), col_idx] += coeff
            for loc_idx, coeff in col_def[2]:
                W[map_C(loc_idx, c_col_off), col_idx] += coeff
    
    return U, V, W

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Start with rank-28 Strassen decomposition
    U, V, W = _generate_strassen_decomposition()
    
    # Try to reduce rank by finding linear dependencies
    # The idea: if we can express one column as a linear combination of others
    # in a way that preserves the tensor decomposition, we can remove it.
    
    # For now, return the rank-28 solution
    if time.time() - start_time > DEADLINE:
        pass
    
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
