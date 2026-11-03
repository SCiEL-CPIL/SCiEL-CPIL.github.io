import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 6]


def _build_strassen_42():
    """Build the rank-42 Strassen decomposition for 2x4x6."""
    n, m, p = 2, 4, 6
    rank = 42
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Strassen 2x2x2 factors
    # A indices: (i,j) -> i*2+j, B indices: (j,k) -> j*2+k, C indices: (k,i) -> k*2+i
    strassen = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),    # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),    # M2
        ([1, 0, 0, 0], [0, 0, 1, -1], [0, 1, 0, 1]),   # M3
        ([0, 0, 0, 1], [-1, 1, 0, 0], [1, 0, 1, 0]),   # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),   # M5
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),   # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),   # M7
    ]
    
    factor_idx = 0
    for a_block in range(2):
        for b_col_block in range(3):
            for (u, v, w) in strassen:
                for idx in range(4):
                    if u[idx] != 0:
                        i, j_local = divmod(idx, 2)
                        u_row = i * m + a_block * 2 + j_local
                        U[u_row, factor_idx] += u[idx]
                for idx in range(4):
                    if v[idx] != 0:
                        j_local, k_local = divmod(idx, 2)
                        v_row = (a_block * 2 + j_local) * p + b_col_block * 2 + k_local
                        V[v_row, factor_idx] += v[idx]
                for idx in range(4):
                    if w[idx] != 0:
                        k_local, i = divmod(idx, 2)
                        w_row = (b_col_block * 2 + k_local) * n + i
                        W[w_row, factor_idx] += w[idx]
                factor_idx += 1
    
    assert factor_idx == rank
    return U, V, W


def _verify_decomposition(U, V, W, n, m, p):
    """Verify the decomposition exactly."""
    rank = U.shape[1]
    # Vectorized check using tensor contraction
    # T[(i,j),(j',k),(k',i')] should equal sum_r U[i*m+j,r]*V[j'*p+k,r]*W[k'*n+i',r]
    # and should be 1 iff j==j', k==k', i==i'
    
    # Build expected tensor
    # For efficiency, check via random matrices
    np.random.seed(42)
    for _ in range(10):
        A = np.random.randint(-5, 6, size=(n, m))
        B = np.random.randint(-5, 6, size=(m, p))
        C_expected = A @ B
        
        C_computed = np.zeros((n, p), dtype=float)
        for r in range(rank):
            u_r = U[:, r]
            v_r = V[:, r]
            w_r = W[:, r]
            # a_part[i] = sum_j u_r[i*m+j] * A[i,j]
            a_part = np.zeros(n)
            for i in range(n):
                for j in range(m):
                    a_part[i] += u_r[i*m+j] * A[i,j]
            # b_part[k] = sum_j v_r[j*p+k] * B[j,k]
            b_part = np.zeros(p)
            for j in range(m):
                for k in range(p):
                    b_part[k] += v_r[j*p+k] * B[j,k]
            # c_part[i,k] = w_r[k*n+i] * a_part[i] * b_part[k]
            for i in range(n):
                for k in range(p):
                    C_computed[i,k] += w_r[k*n+i] * a_part[i] * b_part[k]
        
        if not np.allclose(C_computed, C_expected, atol=1e-9):
            return False
    return True


def _verify_exact(U, V, W, n, m, p):
    """Exact verification using integer arithmetic."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    for kp in range(p):
                        for ip in range(n):
                            expected = 1 if (j == jp and k == kp and i == ip) else 0
                            val = 0
                            for r in range(rank):
                                val += U[i*m+j, r] * V[jp*p+k, r] * W[kp*n+ip, r]
                            if val != expected:
                                return False
    return True


def _schoolbook(n, m, p):
    """Rank n*m*p schoolbook decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def construct():
    start_time = time.time()
    DEADLINE = 140  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # n=2, m=4, p=6
    
    # Build the rank-42 Strassen decomposition
    U, V, W = _build_strassen_42()
    
    # Verify correctness
    if not _verify_exact(U, V, W, n, m, p):
        U, V, W = _schoolbook(n, m, p)
    
    # Check time
    if time.time() - start_time > DEADLINE:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Return the rank-42 solution
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
