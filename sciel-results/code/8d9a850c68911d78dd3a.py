"""Tensor decomposition for 3x4 @ 4x6 matrix multiplication."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact for the matrix multiplication tensor."""
    rank = len(V[0]) if V else 0
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Check with random inputs
    np.random.seed(42)
    for _ in range(10):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_true = A @ B
        C_test = np.zeros((n, p))
        
        # C[i,k] = sum_j sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
        # Efficiently:
        # For each r, compute S_r[i,k] = sum_j U[i*m+j, r] V[j*p+k, r]
        # Then C[i,k] += S_r[i,k] * W[k*n+i, r]
        
        for r in range(rank):
            u_col = U[:, r]
            v_col = V[:, r]
            w_col = W[:, r]
            
            # u_col is length n*m. Reshape to (n, m)
            U_r = u_col.reshape(n, m)
            # v_col is length m*p. Reshape to (m, p)
            V_r = v_col.reshape(m, p)
            # w_col is length p*n. Reshape to (p, n)
            W_r = w_col.reshape(p, n)
            
            # S_r = U_r @ V_r (n, p)
            S_r = U_r @ V_r
            # W_r is (p, n). We need W[k, i] to multiply S_r[i, k].
            # So we need W_r.T (n, p)
            C_test += S_r * W_r.T
            
        if not np.allclose(C_test, C_true, atol=1e-9):
            return False
    return True

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 6
    start_time = time.time()
    deadline = start_time + 130  # Safety margin

    # Strategy: Decompose into n independent 1xm x p multiplications.
    # Rank of 1xm x p tensor is m.
    # Total rank = n * m.
    # For n=3, m=4, p=6, rank = 12.
    
    R = n * m
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    
    # Indices for U: i*m + j. Column r.
    # Indices for V: j*p + k. Column r.
    # Indices for W: k*n + i. Column r.
    
    # We map the rank index r to (i, j) pair?
    # Let r = i * m + j.
    # For a fixed r, it corresponds to specific i and j.
    # U[i*m + j, r] should be 1 if (i,j) matches, else 0.
    # V[j*p + k, r] should be 1 if j matches, else 0.
    # W[k*n + i, r] should be 1 if i matches, else 0.
    
    for i in range(n):
        for j in range(m):
            r = i * m + j
            
            # U: only one non-zero entry at (i, j)
            U[i * m + j, r] = 1.0
            
            # V: non-zero for all k, at index (j, k)
            for k in range(p):
                V[j * p + k, r] = 1.0
            
            # W: non-zero for all k, at index (k, i)
            for k in range(p):
                W[k * n + i, r] = 1.0
                
    # Verify
    if not _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
        # Fallback to schoolbook
        rank_school = n * m * p
        U_school = np.zeros((n * m, rank_school))
        V_school = np.zeros((m * p, rank_school))
        W_school = np.zeros((p * n, rank_school))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i * m + j, c] = 1.0
                    V_school[j * p + k, c] = 1.0
                    W_school[k * n + i, c] = 1.0
                    c += 1
        return {
            "U": U_school.tolist(),
            "V": V_school.tolist(),
            "W": W_school.tolist()
        }
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

# EVOLVE-BLOCK-END
