import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def _verify_decomposition(U, V, W, n, m, p):
    """Verify sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Create index arrays
    i_idx = np.repeat(np.arange(n), m * p)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)
    k_idx = np.tile(np.arange(p), n * m)
    
    u_idx = i_idx * m + j_idx
    v_idx = j_idx * p + k_idx
    w_idx = k_idx * n + i_idx
    
    # Compute sum for each (i,j,k)
    # We need sum_r U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
    # This is a batched operation.
    # U[u_idx] is (n*m*p, rank)
    # V[v_idx] is (n*m*p, rank)
    # W[w_idx] is (n*m*p, rank)
    
    prod = U_arr[u_idx] * V_arr[v_idx] * W_arr[w_idx]
    sums = np.sum(prod, axis=1)
    
    if not np.allclose(sums, 1.0, atol=1e-9):
        return False
    return True

def _build_rank28():
    """Build the known rank-28 decomposition."""
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    
    # Part 1: j=2, k=0..4 (10 terms)
    for i in range(n):
        for k in range(p):
            U[i * m + 2][r] = 1
            V[2 * p + k][r] = 1
            W[k * n + i][r] = 1
            r += 1
            
    # Part 2: j=0,1, k=4 (4 terms)
    for i in range(n):
        for j in range(2):
            U[i * m + j][r] = 1
            V[j * p + 4][r] = 1
            W[4 * n + i][r] = 1
            r += 1
            
    # Part 3: Strassen for j=0,1, k=0,1 (7 terms)
    # M1
    U[0][r] = 1; U[4][r] = 1
    V[0][r] = 1; V[6][r] = 1
    W[0][r] = 1; W[3][r] = 1
    r += 1
    # M2
    U[3][r] = 1; U[4][r] = 1
    V[0][r] = 1
    W[1][r] = 1; W[3][r] = -1
    r += 1
    # M3
    U[0][r] = 1
    V[1][r] = 1; V[6][r] = -1
    W[2][r] = 1; W[3][r] = 1
    r += 1
    # M4
    U[4][r] = 1
    V[5][r] = 1; V[0][r] = -1
    W[0][r] = 1; W[1][r] = 1
    r += 1
    # M5
    U[0][r] = 1; U[1][r] = 1
    V[6][r] = 1
    W[0][r] = -1; W[2][r] = 1
    r += 1
    # M6
    U[3][r] = 1; U[0][r] = -1
    V[0][r] = 1; V[1][r] = 1
    W[3][r] = 1
    r += 1
    # M7
    U[1][r] = 1; U[4][r] = -1
    V[5][r] = 1; V[6][r] = 1
    W[0][r] = 1
    r += 1
    
    # Part 4: Strassen for j=0,1, k=2,3 (7 terms)
    # M1
    U[0][r] = 1; U[4][r] = 1
    V[2][r] = 1; V[8][r] = 1
    W[4][r] = 1; W[7][r] = 1
    r += 1
    # M2
    U[3][r] = 1; U[4][r] = 1
    V[2][r] = 1
    W[5][r] = 1; W[7][r] = -1
    r += 1
    # M3
    U[0][r] = 1
    V[3][r] = 1; V[8][r] = -1
    W[6][r] = 1; W[7][r] = 1
    r += 1
    # M4
    U[4][r] = 1
    V[7][r] = 1; V[2][r] = -1
    W[4][r] = 1; W[5][r] = 1
    r += 1
    # M5
    U[0][r] = 1; U[1][r] = 1
    V[8][r] = 1
    W[4][r] = -1; W[6][r] = 1
    r += 1
    # M6
    U[3][r] = 1; U[0][r] = -1
    V[2][r] = 1; V[3][r] = 1
    W[7][r] = 1
    r += 1
    # M7
    U[1][r] = 1; U[4][r] = -1
    V[7][r] = 1; V[8][r] = 1
    W[4][r] = 1
    r += 1
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    
    U, V, W = _build_rank28()
    
    if not _verify_decomposition(U, V, W, n, m, p):
        raise ValueError("Base decomposition verification failed")
        
    return {"U": U, "V": V, "W": W}
