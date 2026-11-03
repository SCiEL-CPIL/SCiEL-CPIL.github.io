import time
import numpy as np

DIMENSIONS = [3, 4, 5]
DEADLINE = 140

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Schoolbook decomposition as fallback
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    # Try Strassen-based decomposition
    result = _try_strassen_decomposition(n, m, p)
    if result is not None:
        best_result = result
        
    return best_result

def _try_strassen_decomposition(n, m, p):
    if n != 3 or m != 4 or p != 5:
        return None
    
    # Strassen decomposition for 2x2x2
    # M1 = (A11+A22)(B22-B21)
    # M2 = (A11+A12)B22
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A21)B11
    # M6 = (A21+A22)(B11-B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    strassen_U = np.array([
        [1, 1, 1, 0, 1, 0, 0],  # A11
        [0, 1, 0, 0, 0, 1, 1],  # A12
        [1, 0, 0, 1, 1, 1, -1], # A21
        [1, 0, 0, 1, 0, 1, -1]  # A22
    ])
    
    strassen_V = np.array([
        [-1, 0, 0, -1, 1, -1, 1], # B11
        [1, 0, 1, 0, 0, 1, 1],    # B12
        [-1, 0, 0, 1, 0, 0, 1],   # B21
        [1, 1, -1, 0, 0, 0, 0]    # B22
    ])
    
    strassen_W = np.array([
        [1, 0, 0, 1, -1, 0, 1],   # C11
        [0, 0, 1, 0, 1, 0, 0],    # C12
        [0, 1, 0, 1, 0, 0, 0],    # C21
        [1, -1, 1, 0, 0, 1, 0]    # C22
    ])
    
    # Use 4 disjoint 2x2x2 Strassen boxes for i={0,1}, all j, k={0,1,2,3}
    # Plus schoolbook for i=2, all j, all k
    total_rank = 4 * 7 + 20  # 28 + 20 = 48
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col = 0
    
    # Box 1: i={0,1}, j={0,1}, k={0,1}
    for r in range(7):
        for i_local in range(2):
            for j_local in range(2):
                U[i_local * m + j_local, col + r] = strassen_U[i_local * 2 + j_local, r]
        for j_local in range(2):
            for k_local in range(2):
                V[j_local * p + k_local, col + r] = strassen_V[j_local * 2 + k_local, r]
        for k_local in range(2):
            for i_local in range(2):
                W[k_local * n + i_local, col + r] = strassen_W[k_local * 2 + i_local, r]
    col += 7
    
    # Box 2: i={0,1}, j={0,1}, k={2,3}
    for r in range(7):
        for i_local in range(2):
            for j_local in range(2):
                U[i_local * m + j_local, col + r] = strassen_U[i_local * 2 + j_local, r]
        for j_local in range(2):
            for k_local in range(2):
                V[j_local * p + (k_local + 2), col + r] = strassen_V[j_local * 2 + k_local, r]
        for k_local in range(2):
            for i_local in range(2):
                W[(k_local + 2) * n + i_local, col + r] = strassen_W[k_local * 2 + i_local, r]
    col += 7
    
    # Box 3: i={0,1}, j={2,3}, k={0,1}
    for r in range(7):
        for i_local in range(2):
            for j_local in range(2):
                U[i_local * m + (j_local + 2), col + r] = strassen_U[i_local * 2 + j_local, r]
        for j_local in range(2):
            for k_local in range(2):
                V[(j_local + 2) * p + k_local, col + r] = strassen_V[j_local * 2 + k_local, r]
        for k_local in range(2):
            for i_local in range(2):
                W[k_local * n + i_local, col + r] = strassen_W[k_local * 2 + i_local, r]
    col += 7
    
    # Box 4: i={0,1}, j={2,3}, k={2,3}
    for r in range(7):
        for i_local in range(2):
            for j_local in range(2):
                U[i_local * m + (j_local + 2), col + r] = strassen_U[i_local * 2 + j_local, r]
        for j_local in range(2):
            for k_local in range(2):
                V[(j_local + 2) * p + (k_local + 2), col + r] = strassen_V[j_local * 2 + k_local, r]
        for k_local in range(2):
            for i_local in range(2):
                W[(k_local + 2) * n + i_local, col + r] = strassen_W[k_local * 2 + i_local, r]
    col += 7
    
    # Remaining: i=2, j=0..3, k=0..4 (20 elements)
    for i in [2]:
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    if col != total_rank:
        return None
    
    if not _verify(U, V, W, n, m, p):
        return None
        
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist(),
        "rank": total_rank
    }

def _verify(U, V, W, n, m, p):
    np.random.seed(0)
    A = np.random.rand(n, m)
    B = np.random.rand(m, p)
    C_true = A @ B
    
    C_comp = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            val = 0.0
            for r in range(U.shape[1]):
                sum_U = 0.0
                for j in range(m):
                    sum_U += U[i * m + j, r] * A[i, j]
                sum_V = 0.0
                for j in range(m):
                    sum_V += V[j * p + k, r] * B[j, k]
                val += W[k * n + i, r] * sum_U * sum_V
            C_comp[i, k] = val
            
    if not np.allclose(C_true, C_comp, atol=1e-9):
        return False
    return True

def _verify_decomposition(U, V, W, n, m, p):
    return _verify(U, V, W, n, m, p)
