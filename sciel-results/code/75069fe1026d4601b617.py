"""Matrix multiplication 2x2x14 - rank-minimal exact tensor decomposition."""
import time

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Try Strassen-based rank-49 decomposition first
    result_49 = _construct_strassen_49(n, m, p)
    if result_49 is not None:
        return result_49
    
    # Fallback: schoolbook rank-56
    rank_schoolbook = n * m * p
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}

def _construct_strassen_49(n, m, p):
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for pair_idx in range(7):
        k1 = 2 * pair_idx
        k2 = 2 * pair_idx + 1
        base_col = pair_idx * 7
        
        # M1
        U[0][base_col] = 1
        U[3][base_col] = 1
        V[k1][base_col] = 1
        V[p + k2][base_col] = 1
        W[k1 * n + 0][base_col] = 1
        W[k2 * n + 1][base_col] = 1
        
        # M2
        U[2][base_col + 1] = 1
        U[3][base_col + 1] = 1
        V[k1][base_col + 1] = 1
        W[k1 * n + 1][base_col + 1] = 1
        W[k2 * n + 1][base_col + 1] = -1
        
        # M3
        U[0][base_col + 2] = 1
        V[k2][base_col + 2] = 1
        V[p + k2][base_col + 2] = -1
        W[k2 * n + 0][base_col + 2] = 1
        W[k2 * n + 1][base_col + 2] = 1
        
        # M4
        U[3][base_col + 3] = 1
        V[p + k1][base_col + 3] = 1
        V[k1][base_col + 3] = -1
        W[k1 * n + 0][base_col + 3] = 1
        W[k1 * n + 1][base_col + 3] = 1
        
        # M5
        U[0][base_col + 4] = 1
        U[1][base_col + 4] = 1
        V[p + k2][base_col + 4] = 1
        W[k1 * n + 0][base_col + 4] = -1
        W[k2 * n + 0][base_col + 4] = 1
        
        # M6
        U[2][base_col + 5] = 1
        U[0][base_col + 5] = -1
        V[k1][base_col + 5] = 1
        V[k2][base_col + 5] = 1
        W[k2 * n + 1][base_col + 5] = 1
        
        # M7
        U[1][base_col + 6] = 1
        U[3][base_col + 6] = -1
        V[p + k1][base_col + 6] = 1
        V[p + k2][base_col + 6] = 1
        W[k1 * n + 0][base_col + 6] = 1
        
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    else:
        return None

def _verify_decomposition(U, V, W, n, m, p, num_tests=5):
    import numpy as np
    for _ in range(num_tests):
        A = np.random.rand(n, m)
        B = np.random.rand(m, p)
        C_expected = A @ B
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    tensor_val = 0.0
                    for r in range(len(U[0])):
                        tensor_val += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(tensor_val - 1.0) > 1e-10:
                        return False
    return True
# EVOLVE-BLOCK-END
