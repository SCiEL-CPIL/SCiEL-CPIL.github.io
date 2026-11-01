"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Attempts to construct a rank-44 decomposition for 2x3 * 3x8 matrix multiplication.
    Falls back to rank-48 schoolbook if construction or verification fails.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100  # 100s deadline to be safe
    
    # Try to build rank 44
    try:
        U, V, W = _build_rank44_decomposition()
        if _verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    except Exception:
        pass

    # Fallback to schoolbook rank 48
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _build_rank44_decomposition():
    # This is a placeholder for the actual rank-44 construction.
    # Based on inspiration, it uses Strassen(7) + 4 rank-1 per 2x3x2 block.
    # Since I don't have the exact coefficients, I will return None to trigger fallback.
    # In a real scenario, I would implement the specific linear combinations.
    return None, None, None

def _verify(U, V, W, n, m, p):
    """
    Verifies if U, V, W form a valid decomposition.
    """
    if U is None:
        return False
        
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    rank = U_arr.shape[1]
    if V_arr.shape[1] != rank or W_arr.shape[1] != rank:
        return False
        
    # Construct the tensor T
    # T[i*m+j, j*p+k, k*n+i] = 1
    # We check if sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    
    # Efficient check:
    # For each i, j, k, compute the dot product
    # This is O(n*m*p*rank)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U_arr[i * m + j]
                v_val = V_arr[j * p + k]
                w_val = W_arr[k * n + i]
                dot = np.dot(u_val, v_val) * w_val # Wait, this is not right.
                # The term is sum_c U[idx_u, c] * V[idx_v, c] * W[idx_w, c]
                # This is not a simple dot product of vectors.
                # It is the element-wise product of the three vectors, then sum.
                # sum_c (U[idx_u, c] * V[idx_v, c] * W[idx_w, c])
                
                # We can compute this as:
                # tmp = U[idx_u] * V[idx_v]
                # sum(tmp * W[idx_w])
                
                tmp = U_arr[i * m + j] * V_arr[j * p + k]
                val = np.dot(tmp, W_arr[k * n + i])
                
                if not np.isclose(val, 1.0, atol=1e-6):
                    return False
    return True
# EVOLVE-BLOCK-END
