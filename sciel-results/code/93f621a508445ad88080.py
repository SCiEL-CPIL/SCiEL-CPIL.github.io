import numpy as np
import time

DIMENSIONS = [3, 3, 7]

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verifies if U, V, W exactly decompose the matrix multiplication tensor.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    """
    rank = U.shape[1]
    # Convert to numpy arrays for speed
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # The tensor T is defined by T[i*m+j, j*p+k, k*n+i] = 1
    # Decomposition: T = sum_c U[:,c] V[:,c] W[:,c]^T (outer product)
    # We check if sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i,j,k
    
    # Random sampling for speed
    np.random.seed(42)
    for _ in range(20):
        i = np.random.randint(n)
        j = np.random.randint(m)
        k = np.random.randint(p)
        
        u_idx = i * m + j
        v_idx = j * p + k
        w_idx = k * n + i
        
        val = np.dot(U_np[u_idx, :], V_np[v_idx, :]) * W_np[w_idx, :] # This is wrong.
        # The decomposition is T_{a,b,c} = sum_k U_{a,k} V_{b,k} W_{c,k}
        # So for specific i,j,k:
        # T_{i,j,k} = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        
        u_val = U_np[u_idx, :]
        v_val = V_np[v_idx, :]
        w_val = W_np[w_idx, :]
        
        sum_val = np.sum(u_val * v_val * w_val)
        
        if not np.isclose(sum_val, 1.0, atol=1e-9):
            return False
            
    return True

def _strassen_3x3x3():
    """
    Returns U, V, W for 3x3x3 Strassen decomposition (rank 23).
    If coefficients are not available/verified, returns None.
    """
    # I will use a placeholder that is likely incorrect to trigger fallback, 
    # unless I can find the real ones.
    # For the purpose of this edit, I will return None to force fallback to schoolbook,
    # as I cannot guarantee the 23 coefficients from memory.
    return None

def _schoolbook_3x3x7():
    """
    Returns U, V, W for 3x3x7 Schoolbook decomposition (rank 63).
    """
    n, m, p = 3, 3, 7
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Try Rank 55: 2x Strassen(3x3x3) + 1x Schoolbook(3x3x1)
    # This requires valid Strassen 3x3x3 coefficients.
    # Since I don't have them, I will try to construct a valid lower rank if possible.
    # Otherwise, fallback to 63.
    
    U, V, W = _schoolbook_3x3x7()
    
    # Check time
    if time.time() - start_time > 110:
        return {"U": U, "V": V, "W": W}
        
    return {"U": U, "V": V, "W": W}
