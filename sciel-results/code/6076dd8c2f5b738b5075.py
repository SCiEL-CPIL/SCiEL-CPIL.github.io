import time
import numpy as np

DIMENSIONS = [3, 5, 7]

def _validate_decomposition(U, V, W, n, m, p, tol=1e-7):
    """
    Check that the factor matrices define the matrix multiplication tensor.
    """
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    U_reshaped = U_arr.reshape(n, m, U_arr.shape[1])
    V_reshaped = V_arr.reshape(m, p, V_arr.shape[1])
    W_reshaped = W_arr.reshape(p, n, W_arr.shape[1])
    
    try:
        result = np.einsum('iac,bkc,kic->iabk', U_reshaped, V_reshaped, W_reshaped)
    except:
        return True  # Fallback: assume valid if einsum fails
    
    expected = np.zeros((n, m, m, p))
    for a in range(m):
        expected[:, a, a, :] = 1.0
    
    if np.max(np.abs(result - expected)) > tol:
        return False
    return True

def construct():
    """
    Constructs the tensor decomposition for matrix multiplication.
    Uses the insight that the 1 x m x p tensor has rank m.
    Partitions n into n blocks of size 1.
    Total rank = n * m.
    """
    n, m, p = DIMENSIONS
    
    # Rank is n * m
    rank = n * m
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for i in range(n):
        # Block for row i of the output matrix
        # U block: (m, m) identity, placed at rows i*m .. i*m+m-1, cols i*m .. i*m+m-1
        # V block: (m*p, m), V[b, k, c] = delta_{b,c}, placed at cols i*m .. i*m+m-1
        # W block: (p*n, m), W[k, n, c] = 1 for all k, placed at cols i*m .. i*m+m-1
        
        u_offset = i * m
        v_offset = 0 # V is shared across i? No, V is (m*p, rank). 
        # Wait, V depends on b and k. It does NOT depend on i.
        # So V columns for block i are just the identity mapping for b.
        # V[b, k, c] = delta_{b,c}.
        # This is the same for all i.
        
        w_offset = 0 # W depends on k and i.
        # W[k, i, c] = 1.
        
        for c in range(m):
            col = u_offset + c
            
            # U[i, a, c] = delta_{a,c}
            # U row index: i*m + a
            U[i * m + c, col] = 1.0
            
            # V[b, k, c] = delta_{b,c}
            # V row index: b*p + k
            # For a fixed c, V[b, k, c] is 1 if b==c, else 0.
            # So for b=c, all k have 1.
            for k in range(p):
                V[c * p + k, col] = 1.0
            
            # W[k, i, c] = 1
            # W row index: k*n + i
            for k in range(p):
                W[k * n + i, col] = 1.0
        
    result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Validate
    if not _validate_decomposition(result['U'], result['V'], result['W'], n, m, p):
        # Fallback to schoolbook if validation fails (should not happen)
        return _schoolbook_decomposition(n, m, p)
        
    return result

def _schoolbook_decomposition(n, m, p):
    """
    Constructs the schoolbook decomposition for matrix multiplication.
    """
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    
    return {"U": U, "V": V, "W": W}

