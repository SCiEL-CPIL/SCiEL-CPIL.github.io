import numpy as np
import time

DIMENSIONS = [3, 3, 6]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that the decomposition is correct."""
    rank = len(U[0]) if U else 0
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Check shapes
    if U.shape != (n*m, rank):
        return False
    if V.shape != (m*p, rank):
        return False
    if W.shape != (p*n, rank):
        return False
    
    # Check the decomposition
    # For each (i,j,k), we need sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 1
    # For invalid triples, the sum should be 0.
    
    # First, check the non-zero entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U[i*m + j]
                v_val = V[j*p + k]
                w_val = W[k*n + i]
                product_sum = np.dot(u_val, v_val * w_val)
                if abs(product_sum - 1) > tol:
                    return False
    
    # Check some invalid entries (sample)
    # An invalid entry is (alpha, beta, gamma) where alpha=(i,j), beta=(j',k'), gamma=(i'',k'')
    # and not (i=i'', j=j', k'=k'')
    for i in range(n):
        for j in range(m):
            for j2 in range(m):
                if j2 == j:
                    continue
                for k in range(p):
                    u_val = U[i*m + j]
                    v_val = V[j2*p + k]
                    w_val = W[k*n + i]
                    product_sum = np.dot(u_val, v_val * w_val)
                    if abs(product_sum) > tol:
                        return False
    
    return True

def _schoolbook(n, m, p):
    """Generate the schoolbook decomposition."""
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

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s deadline for safety
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    best_result = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try to find a better decomposition
    # This is a placeholder for a more sophisticated search
    # For now, just return the schoolbook decomposition
    
    return best_result
