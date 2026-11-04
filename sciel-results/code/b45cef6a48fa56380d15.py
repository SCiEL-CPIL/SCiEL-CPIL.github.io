"""Matrix multiplication 3x3x3: find a rank-minimal exact tensor decomposition."""
import time
import random
import numpy as np

DIMENSIONS = [3, 3, 3]

def construct():
    """
    Construct U, V, W matrices for the tensor decomposition of 3x3x3 matrix multiplication.
    Returns dict with 'U', 'V', 'W' keys.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    best_rank = n * m * p  # schoolbook fallback
    best_U = None
    best_V = None
    best_W = None
    
    # Try ranks from 20 up to 27
    for rank in range(20, best_rank + 1):
        if time.time() > deadline - 15:
            break
        
        found = _find_decomposition(rank, n, m, p, deadline)
        if found is not None:
            best_U, best_V, best_W = found
            best_rank = rank
            break
    
    if best_U is not None:
        return {"U": best_U, "V": best_V, "W": best_W}
    else:
        # Fallback to schoolbook
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

def _find_decomposition(rank, n, m, p, deadline):
    """
    Find a rank-decomposition using multiple random restarts with gradient descent.
    """
    for trial in range(30):
        if time.time() > deadline - 10:
            break
        
        # Random initialization
        U = np.random.randn(n * m, rank) * 0.3
        V = np.random.randn(m * p, rank) * 0.3
        W = np.random.randn(p * n, rank) * 0.3
        
        # Gradient descent with adaptive learning rate
        lr = 0.05
        for step in range(300):
            if time.time() > deadline - 10:
                break
            
            # Compute error
            err = _max_error(U, V, W, n, m, p)
            
            if err < 1e-8:
                return (U.tolist(), V.tolist(), W.tolist())
            
            # Compute gradients
            gU, gV, gW = _gradients(U, V, W, n, m, p)
            
            # Update
            U -= lr * gU
            V -= lr * gV
            W -= lr * gW
            
            # Adaptive learning rate
            new_err = _max_error(U, V, W, n, m, p)
            if new_err > err * 1.5:
                lr = max(0.001, lr * 0.5)
                U += lr * gU
                V += lr * gV
                W += lr * gW
        
        # Final check
        if _max_error(U, V, W, n, m, p) < 1e-8:
            return (U.tolist(), V.tolist(), W.tolist())
    
    return None

def _max_error(U, V, W, n, m, p):
    """Compute maximum absolute error."""
    max_err = 0.0
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                # Wait, this is wrong. Need element-wise product then sum.
                val = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                max_err = max(max_err, abs(val - 1.0))
    return max_err

def _gradients(U, V, W, n, m, p):
    """Compute gradients of the max error."""
    rank = U.shape[1]
    gU = np.zeros_like(U)
    gV = np.zeros_like(V)
    gW = np.zeros_like(W)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Compute value
                val = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                err = val - 1.0
                
                # Gradients
                gU[i*m+j] += err * V[j*p+k] * W[k*n+i]
                gV[j*p+k] += err * U[i*m+j] * W[k*n+i]
                gW[k*n+i] += err * U[i*m+j] * V[j*p+k]
    
    return gU, gV, gW
