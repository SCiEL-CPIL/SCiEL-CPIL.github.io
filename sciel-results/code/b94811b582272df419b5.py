"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 3, 3]

def _verify(U, V, W, n, m, p):
    """Verify exact decomposition with integer/half-integer entries."""
    # Check entries are integer or half-integer
    for mat in [U, V, W]:
        for row in mat:
            for val in row:
                if abs(val * 2 - round(val * 2)) > 1e-9:
                    return False
    # Check tensor equation
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _snap_to_half(x):
    return np.round(x * 2) / 2

def _try_rank(r, n, m, p, num_trials=50, time_limit=10):
    """Try to find a rank-r decomposition via random init + optimization."""
    start = time.time()
    U_shape = (n*m, r)
    V_shape = (m*p, r)
    W_shape = (p*n, r)
    
    for trial in range(num_trials):
        if time.time() - start > time_limit:
            break
        
        # Random initialization with small values
        np.random.seed(trial * 12345 + 678)
        U = np.random.randn(*U_shape) * 0.5
        V = np.random.randn(*V_shape) * 0.5
        W = np.random.randn(*W_shape) * 0.5
        
        # Gradient descent to minimize error
        lr = 0.01
        for step in range(500):
            if time.time() - start > time_limit:
                break
            
            # Compute error and gradients
            error = 0.0
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for r in range(r):
                            s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        err = s - 1.0
                        error += err * err
                        for r in range(r):
                            grad_U[i*m+j, r] += 2 * err * V[j*p+k, r] * W[k*n+i, r]
                            grad_V[j*p+k, r] += 2 * err * U[i*m+j, r] * W[k*n+i, r]
                            grad_W[k*n+i, r] += 2 * err * U[i*m+j, r] * V[j*p+k, r]
            
            if error < 1e-12:
                break
            
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
        
        # Snap to half-integers
        U_snapped = _snap_to_half(U)
        V_snapped = _snap_to_half(V)
        W_snapped = _snap_to_half(W)
        
        # Verify
        if _verify(U_snapped.tolist(), V_snapped.tolist(), W_snapped.tolist(), n, m, p):
            return U_snapped.tolist(), V_snapped.tolist(), W_snapped.tolist()
    
    return None

def construct():
    n, m, p = DIMENSIONS
    global_start = time.time()
    
    # Try ranks from 9 upward
    for r in range(9, 19):
        if time.time() - global_start > 130:
            break
        result = _try_rank(r, n, m, p, num_trials=30, time_limit=10)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
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
# EVOLVE-BLOCK-END
