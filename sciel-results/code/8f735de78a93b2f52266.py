"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    lower_bound = max(n * m, m * p, p * n)
    
    start_time = time.time()
    deadline = start_time + 135  # 135s deadline, 15s safety margin
    
    for rank in range(lower_bound, 50):
        if time.time() > deadline:
            break
        result = _find_decomposition(n, m, p, rank, deadline)
        if result is not None:
            return result
    
    # Fallback: schoolbook
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
    return {"U": U, "V": V, "W": W}

def _find_decomposition(n, m, p, rank, deadline, max_trials=200):
    """Try to find a rank-r decomposition."""
    for trial in range(max_trials):
        if time.time() > deadline:
            return None
        np.random.seed(1000 + trial)
        U = np.random.randn(n * m, rank)
        V = np.random.randn(m * p, rank)
        W = np.random.randn(p * n, rank)
        
        for it in range(1000):
            if time.time() > deadline:
                return None
            
            # Update W
            new_W = np.zeros((p * n, rank))
            for k in range(p):
                for i in range(n):
                    A = np.zeros((rank, m * m))
                    for c in range(rank):
                        for j in range(m):
                            for j2 in range(m):
                                A[c, j * m + j2] = U[i * m + j, c] * V[j2 * p + k, c]
                    target = np.zeros(m * m)
                    for j in range(m):
                        target[j * m + j] = 1.0
                    try:
                        new_W[k * n + i] = np.linalg.lstsq(A.T, target, rcond=None)[0]
                    except:
                        new_W[k * n + i] = np.random.randn(rank)
            W = new_W
            
            # Update U
            new_U = np.zeros((n * m, rank))
            for i in range(n):
                for j in range(m):
                    A = np.zeros((rank, p * m))
                    for c in range(rank):
                        for k in range(p):
                            for j2 in range(m):
                                A[c, k * m + j2] = W[k * n + i, c] * V[j2 * p + k, c]
                    target = np.zeros(p * m)
                    for k in range(p):
                        target[k * m + j] = 1.0
                    try:
                        new_U[i * m + j] = np.linalg.lstsq(A.T, target, rcond=None)[0]
                    except:
                        new_U[i * m + j] = np.random.randn(rank)
            U = new_U
            
            # Update V
            new_V = np.zeros((m * p, rank))
            for j in range(m):
                for k in range(p):
                    A = np.zeros((rank, n * m))
                    for c in range(rank):
                        for i in range(n):
                            for j2 in range(m):
                                A[c, i * m + j2] = U[i * m + j2, c] * W[k * n + i, c]
                    target = np.zeros(n * m)
                    for i in range(n):
                        target[i * m + j] = 1.0
                    try:
                        new_V[j * p + k] = np.linalg.lstsq(A.T, target, rcond=None)[0]
                    except:
                        new_V[j * p + k] = np.random.randn(rank)
            V = new_V
            
            if _verify(n, m, p, U, V, W, tol=1e-8):
                return _to_lists(U, V, W)
    
    return None

def _verify(n, m, p, U, V, W, tol=1e-8):
    r = U.shape[1]
    for i in range(n):
        for k in range(p):
            for j in range(m):
                for j2 in range(m):
                    val = sum(U[i*m+j, c] * V[j2*p+k, c] * W[k*n+i, c] for c in range(r))
                    exp = 1.0 if j == j2 else 0.0
                    if abs(val - exp) > tol:
                        return False
    return True

def _to_lists(U, V, W):
    return {
        "U": [[float(x) for x in row] for row in U],
        "V": [[float(x) for x in row] for row in V],
        "W": [[float(x) for x in row] for row in W]
    }
# EVOLVE-BLOCK-END

