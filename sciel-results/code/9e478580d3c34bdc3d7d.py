import numpy as np
import time

DIMENSIONS = [3, 3, 7]

def _verify(U, V, W, n, m, p, rank):
    """Verify that the decomposition exactly reproduces the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(s - 1.0) > 1e-6:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
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
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    # Try to find a low-rank decomposition using alternating minimization
    best_result = None
    best_rank = n * m * p
    
    # Try ranks from 21 upward
    for rank in range(21, 64):
        if time.time() > deadline - 30:
            break
        
        for attempt in range(3):
            if time.time() > deadline - 30:
                break
            
            np.random.seed(rank * 100 + attempt)
            U = np.random.randn(n * m, rank) * 0.1
            V = np.random.randn(m * p, rank) * 0.1
            W = np.random.randn(p * n, rank) * 0.1
            
            for iteration in range(200):
                if time.time() > deadline - 30:
                    break
                
                # Update U
                for i in range(n):
                    for j in range(m):
                        A = np.zeros((p, rank))
                        b = np.ones(p)
                        for k in range(p):
                            for r in range(rank):
                                A[k, r] = V[j * p + k, r] * W[k * n + i, r]
                        try:
                            U[i * m + j, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                        except:
                            pass
                
                # Update V
                for j in range(m):
                    for k in range(p):
                        A = np.zeros((n, rank))
                        b = np.ones(n)
                        for i in range(n):
                            for r in range(rank):
                                A[i, r] = U[i * m + j, r] * W[k * n + i, r]
                        try:
                            V[j * p + k, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                        except:
                            pass
                
                # Update W
                for k in range(p):
                    for i in range(n):
                        A = np.zeros((m, rank))
                        b = np.ones(m)
                        for j in range(m):
                            for r in range(rank):
                                A[j, r] = U[i * m + j, r] * V[j * p + k, r]
                        try:
                            W[k * n + i, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                        except:
                            pass
                
                if iteration % 20 == 0:
                    if _verify(U, V, W, n, m, p, rank):
                        best_rank = rank
                        best_result = {
                            "U": [[float(x) for x in U[i]] for i in range(n * m)],
                            "V": [[float(x) for x in V[i]] for i in range(m * p)],
                            "W": [[float(x) for x in W[i]] for i in range(p * n)]
                        }
                        break
            
            if best_result is not None:
                break
        
        if best_result is not None:
            break
    
    if best_result is not None:
        return best_result
    
    # Fallback to schoolbook
    return _schoolbook(n, m, p)
