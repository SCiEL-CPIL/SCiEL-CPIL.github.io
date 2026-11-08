import time
import numpy as np

_DEADLINE = 140.0
_START = time.time()

def _elapsed():
    return time.time() - _START

def _deadline_exceeded():
    return _elapsed() > _DEADLINE

def _schoolbook(n, m, p):
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

def _build_strassen_42():
    # Same as before
    ...

def _verify_bilinear(U, V, W, n=2, m=4, p=6, atol=1e-8):
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    val = np.dot(U_arr[i*m+j, :], V_arr[jp*p+k, :] * W_arr[k*n+i, :])
                    target = 1.0 if j == jp else 0.0
                    if abs(val - target) > atol:
                        return False
    return True

def _numerical_search(n=2, m=4, p=6, start_rank=24, end_rank=41, max_attempts=3):
    best = None
    best_rank = end_rank + 1
    
    for r in range(start_rank, end_rank + 1):
        if _deadline_exceeded():
            break
        
        for attempt in range(max_attempts):
            if _deadline_exceeded():
                break
            
            rng = np.random.default_rng(12345 + r * 7 + attempt)
            U = rng.standard_normal((n*m, r))
            V = rng.standard_normal((m*p, r))
            W = rng.standard_normal((p*n, r))
            
            # Gradient descent
            lr = 0.01
            for step in range(500):
                if _deadline_exceeded():
                    break
                
                # Compute residual and gradient
                res = np.zeros((n, m, m, p))
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                
                for i in range(n):
                    for j in range(m):
                        for jp in range(m):
                            for k in range(p):
                                target = 1.0 if j == jp else 0.0
                                val = np.dot(U[i*m+j, :], V[jp*p+k, :] * W[k*n+i, :])
                                res[i,j,jp,k] = val - target
                                # Gradient
                                g = res[i,j,jp,k]
                                grad_U[i*m+j, :] += g * (V[jp*p+k, :] * W[k*n+i, :])
                                grad_V[jp*p+k, :] += g * (U[i*m+j, :] * W[k*n+i, :])
                                grad_W[k*n+i, :] += g * (U[i*m+j, :] * V[jp*p+k, :])
                
                cost = np.sum(res**2)
                if cost < 1e-12:
                    break
                
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
            
            if _deadline_exceeded():
                break
            
            if _verify_bilinear(U.tolist(), V.tolist(), W.tolist()):
                best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                best_rank = r
                break
        
        if best is not None:
            break
    
    return best, best_rank
