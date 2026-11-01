import numpy as np
from scipy.optimize import least_squares
import time

DIMENSIONS = [3, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # safety margin before 150s kill

    # Target tensor: T[i', j', j'', k', i, k] = 1 if i'=i, k'=k, j'=j'' else 0
    # U: (n*m, R) -> index i*m + j
    # V: (m*p, R) -> index j*p + k
    # W: (p*n, R) -> index k*n + i  (task says W[k*n+i, r])
    
    targets = np.zeros((n, m, m, p, n, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                targets[i, j, j, k, i, k] = 1.0
    target_flat = targets.flatten()

    def residual(x, R):
        U = x[:n*m*R].reshape(n*m, R)
        V = x[n*m*R : n*m*R + m*p*R].reshape(m*p, R)
        W = x[n*m*R + m*p*R :].reshape(p*n, R)
        
        U_reshaped = U.reshape(n, m, R)
        V_reshaped = V.reshape(m, p, R)
        W_reshaped = W.reshape(p, n, R)
        
        res = np.einsum('ijr,jkr,ikr->ijjkik', U_reshaped, V_reshaped, W_reshaped)
        return (res.flatten() - target_flat)

    def verify(U, V, W):
        """Verify that the decomposition is exact."""
        U_reshaped = np.array(U).reshape(n, m, -1)
        V_reshaped = np.array(V).reshape(m, p, -1)
        W_reshaped = np.array(W).reshape(p, n, -1)
        R = U_reshaped.shape[2]
        res = np.einsum('ijr,jkr,ikr->ijjkik', U_reshaped, V_reshaped, W_reshaped)
        return np.max(np.abs(res.flatten() - target_flat)) < 1e-8

    def try_rank(R, max_restarts=20, max_nfev=500):
        start_time = time.time()
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
            
            # Random initialization with scaling
            x0 = np.random.randn(n*m*R + m*p*R + p*n*R) * 0.5
            
            try:
                res = least_squares(residual, x0, args=(R,), method='lm', max_nfev=max_nfev)
                if res.success and np.max(np.abs(res.fun)) < 1e-8:
                    U = res.x[:n*m*R].reshape(n*m, R)
                    V = res.x[n*m*R : n*m*R + m*p*R].reshape(m*p, R)
                    W = res.x[n*m*R + m*p*R :].reshape(p*n, R)
                    return U, V, W
            except Exception:
                continue
        return None

    # Try rank 20 first (theoretical minimum for 3x3 is 19, but 20 is more likely)
    for R in [20, 21, 22, 23]:
        if time.time() > deadline - 10:
            break
        result = try_rank(R)
        if result:
            U, V, W = result
            if verify(U, V, W):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback to schoolbook rank 27
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
