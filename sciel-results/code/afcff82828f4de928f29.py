import time
import numpy as np
from scipy.optimize import least_squares

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0

    def schoolbook():
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

    def verify(U, V, W, tol=1e-9):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(rank))
                    if abs(val - 1.0) > tol:
                        return False
        return True

    def solve_rank(rank):
        # Flatten U, V, W
        # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        size_u = n * m * rank
        size_v = m * p * rank
        size_w = p * n * rank
        total_size = size_u + size_v + size_w
        
        def residuals(x):
            U = x[:size_u].reshape((n*m, rank))
            V = x[size_u:size_u+size_v].reshape((m*p, rank))
            W = x[size_u+size_v:].reshape((p*n, rank))
            
            res = np.zeros(n*m*p)
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                        res[idx] = val - 1.0
                        idx += 1
            return res

        # Initial guess: random small integers
        np.random.seed(42)
        x0 = np.random.randint(-2, 3, total_size)
        
        try:
            result = least_squares(residuals, x0, method='lm')
            x_opt = result.x
            U = x_opt[:size_u].reshape((n*m, rank))
            V = x_opt[size_u:size_u+size_v].reshape((m*p, rank))
            W = x_opt[size_u+size_v:].reshape((p*n, rank))
            
            # Snap to half-integers
            U_snap = np.round(U * 2) / 2
            V_snap = np.round(V * 2) / 2
            W_snap = np.round(W * 2) / 2
            
            if verify(U_snap, V_snap, W_snap):
                return {
                    "U": U_snap.tolist(),
                    "V": V_snap.tolist(),
                    "W": W_snap.tolist()
                }
        except Exception:
            pass
        return None

    # Try ranks from 26 to 35
    for rank in range(26, 36):
        if time.time() > deadline - 10:
            break
        res = solve_rank(rank)
        if res:
            return res

    return schoolbook()
