import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    
    def verify(U, V, W, n, m, p, tol=1e-6):
        r = len(U[0])
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        
        # Diagonal check
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U_arr[i*m+j], V_arr[j*p+k]) @ W_arr[k*n+i]
                    if abs(s - 1.0) > tol:
                        return False
        # Off-diagonal check
        for i in range(n):
            for j in range(m):
                for j2 in range(m):
                    if j == j2: continue
                    for k in range(p):
                        s = np.dot(U_arr[i*m+j], V_arr[j2*p+k]) @ W_arr[k*n+i]
                        if abs(s) > tol:
                            return False
        return True

    def schoolbook(n, m, p):
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

    # Try to find rank 25 solution
    rank = 25
    # Initial guess: random half-integers
    np.random.seed(42)
    def random_init(shape, scale=1.0):
        return np.round(np.random.randn(*shape) * scale * 2) / 2
    
    U_init = random_init((n*m, rank))
    V_init = random_init((m*p, rank))
    W_init = random_init((p*n, rank))
    
    # Flatten
    x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
    
    def error_func(x):
        U = x[:n*m*rank].reshape(n*m, rank)
        V = x[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
        W = x[n*m*rank + m*p*rank:].reshape(p*n, rank)
        
        err = 0.0
        # Diagonal
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i*m+j], V[j*p+k]) @ W[k*n+i]
                    err += (s - 1.0)**2
        # Off-diagonal
        for i in range(n):
            for j in range(m):
                for j2 in range(m):
                    if j == j2: continue
                    for k in range(p):
                        s = np.dot(U[i*m+j], V[j2*p+k]) @ W[k*n+i]
                        err += s**2
        return err

    # Use L-BFGS-B
    try:
        res = minimize(error_func, x0, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-12})
        if res.success and res.fun < 1e-8:
            U_sol = res.x[:n*m*rank].reshape(n*m, rank)
            V_sol = res.x[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
            W_sol = res.x[n*m*rank + m*p*rank:].reshape(p*n, rank)
            
            # Round to nearest half-integer?
            # The problem requires integer/half-integer entries.
            # If the solution is close to half-integers, we can round.
            U_round = np.round(U_sol * 2) / 2
            V_round = np.round(V_sol * 2) / 2
            W_round = np.round(W_sol * 2) / 2
            
            # Verify rounded solution
            if verify(U_round.tolist(), V_round.tolist(), W_round.tolist(), n, m, p):
                return {"U": U_round.tolist(), "V": V_round.tolist(), "W": W_round.tolist()}
            
            # If rounding failed, maybe the solution is exact integers?
            U_int = np.round(U_sol)
            V_int = np.round(V_sol)
            W_int = np.round(W_sol)
            if verify(U_int.tolist(), V_int.tolist(), W_int.tolist(), n, m, p):
                return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
                
    except Exception as e:
        pass

    # Fallback
    U, V, W = schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
