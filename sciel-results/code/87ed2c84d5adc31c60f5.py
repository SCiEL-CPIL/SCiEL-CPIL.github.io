"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=7
    
    deadline = time.time() + 140
    
    def schoolbook():
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
    
    def verify_float(U, V, W, tol=1e-9):
        n, m, p = DIMENSIONS
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U_np[i*m+j] * V_np[j*p+k] * W_np[k*n+i])
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    def try_rank(rank, n_trials=10):
        n, m, p = DIMENSIONS
        size_U = n * m
        size_V = m * p
        size_W = p * n
        
        for trial in range(n_trials):
            if time.time() > deadline - 5:
                break
            
            U = np.random.randn(size_U, rank) * 0.3
            V = np.random.randn(size_V, rank) * 0.3
            W = np.random.randn(size_W, rank) * 0.3
            
            for it in range(100):
                if time.time() > deadline - 5:
                    break
                
                for i in range(n):
                    for j in range(m):
                        A = np.array([V[j*p+k] * W[k*n+i] for k in range(p)])
                        try:
                            U[i*m+j] = np.linalg.lstsq(A, np.ones(p), rcond=None)[0]
                        except:
                            pass
                
                for j in range(m):
                    for k in range(p):
                        A = np.array([U[i*m+j] * W[k*n+i] for i in range(n)])
                        try:
                            V[j*p+k] = np.linalg.lstsq(A, np.ones(n), rcond=None)[0]
                        except:
                            pass
                
                for i in range(n):
                    for k in range(p):
                        A = np.array([U[i*m+j] * V[j*p+k] for j in range(m)])
                        try:
                            W[k*n+i] = np.linalg.lstsq(A, np.ones(m), rcond=None)[0]
                        except:
                            pass
            
            if verify_float(U, V, W):
                U_list = [[float(x) for x in row] for row in U]
                V_list = [[float(x) for x in row] for row in V]
                W_list = [[float(x) for x in row] for row in W]
                
                def to_half(x):
                    return round(x * 2) / 2
                
                U_hi = [[to_half(x) for x in row] for row in U_list]
                V_hi = [[to_half(x) for x in row] for row in V_list]
                W_hi = [[to_half(x) for x in row] for row in W_list]
                
                if verify_float(U_hi, V_hi, W_hi, tol=1e-7):
                    return U_hi, V_hi, W_hi
                return U_list, V_list, W_list
        
        return None
    
    for rank in range(21, 43):
        if time.time() > deadline - 10:
            break
        result = try_rank(rank, n_trials=5)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    return schoolbook()
# EVOLVE-BLOCK-END

