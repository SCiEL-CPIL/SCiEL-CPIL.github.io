"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Try to find rank-minimal decomposition
    for rank in range(30, 61):
        if time.time() > deadline:
            break
        result = _try_rank(n, m, p, rank, deadline)
        if result is not None:
            return result
    
    # Fallback to schoolbook
    return _schoolbook(n, m, p)

def _try_rank(n, m, p, rank, deadline):
    max_restarts = 20
    max_iter = 100
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        rng = np.random.RandomState(12345 + restart)
        U = rng.randn(n*m, rank)
        V = rng.randn(m*p, rank)
        W = rng.randn(p*n, rank)
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            # Solve for U
            for i in range(n):
                for j in range(m):
                    A_mat = np.array([V[j*p+k] * W[k*n+i] for k in range(p)])
                    try:
                        U[i*m+j] = np.linalg.lstsq(A_mat, np.ones(p), rcond=None)[0]
                    except:
                        pass
            
            # Solve for V
            for j in range(m):
                for k in range(p):
                    A_mat = np.array([U[i*m+j] * W[k*n+i] for i in range(n)])
                    try:
                        V[j*p+k] = np.linalg.lstsq(A_mat, np.ones(n), rcond=None)[0]
                    except:
                        pass
            
            # Solve for W
            for k in range(p):
                for i in range(n):
                    A_mat = np.array([U[i*m+j] * V[j*p+k] for j in range(m)])
                    try:
                        W[k*n+i] = np.linalg.lstsq(A_mat, np.ones(m), rcond=None)[0]
                    except:
                        pass
            
            if it % 20 == 19:
                max_err = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                            max_err = max(max_err, abs(val - 1))
                if max_err < 1e-7:
                    result = _rationalize(U, V, W, n, m, p)
                    if result:
                        return result
    
    return None

def _rationalize(U, V, W, n, m, p):
    U_r = np.round(U * 2) / 2
    V_r = np.round(V * 2) / 2
    W_r = np.round(W * 2) / 2
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_r[i*m+j], V_r[j*p+k] * W_r[k*n+i])
                if abs(val - 1) > 1e-9:
                    return None
    
    rank = U.shape[1]
    U_list = [[float(U_r[i,j]) for j in range(rank)] for i in range(U_r.shape[0])]
    V_list = [[float(V_r[i,j]) for j in range(rank)] for i in range(V_r.shape[0])]
    W_list = [[float(W_r[i,j]) for j in range(rank)] for i in range(W_r.shape[0])]
    
    return {"U": U_list, "V": V_list, "W": W_list}

def _schoolbook(n, m, p):
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
