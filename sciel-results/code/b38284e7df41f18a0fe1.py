import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 5]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_arr[i * m + j] * V_arr[j * p + k] * W_arr[k * n + i])
                if abs(val - 1.0) > tol:
                    return False
    return True

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
    return U, V, W

def _find_rank_decomposition(n, m, p, rank, seed, deadline):
    np.random.seed(seed)
    
    # Flatten variables
    # U: n*m x rank
    # V: m*p x rank
    # W: p*n x rank
    nU = n * m * rank
    nV = m * p * rank
    nW = p * n * rank
    total_vars = nU + nV + nW
    
    # Initial guess
    x0 = np.random.randn(total_vars) * 0.5
    x0 = x0 / np.linalg.norm(x0) * 1.0 # Scale
    
    def residuals(x):
        U = x[:nU].reshape(n*m, rank)
        V = x[nU:nU+nV].reshape(m*p, rank)
        W = x[nU+nV:].reshape(p*n, rank)
        
        res = np.zeros(n*m*p)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    res[idx] = val - 1.0
                    idx += 1
        return res
    
    try:
        result = least_squares(residuals, x0, method='lm', max_nfev=10000, xtol=1e-12, ftol=1e-12)
        if result.success and np.max(np.abs(result.fun)) < 1e-9:
            U = result.x[:nU].reshape(n*m, rank)
            V = result.x[nU:nU+nV].reshape(m*p, rank)
            W = result.x[nU+nV:].reshape(p*n, rank)
            
            # Round to half-integers
            U_round = np.round(U * 2) / 2
            V_round = np.round(V * 2) / 2
            W_round = np.round(W * 2) / 2
            
            U_list = U_round.tolist()
            V_list = V_round.tolist()
            W_list = W_round.tolist()
            
            if _verify_decomposition(U_list, V_list, W_list, n, m, p, tol=1e-7):
                return U_list, V_list, W_list
    except Exception:
        pass
    
    return None, None, None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110
    
    # Start with schoolbook
    U_school, V_school, W_school = _schoolbook(n, m, p)
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    
    # Try rank 28
    for rank in [28]:
        for seed in range(20):
            if time.time() > deadline:
                break
            U, V, W = _find_rank_decomposition(n, m, p, rank, seed, deadline)
            if U is not None:
                best_result = {"U": U, "V": V, "W": W}
                break
        if best_result["U"] != U_school:
            break
            
    return best_result
