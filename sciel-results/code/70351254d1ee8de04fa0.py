<<<<<<< SEARCH
# EVOLVE-BLOCK-START
import time

def _verify(U, V, W, n, m, p):
    """Check exact decomposition:
    sum_r W[k*n+i, r] * U[i*m+j1, r] * V[j2*p+k, r] == delta(j1, j2)
    for all i, k, j1, j2
    """
    rank = len(U[0]) if U else 0
    for i in range(n):
        for k in range(p):
            for j1 in range(m):
                for j2 in range(m):
                    s = 0
                    for r in range(rank):
                        s += W[k*n+i][r] * U[i*m+j1][r] * V[j2*p+k][r]
                    target = 1 if j1 == j2 else 0
                    if s != target:
                        return False
    return True

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_schoolbook = n * m * p
    
    # Construct schoolbook decomposition
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j][col] = 1
                V_school[j*p+k][col] = 1
                W_school[k*n+i][col] = 1
                col += 1
    
    # Verify schoolbook works
    if not _verify(U_school, V_school, W_school, n, m, p):
        raise ValueError("Schoolbook verification failed")
        
    return {"U": U_school, "V": V_school, "W": W_school}
# EVOLVE-BLOCK-END
=======
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import least_squares

def _verify(U, V, W, n, m, p):
    """Check exact decomposition:
    sum_r W[k*n+i, r] * U[i*m+j1, r] * V[j2*p+k, r] == delta(j1, j2)
    for all i, k, j1, j2
    """
    rank = len(U[0]) if U else 0
    for i in range(n):
        for k in range(p):
            for j1 in range(m):
                for j2 in range(m):
                    s = 0
                    for r in range(rank):
                        s += W[k*n+i][r] * U[i*m+j1][r] * V[j2*p+k][r]
                    target = 1 if j1 == j2 else 0
                    if abs(s - target) > 1e-9:
                        return False
    return True

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115
    
    # Create the target tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Schoolbook solution
    rank_sb = n * m * p
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best_U = U_sb.tolist()
    best_V = V_sb.tolist()
    best_W = W_sb.tolist()
    best_rank = rank_sb
    
    # Try to find lower rank decompositions
    for r in range(14, best_rank):
        if time.time() > deadline - 15:
            break
        
        def residual(params):
            U = params[:n*m*r].reshape(n*m, r)
            V = params[n*m*r:n*m*r + m*p*r].reshape(m*p, r)
            W = params[n*m*r + m*p*r:].reshape(p*n, r)
            
            T_approx = np.einsum('ar,br,cr->abc', U, V, W)
            return (T_approx - T).flatten()
        
        found = False
        for trial in range(5):
            if time.time() > deadline - 15:
                break
            
            np.random.seed(123 + trial)
            U_init = np.random.randn(n*m, r) * 0.3
            V_init = np.random.randn(m*p, r) * 0.3
            W_init = np.random.randn(p*n, r) * 0.3
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                result = least_squares(residual, x0, method='lm', max_nfev=200)
                if result.cost < 1e-6:
                    U = result.x[:n*m*r].reshape(n*m, r)
                    V = result.x[n*m*r:n*m*r + m*p*r].reshape(m*p, r)
                    W = result.x[n*m*r + m*p*r:].reshape(p*n, r)
                    
                    if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                        best_U = U.tolist()
                        best_V = V.tolist()
                        best_W = W.tolist()
                        best_rank = r
                        found = True
                        break
            except Exception:
                continue
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
