# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import least_squares

def _verify(U, V, W, n, m, p):
    """Verify that the factors correctly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _find_decomposition(n, m, p, R, seed=0):
    np.random.seed(seed)
    # Initialize with random half-integers
    # Using a small set of values to bias towards simple solutions
    values = np.array([0.0, 0.5, 1.0, -0.5, -1.0, 1.5, -1.5])
    U_init = np.random.choice(values, size=(n*m, R))
    V_init = np.random.choice(values, size=(m*p, R))
    W_init = np.random.choice(values, size=(p*n, R))
    
    x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
    
    def residuals(x):
        U = x[:n*m*R].reshape(n*m, R)
        V = x[n*m*R:n*m*R + m*p*R].reshape(m*p, R)
        W = x[n*m*R + m*p*R:].reshape(p*n, R)
        
        res = np.zeros(n*m*p)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    res[idx] = val - 1.0
                    idx += 1
        return res

    try:
        # Use trf for under-determined systems
        sol = least_squares(residuals, x0, method='trf', max_nfev=5000, ftol=1e-12, xtol=1e-12, gtol=1e-12)
    except Exception:
        return None

    if sol.cost < 1e-10:
        x_sol = sol.x
        U_sol = x_sol[:n*m*R].reshape(n*m, R)
        V_sol = x_sol[n*m*R:n*m*R + m*p*R].reshape(m*p, R)
        W_sol = x_sol[n*m*R + m*p*R:].reshape(p*n, R)
        
        # Check if close to half-integers
        U_round = np.round(U_sol * 2) / 2
        V_round = np.round(V_sol * 2) / 2
        W_round = np.round(W_sol * 2) / 2
        
        # Verify rounded solution
        if _verify(U_round.tolist(), V_round.tolist(), W_round.tolist(), n, m, p):
            return U_round.tolist(), V_round.tolist(), W_round.tolist()
            
    return None

def construct():
    n, m, p = DIMENSIONS
    lower_bound = max(n*m, m*p, p*n)
    schoolbook_rank = n * m * p
    
    # Candidate ranks
    ranks = list(range(lower_bound, schoolbook_rank + 1, 5))
    if schoolbook_rank not in ranks:
        ranks.append(schoolbook_rank)
        
    # Ensure we try the schoolbook rank last
    # But we want the lowest rank, so we iterate from low to high
    
    for R in ranks:
        # Try a few seeds for each rank
        for seed in range(10):
            res = _find_decomposition(n, m, p, R, seed)
            if res:
                U, V, W = res
                return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    rank = schoolbook_rank
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
