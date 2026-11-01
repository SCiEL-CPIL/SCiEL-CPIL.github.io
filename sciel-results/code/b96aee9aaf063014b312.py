# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product
from scipy.optimize import minimize

def _verify(U, V, W, n=2, m=2, p=4):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    for i in range(n):
        for k in range(m):
            for j in range(p):
                a = i * m + k
                b = k * p + j
                c = j * n + i
                val = sum(U[a][r] * V[b][r] * W[c][r] for r in range(rank))
                if abs(val - 1) > 1e-9:
                    return False
    return True

def _verify_full(U, V, W, n=2, m=2, p=4):
    """Full verification including zero entries."""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Build expected tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for k in range(m):
            for j in range(p):
                a = i * m + k
                b = k * p + j
                c = j * n + i
                T[a, b, c] = 1.0
    
    # Compute decomposition
    T_comp = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        T_comp += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
    
    if np.max(np.abs(T_comp - T)) > 1e-9:
        return False
    return True

def _construct_rank14():
    """Return the known rank-14 Strassen-based solution."""
    n, m, p = 2, 2, 4
    rank = 14
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    U[0][0] = 1; U[3][0] = 1
    U[2][1] = 1; U[3][1] = 1
    U[0][2] = 1
    U[3][3] = 1
    U[0][4] = 1; U[1][4] = 1
    U[2][5] = 1; U[0][5] = -1
    U[1][6] = 1; U[3][6] = -1
    
    U[0][7] = 1; U[3][7] = 1
    U[2][8] = 1; U[3][8] = 1
    U[0][9] = 1
    U[3][10] = 1
    U[0][11] = 1; U[1][11] = 1
    U[2][12] = 1; U[0][12] = -1
    U[1][13] = 1; U[3][13] = -1
    
    V[0][0] = 1; V[5][0] = 1
    V[0][1] = 1
    V[1][2] = 1; V[5][2] = -1
    V[4][3] = 1; V[0][3] = -1
    V[5][4] = 1
    V[0][5] = 1; V[1][5] = 1
    V[4][6] = 1; V[5][6] = 1
    
    V[2][7] = 1; V[7][7] = 1
    V[2][8] = 1
    V[3][9] = 1; V[7][9] = -1
    V[6][10] = 1; V[2][10] = -1
    V[7][11] = 1
    V[2][12] = 1; V[3][12] = 1
    V[6][13] = 1; V[7][13] = 1
    
    W[0][0] = 1; W[0][3] = 1; W[0][4] = -1; W[0][6] = 1
    W[1][1] = 1; W[1][3] = 1
    W[2][2] = 1; W[2][4] = 1
    W[3][0] = 1; W[3][1] = -1; W[3][2] = 1; W[3][5] = 1
    
    W[4][7] = 1; W[4][10] = 1; W[4][11] = -1; W[4][13] = 1
    W[5][8] = 1; W[5][10] = 1
    W[6][9] = 1; W[6][11] = 1
    W[7][7] = 1; W[7][8] = -1; W[7][9] = 1; W[7][12] = 1
    
    return {"U": U, "V": V, "W": W}

def _search_rank12(deadline):
    """Search for a rank-12 decomposition using numerical optimization."""
    n, m, p = 2, 2, 4
    rank = 12
    
    # Initialize with random values
    np.random.seed(42)
    U = np.random.randn(n * m, rank)
    V = np.random.randn(m * p, rank)
    W = np.random.randn(p * n, rank)
    
    # Flatten
    x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    def objective(x):
        U = x[:n*m*rank].reshape(n*m, rank)
        V = x[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
        W = x[n*m*rank + m*p*rank:].reshape(p*n, rank)
        
        T_comp = np.zeros((n*m, m*p, p*n))
        for r in range(rank):
            T_comp += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        
        T_target = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    a = i * m + k
                    b = k * p + j
                    c = j * n + i
                    T_target[a, b, c] = 1.0
        
        return np.sum((T_comp - T_target)**2)
    
    # Use Nelder-Mead for small problems
    # Time limit check
    if time.time() > deadline - 10:
        return None
        
    try:
        res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6})
        
        if res.fun < 1e-6:
            U = res.x[:n*m*rank].reshape(n*m, rank)
            V = res.x[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
            W = res.x[n*m*rank + m*p*rank:].reshape(p*n, rank)
            
            # Round to nearest half-integer or integer
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            
            if _verify_full(U_r, V_r, W_r):
                return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                
    except Exception:
        pass
        
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Try to find rank-12 solution
    result = _search_rank12(deadline)
    if result is not None:
        return result
    
    # Fall back to rank 14
    return _construct_rank14()
# EVOLVE-BLOCK-END
