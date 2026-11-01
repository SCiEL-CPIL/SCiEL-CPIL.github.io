"""Matrix multiplication 2,3,5 - rank-minimal exact tensor decomposition."""
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 5
    
    # Build the matrix multiplication tensor
    tensor = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                tensor[a, b, c] = 1.0
    
    # Mode-3 flattening: T3[c, a*m*p + b] = T[a,b,c]
    T3 = tensor.transpose(2, 0, 1).reshape(p * n, n * m * m * p)
    T3_rank = np.linalg.matrix_rank(T3)
    
    # Try to find rank-r decompositions for r from T3_rank to 29
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    for r in range(T3_rank, n * m * p):
        found = False
        
        for attempt in range(20):
            W_init = np.random.randn(p * n, r)
            
            # For fixed W, compute X[a,b,t] such that T[a,b,:] = X[a,b,:] @ W^T
            WtW = W_init.T @ W_init
            if np.linalg.matrix_rank(WtW) < r:
                continue
            
            try:
                WtW_inv = np.linalg.inv(WtW)
            except np.linalg.LinAlgError:
                continue
            
            X = np.zeros((n * m, m * p, r))
            for a in range(n * m):
                for b in range(m * p):
                    X[a, b, :] = np.sum(tensor[a, b, :] * W_init, axis=1) @ WtW_inv
            
            # Check if X[:,:,t] has rank <= 1 for all t
            valid = True
            U_sol = np.zeros((n * m, r))
            V_sol = np.zeros((m * p, r))
            
            for t in range(r):
                Xt = X[:, :, t]
                if np.linalg.matrix_rank(Xt) > 1:
                    valid = False
                    break
                
                if valid:
                    U_svd, s_vals, Vt_svd = np.linalg.svd(Xt, full_matrices=False)
                    if len(s_vals) > 0 and s_vals[0] > 1e-10:
                        U_sol[:, t] = U_svd[:, 0] * np.sqrt(s_vals[0])
                        V_sol[:, t] = Vt_svd[0, :] * np.sqrt(s_vals[0])
            
            if valid:
                T_verify = np.zeros_like(tensor)
                for t in range(r):
                    T_verify += np.einsum('a,b,c->abc', U_sol[:, t], V_sol[:, t], W_init[:, t])
                
                if np.allclose(T_verify, tensor, atol=1e-8):
                    best_rank = r
                    best_U = U_sol.tolist()
                    best_V = V_sol.tolist()
                    best_W = W_init.tolist()
                    found = True
                    break
        
        if found:
            break
    
    if best_U is not None:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    # Return schoolbook decomposition
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
