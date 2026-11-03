import numpy as np
import time

DIMENSIONS = [2, 3, 10]

def construct():
    start_time = time.time()
    deadline = 145.0  # Leave 5s safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Build target tensor
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    
    # Verify a decomposition
    def verify(U, V, W, r, tol=1e-10):
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m + j, :], V[j*p + k, :] * W[k*n + i, :])
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err
    
    # Schoolbook rank-60 decomposition
    def schoolbook_decomposition():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        return U, V, W, rank
    
    U_best, V_best, W_best, rank_best = schoolbook_decomposition()
    
    # Try to reduce rank using ALS
    if time.time() < deadline - 10:
        # Try ranks from rank_best - 1 down to some minimum
        for r in range(rank_best - 1, max(nm, mp, pn) - 1, -1):
            if time.time() > deadline - 5:
                break
            
            found = False
            max_restarts = 10
            
            for trial in range(max_restarts):
                if time.time() > deadline - 5:
                    break
                
                np.random.seed(trial * 1000 + r)
                
                # Initialize randomly
                U_c = np.random.randn(nm, r)
                V_c = np.random.randn(mp, r)
                W_c = np.random.randn(pn, r)
                
                # Alternating least squares
                for it in range(20):
                    if time.time() > deadline - 5:
                        break
                    
                    # Solve for U
                    M = np.einsum('jr,kr->jkr', V_c, W_c).reshape(mp * pn, r)
                    try:
                        U_c = T.reshape(nm, mp * pn) @ np.linalg.pinv(M.T)
                    except:
                        break
                    
                    # Solve for V
                    N = np.einsum('ir,kr->ikr', U_c, W_c).reshape(nm * pn, r)
                    try:
                        V_c = T.reshape(mp, nm * pn) @ np.linalg.pinv(N.T)
                    except:
                        break
                    
                    # Solve for W
                    P = np.einsum('ir,jr->ijr', U_c, V_c).reshape(nm * mp, r)
                    try:
                        W_c = T.reshape(pn, nm * mp) @ np.linalg.pinv(P.T)
                    except:
                        break
                
                # Check convergence
                T_check = np.einsum('ir,jr,kr->ijk', U_c, V_c, W_c)
                error = np.max(np.abs(T_check - T))
                
                if error < 1e-8:
                    # Try to round to half-integers
                    U_r = np.round(U_c * 2) / 2
                    V_r = np.round(V_c * 2) / 2
                    W_r = np.round(W_c * 2) / 2
                    
                    T_check2 = np.einsum('ir,jr,kr->ijk', U_r, V_r, W_r)
                    if np.max(np.abs(T_check2 - T)) < 1e-10:
                        U_best, V_best, W_best, rank_best = U_r, V_r, W_r, r
                        found = True
                        break
                    else:
                        # Use unrounded if close enough
                        if error < 1e-10:
                            U_best, V_best, W_best, rank_best = U_c, V_c, W_c, r
                            found = True
                            break
            
            if found:
                break
    
    # Final verification
    final_err = verify(U_best, V_best, W_best, rank_best)
    if final_err > 1e-8:
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        U_best, V_best, W_best = U, V, W
    
    return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}
