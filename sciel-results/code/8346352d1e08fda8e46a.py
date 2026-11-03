import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def construct():
    t_start = time.time()
    deadline = t_start + 125  # 25s safety margin
    
    n, m, p = DIMENSIONS  # 4, 5, 5
    nm, mp, pn = n*m, m*p, p*n  # 20, 25, 20
    
    def verify(U, V, W):
        """Verify tensor decomposition: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1"""
        Ua, Va, Wa = np.array(U, dtype=float), np.array(V, dtype=float), np.array(W, dtype=float)
        R = Ua.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(Ua[i*m+j, :], Va[j*p+k, :] * Wa[k*n+i, :])
                    if abs(val - 1.0) > 1e-6:
                        return False
        return True
    
    def compute_error(U, V, W):
        Ua, Va, Wa = np.array(U, dtype=float), np.array(V, dtype=float), np.array(W, dtype=float)
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(Ua[i*m+j, :], Va[j*p+k, :] * Wa[k*n+i, :])
                    err += (val - 1.0)**2
        return err
    
    def als_solve(U, V, W, R, max_iter=50):
        """Alternating least squares for tensor decomposition."""
        best_err = float('inf')
        best = (U.copy(), V.copy(), W.copy())
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            # Solve for U: each row i*m+j needs dot(U[row], Va[j*p+k,:]*Wa[k*n+i,:]) = 1 summed over k
            for i in range(n):
                for j in range(m):
                    # For fixed (i,j): need sum_k sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
                    # = sum_r U[i*m+j,r] * (sum_k V[j*p+k,r]*W[k*n+i,r]) = 1
                    coeff = np.zeros(R)
                    for k in range(p):
                        coeff += Va[j*p+k, :] * Wa[k*n+i, :]
                    if np.dot(coeff, coeff) > 1e-12:
                        U[i*m+j, :] = coeff / np.dot(coeff, coeff)
            
            # Solve for V: for fixed (j,k): need sum_i sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
            for j in range(m):
                for k in range(p):
                    coeff = np.zeros(R)
                    for i in range(n):
                        coeff += Ua[i*m+j, :] * Wa[k*n+i, :]
                    if np.dot(coeff, coeff) > 1e-12:
                        V[j*p+k, :] = coeff / np.dot(coeff, coeff)
            
            # Solve for W: for fixed (k,i): need sum_j sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
            for k in range(p):
                for i in range(n):
                    coeff = np.zeros(R)
                    for j in range(m):
                        coeff += U[i*m+j, :] * V[j*p+k, :]
                    if np.dot(coeff, coeff) > 1e-12:
                        W[k*n+i, :] = coeff / np.dot(coeff, coeff)
            
            err = compute_error(U, V, W)
            if err < best_err:
                best_err = err
                best = (U.copy(), V.copy(), W.copy())
            if err < 1e-10:
                break
        
        return best, best_err
    
    def try_rank(R, n_trials=5):
        """Try to find decomposition of rank R."""
        best_err = float('inf')
        best = None
        for trial in range(n_trials):
            if time.time() > deadline:
                break
            np.random.seed(trial * 7919 + R * 31)
            U = np.random.randn(nm, R) * 0.05
            V = np.random.randn(mp, R) * 0.05
            W = np.random.randn(pn, R) * 0.05
            U, V, W, err = als_solve(U, V, W, R)
            if err < best_err:
                best_err = err
                best = (U, V, W)
            if best_err < 1e-8:
                break
        if best is not None and best_err < 1e-6:
            # Try rounding to half-integers
            U_r, V_r, W_r = np.round(best[0]*2)/2, np.round(best[1]*2)/2, np.round(best[2]*2)/2
            if verify(U_r, V_r, W_r):
                return (U_r.tolist(), V_r.tolist(), W_r.tolist())
            # Try integer rounding
            U_i, V_i, W_i = np.round(best[0]), np.round(best[1]), np.round(best[2])
            if verify(U_i, V_i, W_i):
                return (U_i.tolist(), V_i.tolist(), W_i.tolist())
            if best_err < 1e-10:
                return (best[0].tolist(), best[1].tolist(), best[2].tolist())
        return None
    
    # Try to find lower-rank decomposition
    # Known: 4x4x4 has rank 49 (vs 64 schoolbook)
    # For 4x5x5, try ranks below 100
    for R in [70, 75, 80, 85, 90, 95]:
        if time.time() > deadline - 10:
            break
        result = try_rank(R, n_trials=8)
        if result is not None:
            return {"U": result[0], "V": result[1], "W": result[2]}
    
    # Fallback: schoolbook (rank = 100)
    rank = n * m * p
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
