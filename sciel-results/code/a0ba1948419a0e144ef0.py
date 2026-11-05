"""Matrix multiplication 3,3,7 - rank-minimal exact tensor decomposition."""
import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Schoolbook decomposition (rank 63)
    rank_sb = n * m * p
    U_sb = [[0]*rank_sb for _ in range(n*m)]
    V_sb = [[0]*rank_sb for _ in range(m*p)]
    W_sb = [[0]*rank_sb for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j][col] = 1
                V_sb[j*p+k][col] = 1
                W_sb[k*n+i][col] = 1
                col += 1
    
    def find_decomposition(r, max_iters=3000):
        """Try to find a rank-r exact decomposition."""
        np.random.seed(999)
        
        U = np.random.randn(n*m, r)
        V = np.random.randn(m*p, r)
        W = np.random.randn(p*n, r)
        
        for it in range(max_iters):
            if time.time() > deadline - 15:
                return None
            
            # Update U: for each (i,j), solve linear system in U[i*m+j, :]
            for i in range(n):
                for j in range(m):
                    M = np.zeros((p, r))
                    for k in range(p):
                        M[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    try:
                        U[i*m+j, :] = np.linalg.lstsq(M, np.ones(p), rcond=None)[0]
                    except:
                        pass
            
            # Update V: for each (j,k), solve linear system in V[j*p+k, :]
            for j in range(m):
                for k in range(p):
                    M = np.zeros((n, r))
                    for i in range(n):
                        M[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    try:
                        V[j*p+k, :] = np.linalg.lstsq(M, np.ones(n), rcond=None)[0]
                    except:
                        pass
            
            # Update W: for each (k,i), solve linear system in W[k*n+i, :]
            for k in range(p):
                for i in range(n):
                    M = np.zeros((m, r))
                    for j in range(m):
                        M[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    try:
                        W[k*n+i, :] = np.linalg.lstsq(M, np.ones(m), rcond=None)[0]
                    except:
                        pass
            
            # Check residual
            max_res = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                        max_res = max(max_res, abs(s - 1.0))
            
            if max_res < 1e-10:
                # Convert to exact fractions
                U_f = [[Fraction(U[idx, rr]).limit_denominator(10**6) for rr in range(r)] for idx in range(n*m)]
                V_f = [[Fraction(V[idx, rr]).limit_denominator(10**6) for rr in range(r)] for idx in range(m*p)]
                W_f = [[Fraction(W[idx, rr]).limit_denominator(10**6) for rr in range(r)] for idx in range(p*n)]
                
                ok = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = Fraction(0)
                            for rr in range(r):
                                s += U_f[i*m+j][rr] * V_f[j*p+k][rr] * W_f[k*n+i][rr]
                            if s != 1:
                                ok = False
                                break
                        if not ok:
                            break
                    if not ok:
                        break
                if ok:
                    return {"U": U_f, "V": V_f, "W": W_f}
        
        return None
    
    # Try decreasing ranks
    for r in range(50, rank_sb + 1):
        if time.time() > deadline - 20:
            break
        result = find_decomposition(r)
        if result is not None:
            return result
    
    return {"U": U_sb, "V": V_sb, "W": W_sb}
