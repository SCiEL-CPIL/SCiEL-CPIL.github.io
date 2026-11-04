import numpy as np
import time

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m  # 8
    mp = m * p  # 16
    pn = p * n  # 8
    
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Schoolbook fallback
    best_rank = n * m * p
    best_U = [[0] * best_rank for _ in range(nm)]
    best_V = [[0] * best_rank for _ in range(mp)]
    best_W = [[0] * best_rank for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i*m+j][col] = 1
                best_V[j*p+k][col] = 1
                best_W[k*n+i][col] = 1
                col += 1
    
    def check_decomposition(U, V, W, tol=1e-9):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def compute_max_error(U, V, W):
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err
    
    # Try ranks from lower bound upward
    for rank in range(16, best_rank):
        if time.time() > deadline - 40:
            break
        
        found = False
        max_seeds = 80
        for seed in range(max_seeds):
            if time.time() > deadline - 40:
                break
            
            np.random.seed(seed * 7919 + rank * 104729 + 12345)
            
            # Initialize with varied scales
            scale = 0.5 + 0.5 * np.random.rand()
            U = np.random.randn(nm, rank) * scale
            V = np.random.randn(mp, rank) * scale
            W = np.random.randn(pn, rank) * scale
            
            # ALS iterations
            converged = False
            prev_err = float('inf')
            for iteration in range(3000):
                if time.time() > deadline - 40:
                    break
                
                # Update W: for each (i,k), solve M.T @ w = ones(m)
                for i in range(n):
                    for k in range(p):
                        M = np.zeros((m, rank))
                        for j in range(m):
                            M[j, :] = U[i*m+j, :] * V[j*p+k, :]
                        try:
                            W[k*n+i, :] = np.linalg.lstsq(M.T, np.ones(m), rcond=1e-12)[0]
                        except:
                            pass
                
                # Update U: for each (i,j), solve N.T @ u = ones(p)
                for i in range(n):
                    for j in range(m):
                        N = np.zeros((p, rank))
                        for k in range(p):
                            N[k, :] = V[j*p+k, :] * W[k*n+i, :]
                        try:
                            U[i*m+j, :] = np.linalg.lstsq(N.T, np.ones(p), rcond=1e-12)[0]
                        except:
                            pass
                
                # Update V: for each (j,k), solve P.T @ v = ones(n)
                for j in range(m):
                    for k in range(p):
                        P = np.zeros((n, rank))
                        for i in range(n):
                            P[i, :] = U[i*m+j, :] * W[k*n+i, :]
                        try:
                            V[j*p+k, :] = np.linalg.lstsq(P.T, np.ones(n), rcond=1e-12)[0]
                        except:
                            pass
                
                # Check convergence
                cur_err = compute_max_error(U, V, W)
                if cur_err < 1e-10:
                    converged = True
                    break
                if abs(prev_err - cur_err) < 1e-15 and cur_err < 1e-6:
                    # Stalled but close - try a few more iterations
                    if iteration > 100:
                        break
                prev_err = cur_err
            
            if converged and check_decomposition(U, V, W, 1e-8):
                found = True
                best_rank = rank
                best_U = U.tolist()
                best_V = V.tolist()
                best_W = W.tolist()
                break
        
        if found:
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
