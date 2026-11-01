import numpy as np
import time

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120
    
    # Build target tensor
    # target[a, b, c] where a = i*m+j, b = j*p+k, c = k*n+i
    # target = 1 if (a//m == c%n) and (a%m == b//p) and (b%p == c//n) else 0
    target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target[a, b, c] = 1.0
    
    best_rank = n * m * p
    best_solution = None
    
    # Schoolbook solution as fallback
    U_sb = np.zeros((n*m, best_rank))
    V_sb = np.zeros((m*p, best_rank))
    W_sb = np.zeros((p*n, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    best_solution = (U_sb, V_sb, W_sb)
    
    # Try alternating minimization for various ranks
    np.random.seed(42)
    
    for rank in range(20, best_rank):
        if time.time() > deadline - 30:
            break
        
        for attempt in range(50):
            if time.time() > deadline - 30:
                break
            
            U = np.random.randn(n*m, rank)
            V = np.random.randn(m*p, rank)
            W = np.random.randn(p*n, rank)
            
            # Alternating minimization
            for it in range(200):
                # Update W: for each c, solve least squares
                for c in range(p*n):
                    # Build matrix: rows are (a,b) pairs, cols are r
                    # A_ls[r] = sum over... no
                    # For fixed c: sum_r U[a,r]*V[b,r]*W[c,r] = target[a,b,c]
                    # This is: sum_r (U[a,r]*V[b,r]) * W[c,r] = target[a,b,c]
                    # So for each (a,b), we have a row: [U[a,0]*V[b,0], U[a,1]*V[b,1], ..., U[a,R-1]*V[b,R-1]] dot W[c,:] = target[a,b,c]
                    
                    A_ls = np.zeros((n*m, rank))
                    for a in range(n*m):
                        for b in range(m*p):
                            pass
                    # Actually, let me vectorize
                    # A_ls[a, r] = U[a, r] * V[b, r] for each b
                    # But we need separate rows for each (a,b)
                    
                    # Better: create a matrix of shape (n*m * m*p, rank)
                    # Row index = a * m*p + b
                    # Entry = U[a, r] * V[b, r]
                    
                    A_mat = np.zeros((n*m * m*p, rank))
                    rhs = np.zeros(n*m * m*p)
                    idx = 0
                    for a in range(n*m):
                        for b in range(m*p):
                            A_mat[idx, :] = U[a, :] * V[b, :]
                            rhs[idx] = target[a, b, c]
                            idx += 1
                    
                    sol, _, _, _ = np.linalg.lstsq(A_mat, rhs, rcond=None)
                    W[c, :] = sol
                
                # Update V: for each b, solve least squares
                for b in range(m*p):
                    A_mat = np.zeros((n*m * p*n, rank))
                    rhs = np.zeros(n*m * p*n)
                    idx = 0
                    for a in range(n*m):
                        for c in range(p*n):
                            A_mat[idx, :] = U[a, :] * W[c, :]
                            rhs[idx] = target[a, b, c]
                            idx += 1
                    sol, _, _, _ = np.linalg.lstsq(A_mat, rhs, rcond=None)
                    V[b, :] = sol
                
                # Update U: for each a, solve least squares
                for a in range(n*m):
                    A_mat = np.zeros((m*p * p*n, rank))
                    rhs = np.zeros(m*p * p*n)
                    idx = 0
                    for b in range(m*p):
                        for c in range(p*n):
                            A_mat[idx, :] = V[b, :] * W[c, :]
                            rhs[idx] = target[a, b, c]
                            idx += 1
                    sol, _, _, _ = np.linalg.lstsq(A_mat, rhs, rcond=None)
                    U[a, :] = sol
                
                # Check convergence
                err = 0
                for a in range(n*m):
                    for b in range(m*p):
                        for c in range(p*n):
                            val = np.sum(U[a, :] * V[b, :] * W[c, :])
                            err += (val - target[a, b, c])**2
                if err < 1e-10:
                    break
            
            if err < 1e-10:
                # Try to round to half-integers
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                err_rounded = 0
                for a in range(n*m):
                    for b in range(m*p):
                        for c in range(p*n):
                            val = np.sum(U_rounded[a, :] * V_rounded[b, :] * W_rounded[c, :])
                            err_rounded += (val - target[a, b, c])**2
                
                if err_rounded < 1e-9:
                    best_rank = rank
                    best_solution = (U_rounded, V_rounded, W_rounded)
                    break
        
        if best_rank == rank:
            break
    
    U, V, W = best_solution
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
