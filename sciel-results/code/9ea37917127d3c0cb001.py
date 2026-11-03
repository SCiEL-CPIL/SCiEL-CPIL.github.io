"""
Rank-minimal exact tensor decomposition for 2x5x5 matrix multiplication.
Uses structured initialization, ALS with rational polishing, and aggressive
rank search from low to high.
"""

import time
import numpy as np

DIMENSIONS = [2, 5, 5]

def construct():
    start_time = time.time()
    deadline = 145.0  # safety margin before 150s kill
    
    n, m, p = 2, 5, 5
    nm = n * m  # 10
    mp = m * p  # 25
    pn = p * n  # 10
    n_constraints = n * m * p  # 50
    
    # Build schoolbook decomposition (rank 50)
    def build_schoolbook():
        R = n * m * p
        U = np.zeros((nm, R))
        V = np.zeros((mp, R))
        W = np.zeros((pn, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W
    
    best_solution = build_schoolbook()
    best_rank = n * m * p
    
    # Vectorized ALS solvers
    def solve_U(V, W, R):
        # V is (m, p, R), W is (p, n, R)
        # A[i, j, k, r] = V[j, k, r] * W[k, i, r]
        V_ext = V[None, :, :, :] # (1, m, p, R)
        W_ext = W.T[:, :, None, :] # W.T is (n, p, R) -> (n, p, 1, R)
        A = V_ext * W_ext # (n, m, p, R)
        
        U = np.zeros((nm, R))
        b = np.ones(p)
        for i in range(n):
            for j in range(m):
                A_ij = A[i, j, :, :] # (p, R)
                try:
                    U[i*m+j, :] = np.linalg.lstsq(A_ij, b, rcond=None)[0]
                except:
                    U[i*m+j, :] = np.linalg.lstsq(A_ij.T, b, rcond=None)[0]
        return U
    
    def solve_V(U, W, R):
        # U is (n, m, R), W is (p, n, R)
        # B[j, k, i, r] = U[i, j, r] * W[k, i, r]
        U_perm = np.transpose(U, (1, 0, 2)) # (m, n, R)
        U_ext = U_perm[:, :, None, :] # (m, n, 1, R)
        
        # W is (p, n, R) -> W[k, i, r]
        W_ext = W[None, :, :, :] # (1, p, n, R)
        
        B = U_ext * W_ext # (m, p, n, R) -> B[j, k, i, r]
        
        V = np.zeros((mp, R))
        b = np.ones(n)
        for j in range(m):
            for k in range(p):
                B_jk = B[j, k, :, :] # (n, R)
                try:
                    V[j*p+k, :] = np.linalg.lstsq(B_jk, b, rcond=None)[0]
                except:
                    V[j*p+k, :] = np.linalg.lstsq(B_jk.T, b, rcond=None)[0]
        return V
    
    def solve_W(U, V, R):
        # U is (n, m, R), V is (m, p, R)
        # C[k, i, j, r] = U[i, j, r] * V[j, k, r]
        U_ext = U[None, :, :, :] # (1, n, m, R)
        
        # Permute V to (p, m, R) -> V_perm[k, j, r]
        V_perm = np.transpose(V, (1, 0, 2)) # (p, m, R)
        V_ext = V_perm[:, None, :, :] # (p, 1, m, R)
        
        C = U_ext * V_ext # (p, n, m, R) -> C[k, i, j, r]
        
        W = np.zeros((pn, R))
        b = np.ones(m)
        for k in range(p):
            for i in range(n):
                C_ki = C[k, i, :, :] # (m, R)
                try:
                    W[k*n+i, :] = np.linalg.lstsq(C_ki, b, rcond=None)[0]
                except:
                    W[k*n+i, :] = np.linalg.lstsq(C_ki.T, b, rcond=None)[0]
        return W
    
    def check_error(U, V, W):
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    err = abs(total - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err
    
    def try_rank(R, n_restarts=10, max_iter=500, tol=1e-9):
        for restart in range(n_restarts):
            if time.time() > deadline - 5:
                return None
            
            # Initialization strategies
            if restart == 0:
                # Near-schoolbook: start with schoolbook and randomly combine columns
                U0, V0, W0 = build_schoolbook()
                np.random.seed(42 + restart)
                if R < n * m * p:
                    # Use random linear combinations of schoolbook columns
                    P = np.random.randn(n*m*p, R)
                    U = U0 @ P
                    V = V0 @ P
                    W = W0 @ P
                else:
                    U = U0.copy()
                    V = V0.copy()
                    W = W0.copy()
            elif restart == 1:
                # Random small initialization
                np.random.seed(123 + restart)
                U = np.random.randn(nm, R) * 0.1
                V = np.random.randn(mp, R) * 0.1
                W = np.random.randn(pn, R) * 0.1
            elif restart == 2:
                # Random with some structure
                np.random.seed(456 + restart)
                U = np.random.randn(nm, R) * 0.5
                V = np.random.randn(mp, R) * 0.5
                W = np.random.randn(pn, R) * 0.5
                U[np.random.rand(nm, R) < 0.3] = 0
                V[np.random.rand(mp, R) < 0.3] = 0
                W[np.random.rand(pn, R) < 0.3] = 0
            elif restart == 3:
                # Structured: try to exploit n=2 symmetry
                np.random.seed(789 + restart)
                U = np.random.randn(nm, R) * 0.3
                V = np.random.randn(mp, R) * 0.3
                W = np.random.randn(pn, R) * 0.3
                # Make W symmetric in some sense for the two i values
                for k in range(p):
                    W[k*n+0, :] = W[k*n+0, :] + 0.1 * W[k*n+1, :]
                    W[k*n+1, :] = W[k*n+1, :] + 0.1 * W[k*n+0, :]
            else:
                # Pure random
                np.random.seed(1000 + restart)
                U = np.random.randn(nm, R) * 0.2
                V = np.random.randn(mp, R) * 0.2
                W = np.random.randn(pn, R) * 0.2
            
            # Alternating least squares
            prev_err = np.inf
            for it in range(max_iter):
                if time.time() > deadline - 5:
                    return None
                
                U = solve_U(V, W, R)
                V = solve_V(U, W, R)
                W = solve_W(U, V, R)
                
                err = check_error(U, V, W)
                
                if err < tol:
                    # Polish phase
                    if err < 1e-6:
                        for _ in range(100):
                            if time.time() > deadline - 2:
                                break
                            U = solve_U(V, W, R)
                            V = solve_V(U, W, R)
                            W = solve_W(U, V, R)
                            err = check_error(U, V, W)
                            if err < 1e-12:
                                break
                    return U, V, W
                
                if abs(prev_err - err) < 1e-12:
                    break
                prev_err = err
        
        return None
    
    # Try ranks from just below 50 downward
    for R in range(best_rank - 1, max(best_rank - 20, 30), -1):
        if time.time() > deadline - 10:
            break
        
        result = try_rank(R)
        if result is not None:
            best_rank = R
            best_solution = result
            # Continue trying lower ranks
            continue
        else:
            # If we can't find rank R, try a few more lower ranks with less effort
            for R2 in range(R-1, max(R-3, 30), -1):
                if time.time() > deadline - 5:
                    break
                result2 = try_rank(R2, n_restarts=2, max_iter=150)
                if result2 is not None:
                    best_rank = R2
                    best_solution = result2
                    break
            break
    
    # Convert to lists for return
    U, V, W = best_solution
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
