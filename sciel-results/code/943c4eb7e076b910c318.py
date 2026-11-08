"""Matrix multiplication 2,4,5 — rank-minimal tensor decomposition."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time
from fractions import Fraction

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin before 120s kill
    
    def make_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def verify_exact(U_list, V_list, W_list, R):
        """Check if decomposition is exact: T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(R):
                        s += U_list[i*m+j][r] * V_list[j*p+k][r] * W_list[k*n+i][r]
                    if s != 1:
                        return False
        return True
    
    def verify_float(U_np, V_np, W_np, R, tol=1e-9):
        """Quick float check"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(R):
                        s += U_np[i*m+j, r] * V_np[j*p+k, r] * W_np[k*n+i, r]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    def try_rank_als(R, max_iters=300, n_restarts=5, seed_base=0):
        """Try to find exact decomposition of rank R using ALS-like refinement"""
        for restart in range(n_restarts):
            if time.time() > deadline:
                return None
            rng = np.random.RandomState(seed_base + restart)
            U = rng.randn(n*m, R)
            V = rng.randn(m*p, R)
            W = rng.randn(p*n, R)
            
            best_err = float('inf')
            
            for it in range(max_iters):
                if time.time() > deadline:
                    return None
                
                # Update U: for each row (i,j), U[i*m+j,:] should satisfy
                # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all k
                # Fix V,W. For fixed i,j:
                # sum_r U[i*m+j,r] * (V[j*p+k,r] * W[k*n+i,r]) = 1 for all k
                # This is a linear system: M_{i,j}^T @ U[i*m+j,:] = 1 (vector of length p)
                # where M_{i,j}[k,r] = V[j*p+k,r] * W[k*n+i,r]
                
                for i in range(n):
                    for j in range(m):
                        # Build matrix M of shape (p, R): M[k,r] = V[j*p+k,r] * W[k*n+i,r]
                        M = np.zeros((p, R))
                        for k in range(p):
                            M[k, :] = V[j*p+k, :] * W[k*n+i, :]
                        # Solve U[i*m+j,:] such that M^T @ U[i*m+j,:] = 1 (length p)
                        # i.e., U[i*m+j,:] @ M = 1
                        # Use least squares: U[i*m+j,:] = argmin ||U[i*m+j,:] @ M - 1||^2
                        try:
                            U[i*m+j, :] = np.linalg.lstsq(M.T, np.ones(p), rcond=None)[0]
                        except:
                            pass
                
                # Update V: for each row (j,k), V[j*p+k,:] should satisfy
                # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all i
                # Fix U,W. For fixed j,k:
                # sum_r V[j*p+k,r] * (U[i*m+j,r] * W[k*n+i,r]) = 1 for all i
                # This is a linear system: N_{j,k}^T @ V[j*p+k,:] = 1 (vector of length n)
                # where N_{j,k}[i,r] = U[i*m+j,r] * W[k*n+i,r]
                
                for j in range(m):
                    for k in range(p):
                        # Build matrix N of shape (n, R): N[i,r] = U[i*m+j,r] * W[k*n+i,r]
                        N = np.zeros((n, R))
                        for i in range(n):
                            N[i, :] = U[i*m+j, :] * W[k*n+i, :]
                        # Solve V[j*p+k,:] such that N^T @ V[j*p+k,:] = 1 (length n)
                        try:
                            V[j*p+k, :] = np.linalg.lstsq(N.T, np.ones(n), rcond=None)[0]
                        except:
                            pass
                
                # Update W: for each row (k,i), W[k*n+i,:] should satisfy
                # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all j
                # Fix U,V. For fixed k,i:
                # sum_r W[k*n+i,r] * (U[i*m+j,r] * V[j*p+k,r]) = 1 for all j
                # This is a linear system: P_{k,i}^T @ W[k*n+i,:] = 1 (vector of length m)
                # where P_{k,i}[j,r] = U[i*m+j,r] * V[j*p+k,r]
                
                for k in range(p):
                    for i in range(n):
                        # Build matrix P of shape (m, R): P[j,r] = U[i*m+j,r] * V[j*p+k,r]
                        P = np.zeros((m, R))
                        for j in range(m):
                            P[j, :] = U[i*m+j, :] * V[j*p+k, :]
                        # Solve W[k*n+i,:] such that P^T @ W[k*n+i,:] = 1 (length m)
                        try:
                            W[k*n+i, :] = np.linalg.lstsq(P.T, np.ones(m), rcond=None)[0]
                        except:
                            pass
                
                # Compute error
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for r in range(R):
                                s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                            err += (s - 1.0) ** 2
                
                if err < best_err:
                    best_err = err
                    if err < 1e-12:
                        # Found exact solution, convert to fractions
                        U_frac = [[Fraction(U[i*m+j, r]).limit_denominator(1000000) for r in range(R)] for i in range(n) for j in range(m)]
                        V_frac = [[Fraction(V[j*p+k, r]).limit_denominator(1000000) for r in range(R)] for j in range(m) for k in range(p)]
                        W_frac = [[Fraction(W[k*n+i, r]).limit_denominator(1000000) for r in range(R)] for k in range(p) for i in range(n)]
                        
                        # Verify exact
                        if verify_exact(U_frac, V_frac, W_frac, R):
                            return {"U": U_frac, "V": V_frac, "W": W_frac}
                
                if time.time() > deadline:
                    return None
            
            if time.time() > deadline:
                return None
    
    # Try ranks from 21 up to 39 (below schoolbook 40)
    for R in range(21, 40):
        if time.time() > deadline:
            break
        result = try_rank_als(R, max_iters=200, n_restarts=3, seed_base=R*100)
        if result is not None:
            return result
    
    # Fallback to Schoolbook
    return make_schoolbook()
# EVOLVE-BLOCK-END
