import numpy as np
import time
from fractions import Fraction

n, m, p = 2, 5, 6
nm, mp, pn = n * m, m * p, p * n

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Schoolbook decomposition as fallback
    rank = n * m * p
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank
    
    # Try to find lower rank decomposition
    # The tensor T[i,j,k] = delta (for matrix mult) can be decomposed as:
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Try ranks from 50 down to 30
    for target_rank in range(50, 29, -1):
        if time.time() > deadline - 10:
            break
            
        # Use ALS (Alternating Least Squares) to find decomposition
        np.random.seed(42)
        
        for attempt in range(3):
            if time.time() > deadline - 5:
                break
                
            U_curr = np.random.randn(nm, target_rank)
            V_curr = np.random.randn(mp, target_rank)
            W_curr = np.random.randn(pn, target_rank)
            
            # Scale to reasonable magnitude
            U_curr /= np.linalg.norm(U_curr) * np.sqrt(target_rank)
            V_curr /= np.linalg.norm(V_curr) * np.sqrt(target_rank)
            W_curr /= np.linalg.norm(W_curr) * np.sqrt(target_rank)
            
            # ALS iterations
            converged = False
            for it in range(200):
                if time.time() > deadline - 5:
                    break
                    
                # Update U
                # For each (i,j), we want U[i*m+j, :] to satisfy:
                # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all k
                # This is: U[i*m+j, :] * diag(W[k*n+i, :]) * V[j*p+k, :]^T = 1
                # Let's form the matrix: for fixed (i,j), the constraint is over k
                # sum_r U[i*m+j, r] * (V[j*p+k, r] * W[k*n+i, r]) = 1 for k=0..p-1
                # This is a linear system: U[i*m+j, :] * M_{ij} = ones(p)
                # where M_{ij}[k, r] = V[j*p+k, r] * W[k*n+i, r]
                
                # Build the system for all (i,j) at once
                # Actually, let's do it more carefully
                
                # For U update:
                # For each i, j: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all k
                # Let A_{ij}[k, r] = V[j*p+k, r] * W[k*n+i, r]
                # Then U[i*m+j, :] @ A_{ij} = ones(p)
                
                U_new = np.zeros_like(U_curr)
                for i in range(n):
                    for j in range(m):
                        # Build A_{ij} of shape (p, target_rank)
                        A_ii = np.zeros((p, target_rank))
                        for k in range(p):
                            A_ii[k, :] = V_curr[j*p+k, :] * W_curr[k*n+i, :]
                        # Solve U[i*m+j, :] @ A_ii = ones(p)
                        # This is A_ii^T @ U[i*m+j, :]^T = ones(p)
                        try:
                            sol, residuals, rank_sys, sv = np.linalg.lstsq(A_ii.T, np.ones(p), rcond=None)
                            U_new[i*m+j, :] = sol
                        except:
                            U_new[i*m+j, :] = U_curr[i*m+j, :]
                
                U_curr = U_new
                
                # Update V
                V_new = np.zeros_like(V_curr)
                for j in range(m):
                    for k in range(p):
                        # For fixed (j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i
                        # Let B_{jk}[i, r] = U[i*m+j, r] * W[k*n+i, r]
                        # Then V[j*p+k, :] @ B_{jk} = ones(n)
                        B_jk = np.zeros((n, target_rank))
                        for i in range(n):
                            B_jk[i, :] = U_curr[i*m+j, :] * W_curr[k*n+i, :]
                        try:
                            sol, residuals, rank_sys, sv = np.linalg.lstsq(B_jk.T, np.ones(n), rcond=None)
                            V_new[j*p+k, :] = sol
                        except:
                            V_new[j*p+k, :] = V_curr[j*p+k, :]
                
                V_curr = V_new
                
                # Update W
                W_new = np.zeros_like(W_curr)
                for k in range(p):
                    for i in range(n):
                        # For fixed (k,i): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all j
                        # Let C_{ki}[j, r] = U[i*m+j, r] * V[j*p+k, r]
                        # Then W[k*n+i, :] @ C_{ki} = ones(m)
                        C_ki = np.zeros((m, target_rank))
                        for j in range(m):
                            C_ki[j, :] = U_curr[i*m+j, :] * V_curr[j*p+k, :]
                        try:
                            sol, residuals, rank_sys, sv = np.linalg.lstsq(C_ki.T, np.ones(m), rcond=None)
                            W_new[k*n+i, :] = sol
                        except:
                            W_new[k*n+i, :] = W_curr[k*n+i, :]
                
                W_curr = W_new
                
                # Check convergence and error
                # Compute the error: for all (i,j,k), check if sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] ≈ 1
                err = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U_curr[i*m+j, :], V_curr[j*p+k, :] * W_curr[k*n+i, :])
                            err += (val - 1.0)**2
                err = np.sqrt(err)
                
                if err < 1e-8:
                    converged = True
                    break
                
                if it % 10 == 0 and time.time() > deadline - 10:
                    break
            
            if converged:
                # Try to rationalize
                U_rat = np.zeros((nm, target_rank))
                V_rat = np.zeros((mp, target_rank))
                W_rat = np.zeros((pn, target_rank))
                
                success = True
                for idx in range(nm):
                    for r in range(target_rank):
                        f = Fraction(U_curr[idx, r]).limit_denominator(10**6)
                        U_rat[idx, r] = f
                for idx in range(mp):
                    for r in range(target_rank):
                        f = Fraction(V_curr[idx, r]).limit_denominator(10**6)
                        V_rat[idx, r] = f
                for idx in range(pn):
                    for r in range(target_rank):
                        f = Fraction(W_curr[idx, r]).limit_denominator(10**6)
                        W_rat[idx, r] = f
                
                # Verify exact decomposition with rationals
                exact_ok = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = Fraction(0)
                            for r in range(target_rank):
                                val += U_rat[i*m+j, r] * V_rat[j*p+k, r] * W_rat[k*n+i, r]
                            if val != Fraction(1):
                                exact_ok = False
                                break
                        if not exact_ok:
                            break
                    if not exact_ok:
                        break
                
                if exact_ok:
                    best_U, best_V, best_W = U_rat, V_rat, W_rat
                    best_rank = target_rank
                    break
                else:
                    # Try with higher precision
                    U_rat = np.zeros((nm, target_rank))
                    V_rat = np.zeros((mp, target_rank))
                    W_rat = np.zeros((pn, target_rank))
                    
                    success = True
                    for idx in range(nm):
                        for r in range(target_rank):
                            f = Fraction(U_curr[idx, r]).limit_denominator(10**9)
                            U_rat[idx, r] = f
                    for idx in range(mp):
                        for r in range(target_rank):
                            f = Fraction(V_curr[idx, r]).limit_denominator(10**9)
                            V_rat[idx, r] = f
                    for idx in range(pn):
                        for r in range(target_rank):
                            f = Fraction(W_curr[idx, r]).limit_denominator(10**9)
                            W_rat[idx, r] = f
                    
                    exact_ok = True
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = Fraction(0)
                                for r in range(target_rank):
                                    val += U_rat[i*m+j, r] * V_rat[j*p+k, r] * W_rat[k*n+i, r]
                                if val != Fraction(1):
                                    exact_ok = False
                                    break
                            if not exact_ok:
                                break
                        if not exact_ok:
                            break
                    
                    if exact_ok:
                        best_U, best_V, best_W = U_rat, V_rat, W_rat
                        best_rank = target_rank
                        break
            else:
                if time.time() > deadline - 5:
                    break
        
        if best_rank < rank:
            break
    
    # Convert to lists
    U_out = np.zeros((nm, best_rank))
    V_out = np.zeros((mp, best_rank))
    W_out = np.zeros((pn, best_rank))
    
    for i in range(nm):
        for r in range(best_rank):
            U_out[i, r] = float(best_U[i, r])
    for i in range(mp):
        for r in range(best_rank):
            V_out[i, r] = float(best_V[i, r])
    for i in range(pn):
        for r in range(best_rank):
            W_out[i, r] = float(best_W[i, r])
    
    return {"U": U_out.tolist(), "V": V_out.tolist(), "W": W_out.tolist()}
