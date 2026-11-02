import numpy as np
from itertools import product
import time

DIMENSIONS = [2, 4, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    n, m, p = DIMENSIONS
    # U: (n*m, r) = (8, r)
    # V: (m*p, r) = (16, r)  
    # W: (p*n, r) = (8, r)
    # Constraint: for all i,j,k: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
    
    # Try decreasing rank
    best_result = None
    for r in range(16, 33):  # Start from lower bound 16
        if time.time() > deadline:
            break
        # Try to find decomposition of rank r
        # Use random initialization and least squares
        success = False
        for attempt in range(50):
            if time.time() > deadline:
                break
            np.random.seed(attempt)
            U = np.random.randn(n*m, r)
            V = np.random.randn(m*p, r)
            W = np.random.randn(p*n, r)
            
            # Set up linear system: for each (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
            # This is nonlinear in U,V,W. Use iterative approach.
            
            # Alternative: fix two, solve for third
            # Fix U and V, solve for W
            # For each (k,i): sum_r W[k*n+i,r] * (sum_j U[i*m+j,r]*V[j*p+k,r]... no
            
            # Actually the constraint is: for each (i,j,k):
            # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
            # Group by (k,i): for fixed k,i, sum over j: 
            # sum_j sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = sum_j 1 = m = 4
            # Let M_{k,i,r} = sum_j U[i*m+j,r] * V[j*p+k,r]
            # Then sum_r M_{k,i,r} * W[k*n+i,r] = 4
            # This is linear in W for fixed U,V!
            
            # Build matrix: for each (k,i), we have equation sum_r M_{k,i,r} * W[k*n+i,r] = 4
            # But M depends on U and V.
            
            # Let's use an iterative approach:
            # 1. Initialize U, V, W randomly
            # 2. Iterate: fix U,V, solve for W; fix V,W, solve for U; fix U,W, solve for V
            
            U_curr = U.copy()
            V_curr = V.copy()
            W_curr = W.copy()
            
            for iteration in range(100):
                if time.time() > deadline:
                    break
                    
                # Solve for W: for each (k,i), sum_r W[k*n+i,r] * (sum_j U[i*m+j,r]*V[j*p+k,r]) = m
                # Let A_{(k,i), r} = sum_j U[i*m+j,r] * V[j*p+k,r]
                # Then A @ W[k*n+i,:] = m for each (k,i)
                
                # Build A matrix: shape (p*n, r) where A[k*n+i, r] = sum_j U[i*m+j,r] * V[j*p+k,r]
                A_mat = np.zeros((p*n, r))
                for k in range(p):
                    for i in range(n):
                        for r_idx in range(r):
                            A_mat[k*n+i, r_idx] = sum(U_curr[i*m+j, r_idx] * V_curr[j*p+k, r_idx] for j in range(m))
                
                # Solve A_mat @ W_col = m for each column... no, W is (p*n, r)
                # For each row (k*n+i) of W, we need: sum_r A_mat[k*n+i,r] * W[k*n+i,r] = m
                # This is: A_mat @ W.T = m * ones(p*n) ... no
                # Actually: for each (k,i), the equation is dot(A_mat[k*n+i,:], W[k*n+i,:]) = m
                # This is underdetermined if r > 1. We can use least squares.
                
                # Minimize sum_{k,i} (dot(A_mat[k*n+i,:], W[k*n+i,:]) - m)^2
                # This is equivalent to: minimize ||A_mat * W.T - m*ones||_F^2 where W.T is (r, p*n)
                # Actually let's just solve row by row with least squares
                
                W_new = np.zeros_like(W_curr)
                for ki in range(p*n):
                    # Solve: A_mat[ki,:] @ W_new[ki,:] = m
                    # Underdetermined, use minimum norm solution
                    if np.linalg.norm(A_mat[ki,:]) > 1e-10:
                        W_new[ki,:] = A_mat[ki,:] / np.dot(A_mat[ki,:], A_mat[ki,:]) * m
                    else:
                        W_new[ki,:] = 0
                W_curr = W_new
                
                # Solve for U: for each (i,j), sum_r U[i*m+j,r] * (sum_k V[j*p+k,r] * W[k*n+i,r]) = p
                # Let B_{(i,j), r} = sum_k V[j*p+k,r] * W[k*n+i,r]
                B_mat = np.zeros((n*m, r))
                for i in range(n):
                    for j in range(m):
                        for r_idx in range(r):
                            B_mat[i*m+j, r_idx] = sum(V_curr[j*p+k, r_idx] * W_curr[k*n+i, r_idx] for k in range(p))
                
                U_new = np.zeros_like(U_curr)
                for ij in range(n*m):
                    if np.linalg.norm(B_mat[ij,:]) > 1e-10:
                        U_new[ij,:] = B_mat[ij,:] / np.dot(B_mat[ij,:], B_mat[ij,:]) * p
                    else:
                        U_new[ij,:] = 0
                U_curr = U_new
                
                # Solve for V: for each (j,k), sum_r V[j*p+k,r] * (sum_i U[i*m+j,r] * W[k*n+i,r]) = n
                C_mat = np.zeros((m*p, r))
                for j in range(m):
                    for k in range(p):
                        for r_idx in range(r):
                            C_mat[j*p+k, r_idx] = sum(U_curr[i*m+j, r_idx] * W_curr[k*n+i, r_idx] for i in range(n))
                
                V_new = np.zeros_like(V_curr)
                for jk in range(m*p):
                    if np.linalg.norm(C_mat[jk,:]) > 1e-10:
                        V_new[jk,:] = C_mat[jk,:] / np.dot(C_mat[jk,:], C_mat[jk,:]) * n
                    else:
                        V_new[jk,:] = 0
                V_curr = V_new
            
            # Check if we have a valid solution
            # Verify: for all i,j,k: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
            max_err = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = sum(U_curr[i*m+j, r_idx] * V_curr[j*p+k, r_idx] * W_curr[k*n+i, r_idx] for r_idx in range(r))
                        max_err = max(max_err, abs(val - 1))
            
            if max_err < 1e-6:
                # Round to nearest half-integers
                U_rounded = np.round(U_curr * 2) / 2
                V_rounded = np.round(V_curr * 2) / 2
                W_rounded = np.round(W_curr * 2) / 2
                
                # Verify rounded solution
                max_err_rounded = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = sum(U_rounded[i*m+j, r_idx] * V_rounded[j*p+k, r_idx] * W_rounded[k*n+i, r_idx] for r_idx in range(r))
                            max_err_rounded = max(max_err_rounded, abs(val - 1))
                
                if max_err_rounded < 1e-9:
                    # Convert to lists
                    U_list = [[float(U_rounded[i, j]) for j in range(r)] for i in range(n*m)]
                    V_list = [[float(V_rounded[i, j]) for j in range(r)] for i in range(m*p)]
                    W_list = [[float(W_rounded[i, j]) for j in range(r)] for i in range(p*n)]
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    return best_result
                else:
                    # Try to fix with more iterations
                    pass
            if time.time() > deadline:
                break
        
        if best_result is not None:
            return best_result
    
    # Fallback: return schoolbook
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
