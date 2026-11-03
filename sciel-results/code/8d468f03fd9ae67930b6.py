import numpy as np
import time
import random

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140.0  # 10s safety margin
    
    # Target tensor: T[(i,j), (j',k'), (k'',i')] = 1 if j=j', k'=k'', i=i' else 0
    # This is the matrix multiplication tensor.
    
    # We want to find U, V, W of shapes (n*m, r), (m*p, r), (p*n, r)
    # such that sum_k U[i,j,k] * V[j,k',k] * W[k'',i,k] = T[i,j,k',k'',i]
    # Note: The indices in the problem statement for W are W[k*n+i, r].
    # So W is indexed by (k, i) -> k*n + i.
    # U is indexed by (i, j) -> i*m + j.
    # V is indexed by (j, k) -> j*p + k.
    
    # The condition is:
    # For all i, j, k, l, i' (let's use distinct indices to avoid confusion)
    # Let A index be (i, j), B index be (j, k), C index be (k, i).
    # T[A, B, C] = 1 if B's first index == A's second index AND C's first index == B's second index AND C's second index == A's first index.
    # i.e., T[i*m+j, j*p+k, k*n+i] = 1.
    # Otherwise 0.
    
    # We search for rank r from 30 to 60.
    # We use a randomized search with ALS (Alternating Least Squares) to minimize
    # || T - sum_k U_k \otimes V_k \otimes W_k ||_F
    
    # Flatten T into a vector of size n*m * m*p * p*n
    # But T is sparse. We only need to check the non-zero entries.
    # Non-zero entries are at (i*m+j, j*p+k, k*n+i) for all i, j, k.
    # There are n*m*p such entries, and they are all 1.
    
    # We can define the residual for a given (U, V, W) as:
    # err = sum_{i,j,k} ( sum_{r} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] - 1 )^2
    
    # We will try to minimize this error.
    # If the error is 0, we have found a valid decomposition.
    
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    # Initialize schoolbook as fallback
    U_school = np.zeros((n*m, n*m*p))
    V_school = np.zeros((m*p, n*m*p))
    W_school = np.zeros((p*n, n*m*p))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    # Search for lower rank
    for r in range(30, 61):
        if time.time() > deadline:
            break
        
        # Try a few random initializations
        found = False
        for attempt in range(10):
            if time.time() > deadline:
                break
            
            # Random initialization
            U = np.random.randn(n*m, r)
            V = np.random.randn(m*p, r)
            W = np.random.randn(p*n, r)
            
            # ALS iterations
            for it in range(100):
                # Update U
                # For each (i,j), we want U[i*m+j, :] to minimize
                # sum_{k} ( sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] - 1 )^2
                # This is a linear system in U[i*m+j, :].
                # Let x = U[i*m+j, :].
                # Minimize sum_k ( x . (V[j*p+k, :] * W[k*n+i, :]) - 1 )^2
                # Let M[i,j,k] = V[j*p+k, :] * W[k*n+i, :]
                # Minimize || M @ x - 1 ||^2
                # x = (M^T M)^{-1} M^T 1
                
                for i in range(n):
                    for j in range(m):
                        # M is p x r
                        M = np.zeros((p, r))
                        for k in range(p):
                            M[k, :] = V[j*p+k, :] * W[k*n+i, :]
                        # Solve M @ x = 1
                        # x = pinv(M) @ 1
                        x = np.linalg.pinv(M) @ np.ones(p)
                        U[i*m+j, :] = x
                
                # Update V
                for j in range(m):
                    for k in range(p):
                        # M is n x r
                        M = np.zeros((n, r))
                        for i in range(n):
                            M[i, :] = U[i*m+j, :] * W[k*n+i, :]
                        x = np.linalg.pinv(M) @ np.ones(n)
                        V[j*p+k, :] = x
                
                # Update W
                for k in range(p):
                    for i in range(n):
                        # M is m x r
                        M = np.zeros((m, r))
                        for j in range(m):
                            M[j, :] = U[i*m+j, :] * V[j*p+k, :]
                        x = np.linalg.pinv(M) @ np.ones(m)
                        W[k*n+i, :] = x
                
                # Check error
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                            err += (val - 1.0)**2
                
                if err < 1e-10:
                    found = True
                    break
            
            if found:
                # Round to nearest integer or half-integer?
                # The problem allows integer/half-integer.
                # We can try to round U, V, W to nearest 0.5
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                # Verify rounded solution
                err_r = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U_rounded[i*m+j, :], V_rounded[j*p+k, :] * W_rounded[k*n+i, :])
                            err_r += (val - 1.0)**2
                
                if err_r < 1e-10:
                    best_rank = r
                    best_U = U_rounded
                    best_V = V_rounded
                    best_W = W_rounded
                    break
        
        if found:
            break
    
    if best_U is None:
        best_U = U_school
        best_V = V_school
        best_W = W_school
        best_rank = n * m * p
    
    # Convert to lists as per interface? 
    # The interface says: factor matrices of shapes ... with integer/half-integer entries.
    # The return type is a dict with 'U', 'V', 'W'.
    # The current program returns lists of lists. I will return numpy arrays or lists.
    # The seed program returns lists. I will return lists to be safe.
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
