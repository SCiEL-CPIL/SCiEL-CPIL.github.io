import numpy as np
import time

DIMENSIONS = [2, 3, 10]

def construct():
    start_time = time.time()
    deadline = 140.0  # Leave 10s safety margin
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Target tensor T[i, j, k] is 1 for all valid i, j, k
    # T_flat index: (i*m+j, j*p+k, k*n+i)
    
    # Fallback: Schoolbook decomposition (Rank = n * m * p)
    def schoolbook():
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
    
    U_s, V_s, W_s, rank_s = schoolbook()
    best_result = (U_s, V_s, W_s, rank_s)
    
    # Function to verify a decomposition
    def verify(U, V, W, r):
        # Reconstruct T
        # T[im+j, jp+k, kn+i] = sum_c U[im+j, c] V[jp+k, c] W[kn+i, c]
        # We can compute this efficiently
        # For each c, outer product of U[:, c] and V[:, c] gives a matrix of size nm x mp
        # Then multiply by W[:, c]? No.
        # T is a 3D array.
        # Let's just loop over i, j, k for simplicity, it's small (2*3*10=60 checks)
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m + j, :], V[j*p + k, :] * W[k*n + i, :])
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err

    # Search for lower rank
    # Try ranks from 1 upwards
    # If r < m, the system A_ki w = 1 is underdetermined.
    # We can still solve it with lstsq, but exact solution is unlikely.
    # We focus on r >= m = 3.
    
    for r in range(1, min(60, n*m*p + 1)):
        if time.time() > deadline - 10:
            break
            
        found = False
        # Number of random restarts
        # For small r, we need many restarts.
        max_restarts = 100 if r < 10 else 20
        
        for trial in range(max_restarts):
            if time.time() > deadline - 5:
                break
                
            # Initialize U and V
            # Use random values
            U = np.random.randn(nm, r)
            V = np.random.randn(mp, r)
            
            # Solve for W
            W = np.zeros((pn, r))
            success = True
            
            for k in range(p):
                for i in range(n):
                    # Construct A_ki of shape (m, r)
                    A_ki = np.zeros((m, r))
                    for j in range(m):
                        A_ki[j, :] = U[i*m + j, :] * V[j*p + k, :]
                    
                    # Solve A_ki @ w = ones(m)
                    try:
                        w, residuals, rank_A, sv = np.linalg.lstsq(A_ki, np.ones(m), rcond=None)
                        W[k*n + i, :] = w
                    except np.linalg.LinAlgError:
                        success = False
                        break
                if not success:
                    break
            
            if not success:
                continue
                
            # Verify
            err = verify(U, V, W, r)
            if err < 1e-9:
                best_result = (U, V, W, r)
                found = True
                break
        
        if found:
            break

    U, V, W, rank = best_result
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
