import numpy as np
import time
import itertools

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    # Target tensor T[i, j, k] = 1
    # We want U (nm x R), V (mp x R), W (pn x R)
    # such that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    
    # Try ranks from 18 to 24
    for R in range(18, 25):
        # Set a time limit for each rank, or just global
        # Let's do a global time limit of 140s
        start_time = time.time()
        
        # Number of restarts
        # If R is small, we might need more restarts
        max_restarts = 5000 if R < 20 else 2000
        
        for restart in range(max_restarts):
            if time.time() - start_time > 140:
                break
            
            # Initialize V and W randomly
            # Using small integers or half-integers might help snapping
            # Let's use random floats in [-1, 1]
            V = np.random.uniform(-1, 1, (m * p, R))
            W = np.random.uniform(-1, 1, (p * n, R))
            
            # ALS Iterations
            for _ in range(10): # 10 iterations of ALS
                # Solve for U
                # For each i, j, solve A_ij U_ij = 1
                # A_ij is p x R
                U = np.zeros((n * m, R))
                for i in range(n):
                    for j in range(m):
                        # Construct A_ij
                        # A_ij[k, r] = V[j*p+k, r] * W[k*n+i, r]
                        A_ij = np.zeros((p, R))
                        for k in range(p):
                            A_ij[k, :] = V[j*p+k, :] * W[k*n+i, :]
                        # Solve A_ij @ U_ij = ones(p)
                        # Least squares
                        U_ij, residuals, rank, s = np.linalg.lstsq(A_ij, np.ones(p), rcond=None)
                        U[i*m+j, :] = U_ij
                        
                # Solve for V
                # For each j, k, solve A_jk V_jk = 1
                # A_jk is n x R
                V = np.zeros((m * p, R))
                for j in range(m):
                    for k in range(p):
                        A_jk = np.zeros((n, R))
                        for i in range(n):
                            A_jk[i, :] = U[i*m+j, :] * W[k*n+i, :]
                        V_jk, residuals, rank, s = np.linalg.lstsq(A_jk, np.ones(n), rcond=None)
                        V[j*p+k, :] = V_jk
                        
                # Solve for W
                # For each k, i, solve A_ki W_ki = 1
                # A_ki is m x R
                W = np.zeros((p * n, R))
                for k in range(p):
                    for i in range(n):
                        A_ki = np.zeros((m, R))
                        for j in range(m):
                            A_ki[j, :] = U[i*m+j, :] * V[j*p+k, :]
                        W_ki, residuals, rank, s = np.linalg.lstsq(A_ki, np.ones(m), rcond=None)
                        W[k*n+i, :] = W_ki
            
            # Snap to half-integers
            U_snapped = np.round(U * 2) / 2
            V_snapped = np.round(V * 2) / 2
            W_snapped = np.round(W * 2) / 2
            
            # Check if snapped solution is exact
            is_exact = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U_snapped[i*m+j, :], V_snapped[j*p+k, :] * W_snapped[k*n+i, :])
                        if abs(val - 1.0) > 1e-9:
                            is_exact = False
                            break
                    if not is_exact:
                        break
                if not is_exact:
                    break
            
            if is_exact:
                # Convert to lists of lists for return
                U_list = U_snapped.tolist()
                V_list = V_snapped.tolist()
                W_list = W_snapped.tolist()
                return {"U": U_list, "V": V_list, "W": W_list}
        
        # If time is up, break outer loop
        if time.time() - start_time > 140:
            break

    # Fallback: Schoolbook decomposition
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
