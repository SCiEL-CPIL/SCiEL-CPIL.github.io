import numpy as np
import time

def construct():
    n, m, p = 4, 4, 5
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    # Precompute index mappings
    # For each (i,j,k), the constraint is:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    best_rank = n * m * p  # Start with schoolbook
    best_solution = None
    
    # Try different ranks
    for target_rank in range(60, n*m*p + 1):
        if time.time() > deadline:
            break
            
        # Try to find a decomposition with this rank
        # Use alternating optimization
        for trial in range(100):
            if time.time() > deadline:
                break
                
            # Initialize randomly
            U = np.random.randn(n*m, target_rank)
            V = np.random.randn(m*p, target_rank)
            W = np.random.randn(p*n, target_rank)
            
            # Alternating optimization
            for it in range(50):
                if time.time() > deadline:
                    break
                    
                # Fix V and W, solve for U
                # For each (i,j): U[i*m+j, :] must satisfy
                # sum_k' ... wait, for each k: U[i*m+j, :] · (V[j*p+k, :] * W[k*n+i, :]) = 1
                
                U_new = np.zeros((n*m, target_rank))
                for i in range(n):
                    for j in range(m):
                        # Build the linear system for U[i*m+j, :]
                        # For each k in range(p):
                        #   U[i*m+j, :] · (V[j*p+k, :] * W[k*n+i, :]) = 1
                        A_mat = np.zeros((p, target_rank))
                        for k in range(p):
                            A_mat[k, :] = V[j*p+k, :] * W[k*n+i, :]
                        # Solve A_mat @ x = ones(p)
                        # Least squares
                        x, _, _, _ = np.linalg.lstsq(A_mat, np.ones(p), rcond=None)
                        U_new[i*m+j, :] = x
                
                # Fix U and W, solve for V
                V_new = np.zeros((m*p, target_rank))
                for j in range(m):
                    for k in range(p):
                        # For each i in range(n):
                        #   V[j*p+k, :] · (U[i*m+j, :] * W[k*n+i, :]) = 1
                        A_mat = np.zeros((n, target_rank))
                        for i in range(n):
                            A_mat[i, :] = U[i*m+j, :] * W[k*n+i, :]
                        x, _, _, _ = np.linalg.lstsq(A_mat, np.ones(n), rcond=None)
                        V_new[j*p+k, :] = x
                
                # Fix U and V, solve for W
                W_new = np.zeros((p*n, target_rank))
                for k in range(p):
                    for i in range(n):
                        # For each j in range(m):
                        #   W[k*n+i, :] · (U[i*m+j, :] * V[j*p+k, :]) = 1
                        A_mat = np.zeros((m, target_rank))
                        for j in range(m):
                            A_mat[j, :] = U[i*m+j, :] * V[j*p+k, :]
                        x, _, _, _ = np.linalg.lstsq(A_mat, np.ones(m), rcond=None)
                        W_new[k*n+i, :] = x
                
                U, V, W = U_new, V_new, W_new
            
            # Check if the solution is exact
            max_err = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                        max_err = max(max_err, abs(val - 1))
            
            if max_err < 1e-6:
                # Found a valid solution
                best_rank = target_rank
                # Convert to list of lists with rounded values
                U_list = [[float(round(x, 10)) for x in U[i]] for i in range(n*m)]
                V_list = [[float(round(x, 10)) for x in V[i]] for i in range(m*p)]
                W_list = [[float(round(x, 10)) for x in W[i]] for i in range(p*n)]
                best_solution = {"U": U_list, "V": V_list, "W": W_list}
                break
        
        if best_solution is not None:
            break
    
    if best_solution is None:
        # Fall back to schoolbook
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
        best_solution = {"U": U, "V": V, "W": W}
    
    return best_solution
