import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Lower bound for rank is max(n*m, m*p, p*n)
    min_rank = max(n * m, m * p, p * n)
    
    # We try ranks starting from the lower bound upwards
    # Given time constraints, we try a few small ranks
    max_attempts = 10
    base_rank = min_rank
    
    for rank_offset in range(0, max_attempts):
        R = base_rank + rank_offset
        
        # Initialize with random values
        np.random.seed(42 + rank_offset) # Reproducibility
        U = np.random.randn(n * m, R)
        V = np.random.randn(m * p, R)
        W = np.random.randn(p * n, R)
        
        best_error = float('inf')
        best_U, best_V, best_W = None, None, None
        
        for it in range(100): # 100 iterations of ALS
            # Solve for U
            # For each (i,j), we need sum_r U[i,j,r] * (V[:, j, :] * W[:, i, :])_r = 1
            # This is a linear system for U[i,j, :]
            # Let C_ij be a matrix of shape (p, R) where C_ij[k, r] = V[j*p+k, r] * W[k*n+i, r]
            # We want C_ij @ U[i*m+j, :] = 1 (vector of ones of length p)
            
            for i in range(n):
                for j in range(m):
                    # Construct C_ij
                    # V_slice: rows j*p to j*p+p-1
                    V_slice = V[j * p:(j + 1) * p, :] # shape p x R
                    # W_indices: k*n + i for k in 0..p-1
                    W_idx = np.array([k * n + i for k in range(p)])
                    W_slice = W[W_idx, :] # shape p x R
                    
                    C_ij = V_slice * W_slice # element-wise product, shape p x R
                    
                    # Solve C_ij @ u = 1
                    # Using lstsq
                    try:
                        u, residuals, rank_c, s = np.linalg.lstsq(C_ij, np.ones(p), rcond=None)
                    except np.linalg.LinAlgError:
                        u = np.zeros(R)
                    
                    U[i * m + j, :] = u
            
            # Solve for V
            # For each (j,k), sum_r V[j,k,r] * (U[:, j, :] * W[:, k, :])_r = 1
            # Let D_jk be a matrix of shape (n, R) where D_jk[i, r] = U[i*m+j, r] * W[k*n+i, r]
            # We want D_jk @ V[j*p+k, :] = 1 (vector of ones of length n)
            
            for j in range(m):
                for k in range(p):
                    # U_indices: i*m + j for i in 0..n-1
                    U_idx = np.array([i * m + j for i in range(n)])
                    U_slice = U[U_idx, :] # shape n x R
                    # W_indices: k*n + i for i in 0..n-1
                    W_idx = np.array([k * n + i for i in range(n)])
                    W_slice = W[W_idx, :] # shape n x R
                    
                    D_jk = U_slice * W_slice # element-wise product, shape n x R
                    
                    try:
                        v, residuals, rank_d, s = np.linalg.lstsq(D_jk, np.ones(n), rcond=None)
                    except np.linalg.LinAlgError:
                        v = np.zeros(R)
                    
                    V[j * p + k, :] = v
            
            # Solve for W
            # For each (k,i), sum_r W[k,i,r] * (U[:, i, :] * V[:, k, :])_r = 1
            # Let E_ki be a matrix of shape (m, R) where E_ki[j, r] = U[i*m+j, r] * V[j*p+k, r]
            # We want E_ki @ W[k*n+i, :] = 1 (vector of ones of length m)
            
            for k in range(p):
                for i in range(n):
                    # U_indices: i*m + j for j in 0..m-1
                    U_idx = np.array([i * m + j for j in range(m)])
                    U_slice = U[U_idx, :] # shape m x R
                    # V_indices: j*p + k for j in 0..m-1
                    V_idx = np.array([j * p + k for j in range(m)])
                    V_slice = V[V_idx, :] # shape m x R
                    
                    E_ki = U_slice * V_slice # element-wise product, shape m x R
                    
                    try:
                        w, residuals, rank_e, s = np.linalg.lstsq(E_ki, np.ones(m), rcond=None)
                    except np.linalg.LinAlgError:
                        w = np.zeros(R)
                    
                    W[k * n + i, :] = w
            
            # Calculate error
            # Error = sum_{i,j,k} (sum_r U[i,j,r]V[j,k,r]W[k,i,r] - 1)^2
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                        err += (val - 1.0) ** 2
            
            if err < best_error:
                best_error = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
            if best_error < 1e-10:
                break

        if best_error < 1e-6:
            # Round to half-integers
            U_int = np.round(best_U * 2) / 2
            V_int = np.round(best_V * 2) / 2
            W_int = np.round(best_W * 2) / 2
            
            # Verify exactness
            is_exact = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U_int[i * m + j, :], V_int[j * p + k, :] * W_int[k * n + i, :])
                        if abs(val - 1.0) > 1e-9:
                            is_exact = False
                            break
                    if not is_exact:
                        break
                if not is_exact:
                    break
            
            if is_exact:
                # Convert to lists of lists
                U_list = U_int.tolist()
                V_list = V_int.tolist()
                W_list = W_int.tolist()
                return {"U": U_list, "V": V_list, "W": W_list}
        
        # If not exact, try next rank

    # Fallback to schoolbook if no low-rank solution found
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
# EVOLVE-BLOCK-END
