import numpy as np
import time

DIMENSIONS = [3, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # n=3 rows of A, m=4 inner, p=6 cols of B
    # U: (n*m) x rank, V: (m*p) x rank, W: (p*n) x rank
    # M[idx1, idx2, idx3] = 1 if valid, else 0
    # idx1 = i*m+j, idx2 = j*p+k, idx3 = k*n+i
    
    # Build the multiplication tensor M as a 3D array (n*m, m*p, p*n)
    M = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx1 = i*m + j
                idx2 = j*p + k
                idx3 = k*n + i
                M[idx1, idx2, idx3] = 1.0
    
    # Precompute flattened matrices for ALS
    M_flat = M.reshape(n*m, m*p * p*n)  # (n*m, m*p*p*n)
    M_T_flat = M.T.reshape(m*p, n*m*p*n)  # (m*p, n*m*p*n)
    M_P_flat = M.reshape(n*m*m*p, p*n).T  # (p*n, n*m*m*p)
    
    # Try ranks from lower bound upward
    # Lower bound: max(n*p, m*n, m*p) = max(18,12,24) = 24
    # Known results suggest rank 24 may be achievable for 3x4x6
    
    best_result = None
    best_rank = None
    
    # Number of restarts per rank - adapt based on time
    max_restarts_per_rank = 8
    
    for rank in range(24, 73):
        if time.time() > deadline - 20:
            break
        
        found = False
        
        for restart in range(max_restarts_per_rank):
            if time.time() > deadline - 15:
                break
            
            # Use different seeds for each restart
            seed = rank * 1000 + restart
            np.random.seed(seed)
            
            # Initialize with random values
            U = np.random.randn(n*m, rank)
            V = np.random.randn(m*p, rank)
            W = np.random.randn(p*n, rank)
            
            # ALS iterations
            converged = False
            prev_residual = np.inf
            
            for iteration in range(150):
                if time.time() > deadline - 10:
                    break
                
                # Update U: fix V, W
                T = (V[:, np.newaxis, :] * W[np.newaxis, :, :])  # (m*p, p*n, rank)
                T_flat = T.reshape(m*p*p*n, rank)  # (m*p*p*n, rank)
                
                try:
                    U_new, _, _, _ = np.linalg.lstsq(T_flat, M_flat.T, rcond=None)
                    U_new = U_new.T  # (n*m, rank)
                except:
                    break
                
                # Update V: fix U, W
                T2 = (U[:, np.newaxis, :] * W[np.newaxis, :, :])  # (n*m, p*n, rank)
                T2_flat = T2.reshape(n*m*p*n, rank)
                
                try:
                    V_new, _, _, _ = np.linalg.lstsq(T2_flat, M_T_flat.T, rcond=None)
                    V_new = V_new.T  # (m*p, rank)
                except:
                    break
                
                # Update W: fix U, V
                T3 = (U[:, np.newaxis, :] * V[np.newaxis, :, :])  # (n*m, m*p, rank)
                T3_flat = T3.reshape(n*m*m*p, rank)
                
                try:
                    W_new, _, _, _ = np.linalg.lstsq(T3_flat, M_P_flat.T, rcond=None)
                    W_new = W_new.T  # (p*n, rank)
                except:
                    break
                
                U, V, W = U_new, V_new, W_new
                
                # Check convergence - compute residual
                T_check = (U[:, np.newaxis, np.newaxis, :] * V[np.newaxis, :, np.newaxis, :] * W[np.newaxis, np.newaxis, :, :])
                residual = np.abs(np.sum(T_check, axis=3) - M).max()
                
                if residual < 1e-10:
                    converged = True
                    break
                
                # Early stop if not improving
                if residual > prev_residual * 0.999:
                    if iteration > 20:
                        break
                prev_residual = residual
            
            if converged:
                # Final polish with tighter tolerance
                for polish_iter in range(50):
                    if time.time() > deadline - 5:
                        break
                    
                    T = (V[:, np.newaxis, :] * W[np.newaxis, :, :])
                    T_flat = T.reshape(m*p*p*n, rank)
                    try:
                        U_new, _, _, _ = np.linalg.lstsq(T_flat, M_flat.T, rcond=None)
                        U_new = U_new.T
                    except:
                        break
                    
                    T2 = (U[:, np.newaxis, :] * W[np.newaxis, :, :])
                    T2_flat = T2.reshape(n*m*p*n, rank)
                    try:
                        V_new, _, _, _ = np.linalg.lstsq(T2_flat, M_T_flat.T, rcond=None)
                        V_new = V_new.T
                    except:
                        break
                    
                    T3 = (U[:, np.newaxis, :] * V[np.newaxis, :, :])
                    T3_flat = T3.reshape(n*m*m*p, rank)
                    try:
                        W_new, _, _, _ = np.linalg.lstsq(T3_flat, M_P_flat.T, rcond=None)
                        W_new = W_new.T
                    except:
                        break
                    
                    U, V, W = U_new, V_new, W_new
                    
                    T_check = (U[:, np.newaxis, np.newaxis, :] * V[np.newaxis, :, np.newaxis, :] * W[np.newaxis, np.newaxis, :, :])
                    residual = np.abs(np.sum(T_check, axis=3) - M).max()
                    
                    if residual < 1e-12:
                        break
                
                # Verify final result
                T_check = (U[:, np.newaxis, np.newaxis, :] * V[np.newaxis, :, np.newaxis, :] * W[np.newaxis, np.newaxis, :, :])
                final_residual = np.abs(np.sum(T_check, axis=3) - M).max()
                
                if final_residual < 1e-10:
                    best_rank = rank
                    U_list = [[float(x) for x in row] for row in U]
                    V_list = [[float(x) for x in row] for row in V]
                    W_list = [[float(x) for x in row] for row in W]
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    found = True
                    break
        
        if found:
            break
    
    if best_result is not None:
        return best_result
    else:
        # Fall back to schoolbook
        rank = n * m * p
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1.0
                    V[j*p+k][col] = 1.0
                    W[k*n+i][col] = 1.0
                    col += 1
        return {"U": U, "V": V, "W": W}
