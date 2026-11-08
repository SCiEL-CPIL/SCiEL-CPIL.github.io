"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    # Schoolbook decomposition as baseline
    rank_school = n * m * p
    U = np.zeros((n * m, rank_school))
    V = np.zeros((m * p, rank_school))
    W = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_school
    best_U, best_V, best_W = U, V, W
    
    # Try to find lower rank decompositions
    # The tensor T[i,j,k] = 1 for all i,j,k in the multiplication tensor
    # We need: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Build the target tensor
    T = np.ones((n, m, p))
    
    # Try ranks from lower to higher
    for target_rank in range(1, rank_school + 1):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        max_attempts = 20
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Random initialization
            U_r = np.random.randn(n * m, target_rank)
            V_r = np.random.randn(m * p, target_rank)
            W_r = np.random.randn(p * n, target_rank)
            
            # Define the objective: minimize ||sum_r U[:,:,r] * V[:,:,r] * W[:,:,r] - T||^2
            def objective(params):
                U_flat, V_flat, W_flat = params
                U_mat = U_flat.reshape(n * m, target_rank)
                V_mat = V_flat.reshape(m * p, target_rank)
                W_mat = W_flat.reshape(p * n, target_rank)
                
                # Compute the reconstructed tensor
                # T_recon[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                T_recon = np.zeros((n, m, p))
                for r in range(target_rank):
                    U_col = U_mat[:, r]  # shape (n*m,)
                    V_col = V_mat[:, r]  # shape (m*p,)
                    W_col = W_mat[:, r]  # shape (p*n,)
                    
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                T_recon[i, j, k] += U_col[i * m + j] * V_col[j * p + k] * W_col[k * n + i]
                
                error = np.sum((T_recon - T) ** 2)
                return error
            
            # Use Nelder-Mead or similar for small problems
            # This is expensive, so only try for small ranks
            if target_rank <= 20:
                x0 = np.concatenate([U_r.flatten(), V_r.flatten(), W_r.flatten()])
                try:
                    result = minimize(objective, x0, method='Nelder-Mead', 
                                     options={'maxiter': 100, 'xatol': 1e-10, 'fatol': 1e-10})
                    if result.fun < 1e-6:
                        # Found a valid decomposition
                        U_final = result.x[:n*m*target_rank].reshape(n * m, target_rank)
                        V_final = result.x[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape(m * p, target_rank)
                        W_final = result.x[n*m*target_rank + m*p*target_rank :].reshape(p * n, target_rank)
                        
                        # Verify
                        T_recon = np.zeros((n, m, p))
                        for r in range(target_rank):
                            for i in range(n):
                                for j in range(m):
                                    for k in range(p):
                                        T_recon[i, j, k] += U_final[i*m+j, r] * V_final[j*p+k, r] * W_final[k*n+i, r]
                        if np.allclose(T_recon, T, atol=1e-6):
                            best_rank = target_rank
                            best_U, best_V, best_W = U_final, V_final, W_final
                            return {
                                "U": best_U.tolist(),
                                "V": best_V.tolist(),
                                "W": best_W.tolist()
                            }
                except Exception:
                    pass
    
    # Return the best found
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
