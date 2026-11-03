"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    # Tensor dimensions
    dim_u = n * m  # 12
    dim_v = m * p  # 20
    dim_w = p * n  # 15
    
    # Build the target tensor T[(i,j),(j,k),(k,i)] = 1
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # Try ranks from low to high
    # Lower bound: max(dim_u, dim_v, dim_w) = 20
    # Schoolbook: n*m*p = 60
    
    best_result = None
    best_rank = None
    
    # Schoolbook as fallback
    rank_sb = n * m * p
    U_sb = np.zeros((dim_u, rank_sb))
    V_sb = np.zeros((dim_v, rank_sb))
    W_sb = np.zeros((dim_w, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    best_result = (U_sb, V_sb, W_sb)
    best_rank = rank_sb
    
    # Try to find lower rank decompositions
    # Use a randomized approach with better initialization and optimization
    
    for rank in range(20, rank_sb + 1):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations with different strategies
        for attempt in range(5):
            if time.time() > deadline:
                break
            
            # Different initialization strategies
            if attempt == 0:
                # Standard random
                np.random.seed(rank * 100 + attempt)
                U_init = np.random.randn(dim_u, rank)
                V_init = np.random.randn(dim_v, rank)
                W_init = np.random.randn(dim_w, rank)
            elif attempt == 1:
                # Sparse initialization
                np.random.seed(rank * 100 + attempt + 1)
                U_init = np.random.randn(dim_u, rank)
                V_init = np.random.randn(dim_v, rank)
                W_init = np.random.randn(dim_w, rank)
                # Make sparse
                U_init[np.random.rand(*U_init.shape) < 0.5] = 0
                V_init[np.random.rand(*V_init.shape) < 0.5] = 0
                W_init[np.random.rand(*W_init.shape) < 0.5] = 0
            elif attempt == 2:
                # Block diagonal-like structure
                np.random.seed(rank * 100 + attempt + 2)
                U_init = np.random.randn(dim_u, rank)
                V_init = np.random.randn(dim_v, rank)
                W_init = np.random.randn(dim_w, rank)
                # Scale
                U_init /= np.sqrt(rank)
                V_init /= np.sqrt(rank)
                W_init /= np.sqrt(rank)
            else:
                # Very sparse
                np.random.seed(rank * 100 + attempt + 3)
                U_init = np.zeros((dim_u, rank))
                V_init = np.zeros((dim_v, rank))
                W_init = np.zeros((dim_w, rank))
                for r in range(rank):
                    # Random sparse vectors
                    u_support = np.random.choice(dim_u, size=max(1, dim_u // 4), replace=False)
                    v_support = np.random.choice(dim_v, size=max(1, dim_v // 4), replace=False)
                    w_support = np.random.choice(dim_w, size=max(1, dim_w // 4), replace=False)
                    U_init[u_support, r] = np.random.choice([-1, 1], size=len(u_support))
                    V_init[v_support, r] = np.random.choice([-1, 1], size=len(v_support))
                    W_init[w_support, r] = np.random.choice([-1, 1], size=len(w_support))
            
            # Flatten
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            # Define the objective more efficiently
            def obj_fast(x):
                U = x[:dim_u * rank].reshape(dim_u, rank)
                V = x[dim_u * rank:dim_u * rank + dim_v * rank].reshape(dim_v, rank)
                W = x[dim_u * rank + dim_v * rank:].reshape(dim_w, rank)
                
                # For each (i,j,k), compute sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
                # and compare to 1
                total_err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            idx_u = i * m + j
                            idx_v = j * p + k
                            idx_w = k * n + i
                            s = np.dot(U[idx_u, :], V[idx_v, :] * W[idx_w, :])
                            total_err += (s - 1.0) ** 2
                return total_err
            
            # Use Nelder-Mead for small ranks
            if rank <= 35:
                result = minimize(obj_fast, x0, method='Nelder-Mead', 
                                  options={'maxiter': 800, 'xatol': 1e-10, 'fatol': 1e-10})
                if result.fun < 1e-8:
                    U_opt = result.x[:dim_u * rank].reshape(dim_u, rank)
                    V_opt = result.x[dim_u * rank:dim_u * rank + dim_v * rank].reshape(dim_v, rank)
                    W_opt = result.x[dim_u * rank + dim_v * rank:].reshape(dim_w, rank)
                    
                    # Round to half-integers
                    U_rounded = np.round(U_opt * 2) / 2
                    V_rounded = np.round(V_opt * 2) / 2
                    W_rounded = np.round(W_opt * 2) / 2
                    
                    # Verify
                    max_err = 0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                idx_u = i * m + j
                                idx_v = j * p + k
                                idx_w = k * n + i
                                s = np.dot(U_rounded[idx_u, :], V_rounded[idx_v, :] * W_rounded[idx_w, :])
                                max_err = max(max_err, abs(s - 1.0))
                    
                    if max_err < 1e-9:
                        best_result = (U_rounded, V_rounded, W_rounded)
                        best_rank = rank
                        break  # Found a valid decomposition at this rank
        
        if best_rank == rank:
            break  # Found the best possible for this rank
    
    # Convert to lists
    U_list = best_result[0].tolist()
    V_list = best_result[1].tolist()
    W_list = best_result[2].tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
