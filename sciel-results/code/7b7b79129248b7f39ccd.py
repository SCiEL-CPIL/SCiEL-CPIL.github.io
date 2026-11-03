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
    # Use a randomized approach: try rank r, initialize randomly, optimize
    
    for rank in range(20, rank_sb + 1):
        if time.time() > deadline:
            break
        
        # Try a few random initializations
        for attempt in range(3):
            if time.time() > deadline:
                break
            
            # Random initialization
            np.random.seed(rank * 100 + attempt)
            U_init = np.random.randn(dim_u, rank)
            V_init = np.random.randn(dim_v, rank)
            W_init = np.random.randn(dim_w, rank)
            
            # Flatten
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            def objective(x):
                U = x[:dim_u * rank].reshape(dim_u, rank)
                V = x[dim_u * rank:dim_u * rank + dim_v * rank].reshape(dim_v, rank)
                W = x[dim_u * rank + dim_v * rank:].reshape(dim_w, rank)
                
                # Compute sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] for all i,j,k
                # This is a 3-way tensor contraction
                # For each r, the outer product U[:,r] ⊗ V[:,r] ⊗ W[:,r]
                # We need the sum to equal 1 at positions (i*m+j, j*p+k, k*n+i)
                
                error = 0.0
                for r in range(rank):
                    u_r = U[:, r]
                    v_r = V[:, r]
                    w_r = W[:, r]
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                idx_u = i * m + j
                                idx_v = j * p + k
                                idx_w = k * n + i
                                val = u_r[idx_u] * v_r[idx_v] * w_r[idx_w]
                                error += (val - (1.0 / rank if rank > 0 else 1)) ** 2
                return error
            
            # This is too slow for gradient-based optimization with rank > 20
            # Let me use a different approach: use the structure more directly
            
            # Alternative: use ALS (Alternating Least Squares) or a direct solve
            # For fixed V and W, solve for U, etc.
            
            # Let me use a simpler approach: random search with local optimization
            # Or use the fact that we can set up a system of equations
            
            # Actually, let me try a different strategy: 
            # For each rank-1 term, choose which (i,j,k) triples it covers
            # A rank-1 term with U having support S_U, V having support S_V, W having support S_W
            # contributes to (i,j,k) where (i,j) in S_U, (j,k) in S_V, (k,i) in S_W
            
            # This is getting complex. Let me use a numerical approach with scipy.
            
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
            
            # Use Nelder-Mead for small ranks, or just accept if we can get close
            # Given time constraints, let me just try a few ranks with simple methods
            
            # For rank = 20 (the lower bound), try to solve
            if rank <= 30:
                result = minimize(obj_fast, x0, method='Nelder-Mead', 
                                  options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8})
                if result.fun < 1e-6:
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
