import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # Safety margin

    # Target tensor: T[i,j,k] = 1
    # We want sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1
    
    # Try ranks in increasing order of likelihood (higher ranks easier to find)
    # Current best is 42. Let's try 40, 36, 32.
    target_ranks = [40, 38, 36, 34, 32]
    
    best_result = None
    best_rank = 48 # Fallback
    
    # Precompute indices for vectorization
    # U: (n*m, R) -> flat
    # V: (m*p, R) -> flat
    # W: (p*n, R) -> flat
    
    def objective(params, R):
        # Reshape params
        size_U = n * m * R
        size_V = m * p * R
        size_W = p * n * R
        
        U_flat = params[:size_U]
        V_flat = params[size_U:size_U + size_V]
        W_flat = params[size_U + size_V:size_U + size_V + size_W]
        
        U = U_flat.reshape((n * m, R))
        V = V_flat.reshape((m * p, R))
        W = W_flat.reshape((p * n, R))
        
        # Compute tensor
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # We can compute this efficiently using einsum or loops
        # Given small dimensions, loops might be okay, but einsum is better.
        
        # Create index arrays
        # This is a bit slow if done inside the objective.
        # Better to precompute the mapping.
        
        # Actually, let's just compute the residual.
        # Res[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] - 1
        
        # For speed, we can use a loop over r?
        # No, vectorized is better.
        
        # Let's use einsum:
        # U: (n, m, R)
        # V: (m, p, R)
        # W: (p, n, R)
        
        U_3d = U.reshape((n, m, R))
        V_3d = V.reshape((m, p, R))
        W_3d = W.reshape((p, n, R))
        
        # T = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
        # einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
        
        T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d, optimize=True)
        
        # Residual
        Res = T - 1.0
        return np.sum(Res * Res)

    def try_rank(R, num_restarts=5):
        for restart in range(num_restarts):
            if time.time() > deadline:
                return None
            
            # Random initialization
            # Scale initialization to help convergence
            # U, V, W should be roughly O(1)
            
            size_U = n * m * R
            size_V = m * p * R
            size_W = p * n * R
            total_size = size_U + size_V + size_W
            
            # Heuristic initialization
            # We can start with the trivial decomposition if R >= n*m*p, but here R < 48.
            # So random init is the way.
            
            np.random.seed(42 + restart)
            params = np.random.randn(total_size) * 0.5
            
            # Optimize
            try:
                res = minimize(
                    objective, 
                    params, 
                    method='L-BFGS-B', 
                    options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
                )
                
                # Check if converged
                if res.success or res.fun < 1e-8:
                    # Extract U, V, W
                    U_flat = res.x[:size_U]
                    V_flat = res.x[size_U:size_U + size_V]
                    W_flat = res.x[size_U + size_V:size_U + size_V + size_W]
                    
                    U = U_flat.reshape((n * m, R))
                    V = V_flat.reshape((m * p, R))
                    W = W_flat.reshape((p * n, R))
                    
                    # Verify
                    U_3d = U.reshape((n, m, R))
                    V_3d = V.reshape((m, p, R))
                    W_3d = W.reshape((p, n, R))
                    T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d, optimize=True)
                    if np.max(np.abs(T - 1.0)) < 1e-6:
                        return (U, V, W)
                        
            except Exception as e:
                pass
                
        return None

    # Try ranks
    for R in target_ranks:
        if time.time() > deadline:
            break
            
        result = try_rank(R)
        if result is not None:
            U, V, W = result
            best_result = (U, V, W)
            best_rank = R
            break # Found a lower rank, stop
            
    # If no lower rank found, return the known rank 42 solution from the seed
    # Or just return the best found. If none found, we must return a valid one.
    # The seed program has a valid rank 42 solution. We should include that as a fallback.
    
    if best_result is None:
        # Fallback to Rank 42 Strassen
        rank_s = 42
        U = np.zeros((n * m, rank_s), dtype=float)
        V = np.zeros((m * p, rank_s), dtype=float)
        W = np.zeros((p * n, rank_s), dtype=float)
        
        def add_strassen_block(i_start, j_start, k_start, offset):
            col = offset
            U[(i_start+0)*m + (j_start+0)][col] += 1
            U[(i_start+1)*m + (j_start+1)][col] += 1
            V[(j_start+0)*p + (k_start+0)][col] += 1
            V[(j_start+1)*p + (k_start+1)][col] += 1
            W[(k_start+0)*n + (i_start+0)][col] += 1
            W[(k_start+1)*n + (i_start+1)][col] += 1

            col = offset + 1
            U[(i_start+1)*m + (j_start+0)][col] += 1
            U[(i_start+1)*m + (j_start+1)][col] += 1
            V[(j_start+0)*p + (k_start+0)][col] += 1
            W[(k_start+0)*n + (i_start+1)][col] += 1
            W[(k_start+1)*n + (i_start+1)][col] -= 1

            col = offset + 2
            U[(i_start+0)*m + (j_start+0)][col] += 1
            V[(j_start+0)*p + (k_start+1)][col] += 1
            V[(j_start+1)*p + (k_start+1)][col] -= 1
            W[(k_start+1)*n + (i_start+0)][col] += 1
            W[(k_start+1)*n + (i_start+1)][col] += 1

            col = offset + 3
            U[(i_start+1)*m + (j_start+1)][col] += 1
            V[(j_start+1)*p + (k_start+0)][col] += 1
            V[(j_start+0)*p + (k_start+0)][col] -= 1
            W[(k_start+0)*n + (i_start+0)][col] += 1
            W[(k_start+0)*n + (i_start+1)][col] += 1

            col = offset + 4
            U[(i_start+0)*m + (j_start+0)][col] += 1
            U[(i_start+0)*m + (j_start+1)][col] += 1
            V[(j_start+1)*p + (k_start+1)][col] += 1
            W[(k_start+0)*n + (i_start+0)][col] -= 1
            W[(k_start+1)*n + (i_start+0)][col] += 1

            col = offset + 5
            U[(i_start+1)*m + (j_start+0)][col] += 1
            U[(i_start+0)*m + (j_start+0)][col] -= 1
            V[(j_start+0)*p + (k_start+0)][col] += 1
            V[(j_start+0)*p + (k_start+1)][col] += 1
            W[(k_start+1)*n + (i_start+1)][col] += 1

            col = offset + 6
            U[(i_start+0)*m + (j_start+1)][col] += 1
            U[(i_start+1)*m + (j_start+1)][col] -= 1
            V[(j_start+1)*p + (k_start+0)][col] += 1
            V[(j_start+1)*p + (k_start+1)][col] += 1
            W[(k_start+0)*n + (i_start+0)][col] += 1

        add_strassen_block(0, 0, 0, 0)
        add_strassen_block(0, 0, 2, 7)
        add_strassen_block(0, 0, 4, 14)
        add_strassen_block(0, 2, 0, 21)
        add_strassen_block(0, 2, 2, 28)
        add_strassen_block(0, 2, 4, 35)
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        U, V, W = best_result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
