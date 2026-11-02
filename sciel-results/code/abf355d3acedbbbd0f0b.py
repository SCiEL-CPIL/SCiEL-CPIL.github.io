import numpy as np
import time

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_school = n * m * p  # 120
    
    # Deadline guard - keep 20s margin from 150s limit
    start_time = time.time()
    deadline = 130  # 20s safety margin
    
    # Initialize best result as schoolbook
    def make_schoolbook():
        U = [[0] * rank_school for _ in range(n * m)]
        V = [[0] * rank_school for _ in range(m * p)]
        W = [[0] * rank_school for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W
    
    best_U, best_V, best_W = make_schoolbook()
    best_rank = rank_school
    
    # Target tensor T[i,j,k] = 1 for all i,j,k
    T = np.ones((n, m, p), dtype=np.float64)
    
    # We search for real-valued U, V, W (will be rational/half-integer if lucky)
    # such that sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = T[i,j,k] = 1
    
    # Try rank 119 first (known to be achievable)
    target_rank = 119
    
    # Strategy: Start from schoolbook, remove one term, and try to repair
    # by adjusting all 119 remaining terms slightly
    
    # Use coordinate descent with proper residual computation
    def try_rank(R, max_sweeps=50, max_restarts=3):
        nonlocal best_U, best_V, best_W, best_rank
        nonlocal deadline, start_time
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                return False
            
            # Initialize: use schoolbook terms 0..R-1 as starting point
            # Then randomly perturb to escape local minima
            U = np.zeros((n * m, R), dtype=np.float64)
            V = np.zeros((m * p, R), dtype=np.float64)
            W = np.zeros((p * n, R), dtype=np.float64)
            
            # Copy first R schoolbook terms
            for r in range(R):
                i, j, k = r // (m*p), (r // p) % m, r % p
                U[i*m + j, r] = 1.0
                V[j*p + k, r] = 1.0
                W[k*n + i, r] = 1.0
            
            # Add small random perturbation to help escape
            if restart > 0:
                scale = 0.1 * restart
                U += np.random.normal(0, scale, U.shape)
                V += np.random.normal(0, scale, V.shape)
                W += np.random.normal(0, scale, W.shape)
            
            U_reshaped = U.reshape(n, m, R)
            V_reshaped = V.reshape(m, p, R)
            W_reshaped = W.reshape(p, n, R)
            
            for sweep in range(max_sweeps):
                if time.time() > deadline:
                    return False
                
                # Compute current tensor sum
                tensor_sum = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
                error = np.sum((tensor_sum - T) ** 2)
                
                if error < 1e-12:
                    # Found exact solution!
                    best_rank = R
                    best_U = U.tolist()
                    best_V = V.tolist()
                    best_W = W.tolist()
                    return True
                
                # Coordinate descent: update U, then V, then W
                # For each r, compute residual excluding r, then optimal value
                
                # Update U
                for r in range(R):
                    # Residual excluding r: T - sum_{r' != r} U_r' V_r' W_r'
                    # = T - tensor_sum + U_r V_r W_r
                    contrib_r = np.einsum('ij,jk,ki->ijk', 
                                          U_reshaped[:,:,r], 
                                          V_reshaped[:,:,r], 
                                          W_reshaped[:,:,r])
                    residual = T - tensor_sum + contrib_r
                    
                    # Optimal U_r[i,j] = sum_k residual[i,j,k] * V_r[j,k] * W_r[k,i] / sum_k V_r[j,k]^2 * W_r[k,i]^2
                    num = np.einsum('ijk,jk,ki->ij', residual, V_reshaped[:,:,r], W_reshaped[:,:,r])
                    den = np.einsum('jk,jk,ki,ki->ij', V_reshaped[:,:,r], V_reshaped[:,:,r], 
                                    W_reshaped[:,:,r], W_reshaped[:,:,r])
                    
                    # Avoid division by zero
                    mask = den > 1e-15
                    new_U_r = np.zeros((n, m), dtype=np.float64)
                    new_U_r[mask] = num[mask] / den[mask]
                    U_reshaped[:,:,r] = new_U_r
                
                # Update V
                for r in range(R):
                    contrib_r = np.einsum('ij,jk,ki->ijk', 
                                          U_reshaped[:,:,r], 
                                          V_reshaped[:,:,r], 
                                          W_reshaped[:,:,r])
                    residual = T - tensor_sum + contrib_r
                    
                    # Optimal V_r[j,k] = sum_i residual[i,j,k] * U_r[i,j] * W_r[k,i] / sum_i U_r[i,j]^2 * W_r[k,i]^2
                    num = np.einsum('ijk,ij,ki->jk', residual, U_reshaped[:,:,r], W_reshaped[:,:,r])
                    den = np.einsum('ij,ij,ki,ki->jk', U_reshaped[:,:,r], U_reshaped[:,:,r], 
                                    W_reshaped[:,:,r], W_reshaped[:,:,r])
                    
                    mask = den > 1e-15
                    new_V_r = np.zeros((m, p), dtype=np.float64)
                    new_V_r[mask] = num[mask] / den[mask]
                    V_reshaped[:,:,r] = new_V_r
                
                # Update W
                for r in range(R):
                    contrib_r = np.einsum('ij,jk,ki->ijk', 
                                          U_reshaped[:,:,r], 
                                          V_reshaped[:,:,r], 
                                          W_reshaped[:,:,r])
                    residual = T - tensor_sum + contrib_r
                    
                    # Optimal W_r[k,i] = sum_j residual[i,j,k] * U_r[i,j] * V_r[j,k] / sum_j U_r[i,j]^2 * V_r[j,k]^2
                    num = np.einsum('ijk,ij,jk->ki', residual, U_reshaped[:,:,r], V_reshaped[:,:,r])
                    den = np.einsum('ij,ij,jk,jk->ki', U_reshaped[:,:,r], U_reshaped[:,:,r], 
                                    V_reshaped[:,:,r], V_reshaped[:,:,r])
                    
                    mask = den > 1e-15
                    new_W_r = np.zeros((p, n), dtype=np.float64)
                    new_W_r[mask] = num[mask] / den[mask]
                    W_reshaped[:,:,r] = new_W_r
                
                # Recompute tensor_sum for next iteration
                tensor_sum = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
                error = np.sum((tensor_sum - T) ** 2)
                
                if error < 1e-12:
                    best_rank = R
                    best_U = U.reshape(n*m, R).tolist()
                    best_V = V.reshape(m*p, R).tolist()
                    best_W = W.reshape(p*n, R).tolist()
                    return True
                
                # Early stop if error isn't decreasing
                if sweep > 10 and error > 100:
                    break
            
            # If we get close but not exact, try to snap to rationals
            tensor_sum = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
            error = np.sum((tensor_sum - T) ** 2)
            if error < 1e-6:
                # Try rounding to half-integers
                U_half = np.round(U * 2) / 2
                V_half = np.round(V * 2) / 2
                W_half = np.round(W * 2) / 2
                U_h = U_half.reshape(n, m, R)
                V_h = V_half.reshape(m, p, R)
                W_h = W_half.reshape(p, n, R)
                tensor_h = np.einsum('ijr,jkr,kir->ijk', U_h, V_h, W_h)
                error_h = np.sum((tensor_h - T) ** 2)
                if error_h < 1e-9:
                    best_rank = R
                    best_U = U_half.tolist()
                    best_V = V_half.tolist()
                    best_W = W_half.tolist()
                    return True
        
        return False
    
    # Try rank 119
    if try_rank(target_rank, max_sweeps=30, max_restarts=2):
        pass
    elif time.time() < deadline - 10:
        # Try rank 118 as fallback (less likely to succeed)
        try_rank(118, max_sweeps=20, max_restarts=2)
    
    # Final check: if we have a valid lower-rank solution, return it
    # Otherwise return schoolbook
    if best_rank < rank_school:
        return {"U": best_U, "V": best_V, "W": best_W}
    else:
        return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
