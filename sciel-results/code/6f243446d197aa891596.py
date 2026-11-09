import numpy as np
import time

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 5x5 matrix multiplication.
    Returns {'U', 'V', 'W'} with shapes (25, rank), (25, rank), (25, rank).
    The decomposition satisfies: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k.
    """
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s kill
    
    n, m, p = DIMENSIONS
    
    def build_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, rank
    
    # Get schoolbook solution as fallback
    U_sb, V_sb, W_sb, rank_sb = build_schoolbook()
    
    def compute_residual_fast(U, V, W):
        """Faster residual computation using numpy."""
        R = U.shape[1]
        U_r = U.reshape(n, m, R)
        V_r = V.reshape(m, p, R)
        W_r = W.reshape(p, n, R)
        
        W_r_trans = np.transpose(W_r, (1, 0, 2))  # W_r_trans[i,k,:] = W_r[k,i,:]
        
        T = np.einsum('ijr,jkr,ikr->ijk', U_r, V_r, W_r_trans)
        
        diff = T - 1.0
        return np.sum(diff * diff)
    
    # Try to find a lower rank decomposition
    best_rank = rank_sb
    best_U = U_sb
    best_V = V_sb
    best_W = W_sb
    
    # Strategy: Try ranks from 124 down to 100
    # Starting closer to the schoolbook rank is more likely to succeed
    # because the initialization (truncated schoolbook) is closer to a valid solution.
    
    np.random.seed(42)
    
    for target_rank in range(rank_sb - 1, 99, -1):
        if time.time() - start_time > deadline - 5:
            break
            
        # Initialize with random perturbation of schoolbook-like structure
        # Use first target_rank columns of schoolbook as initialization
        U_init = U_sb[:, :target_rank].copy()
        V_init = V_sb[:, :target_rank].copy()
        W_init = W_sb[:, :target_rank].copy()
        
        # Add small random perturbation
        noise_scale = 0.1
        U_init += np.random.normal(0, noise_scale, U_init.shape)
        V_init += np.random.normal(0, noise_scale, V_init.shape)
        W_init += np.random.normal(0, noise_scale, W_init.shape)
        
        # Gradient descent
        U_curr = U_init.copy()
        V_curr = V_init.copy()
        W_curr = W_init.copy()
        
        learning_rate = 0.01
        max_iterations = 200
        convergence_threshold = 1e-12
        
        for it in range(max_iterations):
            if time.time() - start_time > deadline - 2:
                break
                
            # Compute residual
            res = compute_residual_fast(U_curr, V_curr, W_curr)
            
            if res < convergence_threshold:
                # Found a valid solution!
                best_rank = target_rank
                best_U = U_curr
                best_V = V_curr
                best_W = W_curr
                break
                
            # Compute gradients
            U_r = U_curr.reshape(n, m, target_rank)
            V_r = V_curr.reshape(m, p, target_rank)
            W_r = W_curr.reshape(p, n, target_rank)
            W_r_trans = np.transpose(W_r, (1, 0, 2))
            
            T = np.einsum('ijr,jkr,ikr->ijk', U_r, V_r, W_r_trans)
            diff = T - 1.0
            
            # dU[i,j,r] = 2 * sum_k diff[i,j,k] * V[j,k,r] * W[k,i,r]
            dU = 2.0 * np.einsum('ijk,jkr,ikr->ijr', diff, V_r, W_r_trans)
            
            # dV[j,k,r] = 2 * sum_i diff[i,j,k] * U[i,j,r] * W[k,i,r]
            dV = 2.0 * np.einsum('ijk,ijr,ikr->jkr', diff, U_r, W_r_trans)
            
            # dW[k,i,r] = 2 * sum_j diff[i,j,k] * U[i,j,r] * V[j,k,r]
            dW_pnr = 2.0 * np.einsum('ijk,ijr,jkr->kir', diff, U_r, V_r)
            
            # Update
            U_curr -= learning_rate * dU.reshape(n*m, target_rank)
            V_curr -= learning_rate * dV.reshape(m*p, target_rank)
            W_curr -= learning_rate * dW_pnr.reshape(p*n, target_rank)
            
            # Check convergence
            if it % 10 == 0:
                res_check = compute_residual_fast(U_curr, V_curr, W_curr)
                if res_check < convergence_threshold:
                    best_rank = target_rank
                    best_U = U_curr
                    best_V = V_curr
                    best_W = W_curr
                    break
        
        if best_rank <= target_rank:
            break  # Found a solution, no need to try higher ranks
    
    # If we found a better solution, use it; otherwise use schoolbook
    if best_rank < rank_sb:
        U_out = best_U
        V_out = best_V
        W_out = best_W
    else:
        U_out = U_sb
        V_out = V_sb
        W_out = W_sb
    
    # Convert to lists for return
    return {"U": U_out.tolist(), "V": V_out.tolist(), "W": W_out.tolist()}
# EVOLVE-BLOCK-END
