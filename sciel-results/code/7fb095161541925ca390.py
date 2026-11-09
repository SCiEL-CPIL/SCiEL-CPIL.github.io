"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # safety margin
    
    # Build the target tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    best_rank = n * m * p  # schoolbook
    best_U = None
    best_V = None
    best_W = None
    
    # Schoolbook as fallback
    U_school = np.zeros((n*m, best_rank))
    V_school = np.zeros((m*p, best_rank))
    W_school = np.zeros((p*n, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Try to find lower rank decompositions
    for target_rank in range(20, best_rank):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        for trial in range(50):
            if time.time() > deadline:
                break
            
            U = np.random.randn(n*m, target_rank)
            V = np.random.randn(m*p, target_rank)
            W = np.random.randn(p*n, target_rank)
            
            # ALS optimization
            for it in range(200):
                if time.time() > deadline:
                    break
                
                # Update U: for each column r, U[:,r] = argmin ||T[:,:,r] - U[:,r] ⊙ V[:,r] ⊙ W[:,r]||
                # Actually, we need to solve: sum_r U[:,r] V[:,r] W[:,r] = T
                # This is not straightforward ALS because it's a 3-way tensor.
                
                # Use gradient descent instead
                # Compute current tensor estimate
                # T_est[i,a,b] = sum_r U[i,r] * V[a,r] * W[b,r]
                
                # Gradient w.r.t. U: grad_U[i,r] = sum_{a,b} T[i,a,b] * V[a,r] * W[b,r] - sum_r' ... 
                # Actually: dT_est[i,a,b]/dU[i,r] = V[a,r] * W[b,r]
                # So grad_U[i,r] = sum_{a,b} (T[i,a,b] - T_est[i,a,b]) * V[a,r] * W[b,r]
                
                # This is expensive. Let me use a simpler approach.
                
                # Compute T_est
                T_est = np.einsum('ir,ar,br->iab', U, V, W)
                diff = T - T_est
                
                # Gradient for U
                # grad_U[i,r] = sum_{a,b} diff[i,a,b] * V[a,r] * W[b,r]
                grad_U = np.einsum('iab,ar,br->ir', diff, V, W)
                grad_V = np.einsum('iab,ir,br->ar', diff, U, W)
                grad_W = np.einsum('iab,ir,ar->br', diff, U, V)
                
                # Learning rate (adaptive)
                lr = 0.01 / (1 + it * 0.01)
                
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                # Check convergence
                if it % 50 == 0:
                    err = np.linalg.norm(diff) / np.linalg.norm(T)
                    if err < 1e-10:
                        break
            
            # Check final error
            T_est = np.einsum('ir,ar,br->iab', U, V, W)
            err = np.linalg.norm(T - T_est) / np.linalg.norm(T)
            
            if err < 1e-8:
                # Try to round to integer/half-integer
                U_round = np.round(U * 2) / 2
                V_round = np.round(V * 2) / 2
                W_round = np.round(W * 2) / 2
                
                T_est_round = np.einsum('ir,ar,br->iab', U_round, V_round, W_round)
                err_round = np.linalg.norm(T - T_est_round) / np.linalg.norm(T)
                
                if err_round < 1e-10:
                    # Found an exact decomposition!
                    best_rank = target_rank
                    best_U, best_V, best_W = U_round, V_round, W_round
                    break
        
        if best_rank == target_rank:
            break
    
    # Convert to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
