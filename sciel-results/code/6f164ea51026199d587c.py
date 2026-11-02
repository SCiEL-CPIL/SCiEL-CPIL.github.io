"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Build the multiplication tensor T of shape (n*p, n*m, m*p)
    # T[c, a, b] = 1 if a%m == b//p, else 0
    # where c = (b%p)*n + (a//m)
    T = np.zeros((n * p, n * m, m * p), dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[c, a, b] = 1.0
    
    # Schoolbook decomposition as fallback
    rank_schoolbook = n * m * p
    U_school = np.zeros((n * p, rank_schoolbook))
    V_school = np.zeros((n * m, rank_schoolbook))
    W_school = np.zeros((m * p, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_school[c, col] = 1.0
                V_school[a, col] = 1.0
                W_school[b, col] = 1.0
                col += 1
    
    # Try to find lower rank decompositions
    start_time = time.time()
    deadline = start_time + 120  # 120 second deadline
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_schoolbook
    
    # Try ranks from lower to higher
    for target_rank in range(30, rank_schoolbook + 1):
        if time.time() > deadline - 5:
            break
        
        # Try multiple random initializations
        for attempt in range(50):
            if time.time() > deadline - 5:
                break
            
            # Random initialization
            U = np.random.randn(n * p, target_rank)
            V = np.random.randn(n * m, target_rank)
            W = np.random.randn(m * p, target_rank)
            
            # Iterative optimization: minimize ||T - sum_r U_r ⊗ V_r ⊗ W_r||
            for it in range(100):
                if time.time() > deadline - 5:
                    break
                
                # Compute current tensor estimate
                # T_est[c,a,b] = sum_r U[c,r] * V[a,r] * W[b,r]
                # Update each factor in turn
                
                # Update U: for each r, U[:,r] = argmin ||T - sum_r' ...||
                # This is a least squares problem
                for r in range(target_rank):
                    # Compute residual without term r
                    T_est = np.zeros_like(T)
                    for r2 in range(target_rank):
                        if r2 == r:
                            continue
                        T_est += np.outer(np.outer(U[:, r2], V[:, r2]), W[:, r2])
                    
                    residual = T - T_est
                    # Now solve: minimize ||residual - U[:,r] ⊗ V[:,r] ⊗ W[:,r]||
                    # This is: for each (a,b), residual[a,b,:] = U[:,r] * (V[a,r] * W[b,r])
                    # So U[:,r] = argmin ||residual - U_new ⊗ (V[:,r] * W[:,r])||
                    # Flatten: residual.reshape(-1) = U_new ⊗ (V[:,r]*W[:,r]) flattened
                    # This is a rank-1 approximation, use SVD
                    vw = V[:, r] * W[:, r]  # shape (n*m, m*p)
                    # residual as matrix: residual[:, :, :] -> for each c, residual[c, a, b]
                    # We want U_new[c] such that sum_{a,b} (residual[c,a,b] - U_new[c]*V[a,r]*W[b,r])^2 is min
                    # = sum_c (sum_{a,b} residual[c,a,b]^2 - 2*U_new[c]*sum_{a,b} residual[c,a,b]*V[a,r]*W[b,r] + U_new[c]^2 * sum_{a,b} V[a,r]^2*W[b,r]^2)
                    # Optimal: U_new[c] = sum_{a,b} residual[c,a,b]*V[a,r]*W[b,r] / sum_{a,b} V[a,r]^2*W[b,r]^2
                    
                    denom = np.sum(vw ** 2)
                    if denom < 1e-12:
                        continue
                    numer = np.einsum('cab,ab->c', residual, vw)
                    U[:, r] = numer / denom
                
                # Similarly update V and W
                for r in range(target_rank):
                    T_est = np.zeros_like(T)
                    for r2 in range(target_rank):
                        if r2 == r:
                            continue
                        T_est += np.outer(np.outer(U[:, r2], V[:, r2]), W[:, r2])
                    residual = T - T_est
                    uw = U[:, r] * W[:, r]  # shape (n*p, m*p)
                    denom = np.sum(uw ** 2)
                    if denom < 1e-12:
                        continue
                    numer = np.einsum('cab,cb->a', residual, uw)
                    V[:, r] = numer / denom
                
                for r in range(target_rank):
                    T_est = np.zeros_like(T)
                    for r2 in range(target_rank):
                        if r2 == r:
                            continue
                        T_est += np.outer(np.outer(U[:, r2], V[:, r2]), W[:, r2])
                    residual = T - T_est
                    uv = U[:, r] * V[:, r]  # shape (n*p, n*m)
                    denom = np.sum(uv ** 2)
                    if denom < 1e-12:
                        continue
                    numer = np.einsum('cab,ca->b', residual, uv)
                    W[:, r] = numer / denom
            
            # Check if we found an exact decomposition
            T_est = np.zeros_like(T)
            for r in range(target_rank):
                T_est += np.outer(np.outer(U[:, r], V[:, r]), W[:, r])
            
            error = np.linalg.norm(T - T_est)
            if error < 1e-8:
                # Found exact decomposition!
                best_U, best_V, best_W = U, V, W
                best_rank = target_rank
                break
        
        if best_rank == target_rank:
            break
    
    # Convert to lists for return
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END

