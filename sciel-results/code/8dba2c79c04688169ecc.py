import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Target ranks to try
    target_ranks = [20, 22, 24, 28, 36]
    
    # Precompute the tensor T as a 3D array or flattened
    # T[i, j, k] = 1 if it contributes to C[i, k] from A[i, j] and B[j, k]
    # Actually, the tensor T is defined by T[(i,j), (j',k), (i',k')] = delta_{j,j'} delta_{i,i'} delta_{k,k'}
    # We want U, V, W such that sum_r U[:,r] V[:,r] W[:,r] == T_flattened
    # Dimensions:
    # U: n*m x R = 9 x R
    # V: m*p x R = 12 x R
    # W: p*n x R = 12 x R
    
    # Flatten T
    # Index for U: i*m + j
    # Index for V: j*p + k
    # Index for W: k*n + i
    # T is a tensor of shape (n*m, m*p, p*n)
    # T[a, b, c] = 1 if a=(i,j), b=(j,k), c=(k,i) and j matches j, i matches i, k matches k?
    # Let's construct T explicitly.
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i*m + j
                v_idx = j*p + k
                w_idx = k*n + i
                T[u_idx, v_idx, w_idx] = 1.0
                
    # Function to compute residual for given U, V, W
    def residual(params):
        R = len(target_ranks) # No, R is passed in
        # This function needs to know R. We will define it inside the loop.
        pass

    best_solution = None
    best_rank = n * m * p
    best_U = [[0] * best_rank for _ in range(n * m)]
    best_V = [[0] * best_rank for _ in range(m * p)]
    best_W = [[0] * best_rank for _ in range(p * n)]
    
    # Initialize schoolbook as fallback
    rank_sb = n * m * p
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    best_U = U_sb.tolist()
    best_V = V_sb.tolist()
    best_W = W_sb.tolist()
    best_rank = rank_sb

    for R in target_ranks:
        if time.time() > deadline - 20:
            break
            
        # Try to find a decomposition of rank R
        # We will do a few random restarts
        max_restarts = 5 if R <= 22 else 3
        
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
                
            # Random initialization
            np.random.seed(42 + restart)
            U0 = np.random.randn(n*m, R) * 0.1
            V0 = np.random.randn(m*p, R) * 0.1
            W0 = np.random.randn(p*n, R) * 0.1
            
            # Flatten
            x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
            
            def objective(x):
                U = x[:n*m*R].reshape(n*m, R)
                V = x[n*m*R : n*m*R + m*p*R].reshape(m*p, R)
                W = x[n*m*R + m*p*R :].reshape(p*n, R)
                
                # Compute sum of U[:,r] V[:,r] W[:,r]
                # This is a tensor reconstruction
                # T_recon[a, b, c] = sum_r U[a,r] V[b,r] W[c,r]
                # We can do this efficiently with einsum or loops
                # T_recon = np.einsum('ar,br,cr->abc', U, V, W)
                T_recon = np.einsum('ar,br,cr->abc', U, V, W)
                diff = T_recon - T
                return np.sum(diff * diff)
            
            # Optimize
            res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 50, 'ftol': 1e-10})
            
            if res.success or res.fun < 1e-6:
                # Check if we can snap to rationals
                U_opt = res.x[:n*m*R].reshape(n*m, R)
                V_opt = res.x[n*m*R : n*m*R + m*p*R].reshape(m*p, R)
                W_opt = res.x[n*m*R + m*p*R :].reshape(p*n, R)
                
                # Snap to nearest half-integer
                U_snap = np.round(U_opt * 2) / 2
                V_snap = np.round(V_opt * 2) / 2
                W_snap = np.round(W_opt * 2) / 2
                
                # Verify exactness
                T_snap = np.einsum('ar,br,cr->abc', U_snap, V_snap, W_snap)
                if np.allclose(T_snap, T, atol=1e-9):
                    # Found a valid decomposition
                    best_rank = R
                    best_U = U_snap.tolist()
                    best_V = V_snap.tolist()
                    best_W = W_snap.tolist()
                    break # Break restart loop, try smaller R? No, we iterate R ascending, so this is the best R for this loop iteration.
                    # Actually, we want the smallest R. So if we find R=20, we should stop.
                    # The loop is for R in [20, 22, ...]. So if we find 20, we break out of the R loop too.
                    # Let's use a flag.
                    if R == 20:
                        # Return immediately if we find the lowest target
                        return {"U": best_U, "V": best_V, "W": best_W}
                    # If we find 22, we might want to check if 20 is possible? 
                    # But 20 is already tried and failed. So 22 is the best found so far.
                    # We can break the R loop if we find a solution, as we are going from low to high.
                    break
        
        if best_rank < rank_sb:
            # We found a better rank
            # Since we iterate R in increasing order, the first success is the minimum R in our list.
            # We can stop searching for higher ranks.
            break

    return {"U": best_U, "V": best_V, "W": best_W}
