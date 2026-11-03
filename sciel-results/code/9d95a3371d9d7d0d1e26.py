import time
import numpy as np

DIMENSIONS = [2, 3, 8]

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_schoolbook = n * m * p
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
                
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_schoolbook
    
    # Build target tensor: T[i*m+j, j*p+k, k*n+i] = 1
    # Actually, the tensor is defined as T[a,b,c] where a = i*m+j, b = j*p+k, c = k*n+i
    # The matrix multiplication tensor: T[i,j,k] = 1 for the (i,j,k) term
    # In flattened form: T[(i*m+j), (j*p+k), (k*n+i)] = 1
    
    # Build the target tensor as a 3D array
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Try to find a low-rank decomposition using gradient descent
    # We'll try ranks from 40 down to 32
    target_ranks = [40, 38, 36, 34, 32]
    
    for rank in target_ranks:
        if time.time() > deadline - 10:
            break
        
        # Initialize with random values
        np.random.seed(42)
        U = np.random.randn(n*m, rank) * 0.1
        V = np.random.randn(m*p, rank) * 0.1
        W = np.random.randn(p*n, rank) * 0.1
        
        # Also try initializing with schoolbook subset
        # Use first 'rank' schoolbook terms as initialization
        U_init = np.zeros((n*m, rank))
        V_init = np.zeros((m*p, rank))
        W_init = np.zeros((p*n, rank))
        for r in range(min(rank, n*m*p)):
            i = r // (m*p)
            j = (r // p) % m
            k = r % p
            U_init[i*m+j, r] = 1
            V_init[j*p+k, r] = 1
            W_init[k*n+i, r] = 1
        
        # Try both initializations
        for init_type in ['random', 'schoolbook']:
            if time.time() > deadline - 5:
                break
            
            if init_type == 'random':
                U_curr = U.copy()
                V_curr = V.copy()
                W_curr = W.copy()
            else:
                U_curr = U_init.copy()
                V_curr = V_init.copy()
                W_curr = W_init.copy()
            
            # Gradient descent on loss = ||T - sum_r U_r ⊗ V_r ⊗ W_r||^2
            lr = 0.01
            best_loss = np.inf
            best_U, best_V, best_W = U_curr.copy(), V_curr.copy(), W_curr.copy()
            
            for step in range(5000):
                if time.time() > deadline - 2:
                    break
                
                # Compute current reconstruction
                # T_recon[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
                T_recon = np.einsum('ar,br,cr->abc', U_curr, V_curr, W_curr)
                loss = np.sum((T - T_recon) ** 2)
                
                if loss < best_loss:
                    best_loss = loss
                    best_U, best_V, best_W = U_curr.copy(), V_curr.copy(), W_curr.copy()
                
                if loss < 1e-10:
                    break
                
                # Compute gradients
                # dL/dU[a,r] = 2 * sum_{b,c} (T[a,b,c] - T_recon[a,b,c]) * V[b,r] * W[c,r]
                residual = T - T_recon
                grad_U = 2 * np.einsum('abc,br,cr->ar', residual, V_curr, W_curr)
                grad_V = 2 * np.einsum('abc,ar,cr->br', residual, U_curr, W_curr)
                grad_W = 2 * np.einsum('abc,ar,br->cr', residual, U_curr, V_curr)
                
                # Update
                U_curr -= lr * grad_U
                V_curr -= lr * grad_V
                W_curr -= lr * grad_W
            
            if best_loss < 1e-10:
                # Found a valid decomposition
                # Convert to integer/half-integer if possible
                U_out = best_U.tolist()
                V_out = best_V.tolist()
                W_out = best_W.tolist()
                
                # Round to nearest 0.5 and check if still valid
                U_rounded = np.round(U_out * 2) / 2
                V_rounded = np.round(V_out * 2) / 2
                W_rounded = np.round(W_out * 2) / 2
                
                T_check = np.einsum('ar,br,cr->abc', U_rounded, V_rounded, W_rounded)
                if np.max(np.abs(T_check - T)) < 1e-9:
                    best_result = {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
                    best_rank = rank
                    break
                else:
                    # Use the unrounded version if it's close enough
                    if best_loss < 1e-8:
                        best_result = {"U": U_out, "V": V_out, "W": W_out}
                        best_rank = rank
                        break
        
        if best_rank < rank_schoolbook:
            break
    
    return best_result
