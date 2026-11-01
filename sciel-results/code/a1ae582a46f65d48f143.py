import time
import numpy as np

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank is n*m*p = 45
    # We'll try to find a lower rank decomposition using ALS
    # For now, return the schoolbook decomposition as a fallback
    
    rank = n * m * p
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U[a, col] = 1
                V[b, col] = 1
                W[c, col] = 1
                col += 1
    
    # Try to reduce rank using ALS with a deadline
    deadline = time.time() + 100  # 100 seconds for search
    
    # Build the matrix multiplication tensor
    # T[i,j,k] = 1 if the output position (k*n+i) corresponds to input positions (i*m+j) and (j*p+k)
    # The tensor is: T[a,b,c] where a=i*m+j, b=j*p+k, c=k*n+i
    # We want: sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c]
    
    # Create the target tensor
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1
    
    # Try ALS with decreasing rank
    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for trial_rank in range(30, rank + 1):  # Try ranks from 30 to 45
        if time.time() > deadline - 30:
            break
        
        # Initialize with random matrices
        np.random.seed(42 + trial_rank)
        U_try = np.random.randn(n*m, trial_rank)
        V_try = np.random.randn(m*p, trial_rank)
        W_try = np.random.randn(p*n, trial_rank)
        
        # ALS iterations
        converged = False
        for it in range(50):
            if time.time() > deadline - 10:
                break
            
            # Update U: for each a, U[a,:] = (V^T * W^T * T[a,:,:]) / (V^T * W^T)
            # Actually, we need to solve: U[a,r] = sum_{b,c} T[a,b,c] * V[b,r] * W[c,r] / (sum_{b,c} V[b,r]^2 * W[c,r]^2)
            # This is a least squares problem
            
            # For each a, we want to minimize: sum_{b,c} (T[a,b,c] - sum_r U[a,r] V[b,r] W[c,r])^2
            # Taking derivative w.r.t U[a,r]: sum_{b,c} T[a,b,c] V[b,r] W[c,r] - U[a,r] sum_{b,c} V[b,r]^2 W[c,r]^2 = 0
            # So U[a,r] = sum_{b,c} T[a,b,c] V[b,r] W[c,r] / sum_{b,c} V[b,r]^2 W[c,r]^2
            
            denom_U = np.einsum('br,cr->r', V_try, W_try)  # sum_{b,c} V[b,r]^2 W[c,r]^2
            # Actually that's wrong. Let me redo.
            # sum_{b,c} V[b,r]^2 W[c,r]^2 = (sum_b V[b,r]^2) * (sum_c W[c,r]^2)
            sum_V2 = np.sum(V_try**2, axis=0)  # shape (trial_rank,)
            sum_W2 = np.sum(W_try**2, axis=0)  # shape (trial_rank,)
            denom_U = sum_V2 * sum_W2  # shape (trial_rank,)
            
            # numerator[a,r] = sum_{b,c} T[a,b,c] V[b,r] W[c,r]
            # This is: for each a, (T[a,:,:] * V[:,r] * W[:,r]).sum()
            # We can compute this as: einsum('abc,br,cr->ar', T, V_try, W_try)
            numer_U = np.einsum('abc,br,cr->ar', T, V_try, W_try)
            
            U_new = numer_U / denom_U[None, :]
            
            # Update V
            sum_U2 = np.sum(U_new**2, axis=0)
            sum_W2 = np.sum(W_try**2, axis=0)
            denom_V = sum_U2 * sum_W2
            numer_V = np.einsum('abc,ar,cr->br', T, U_new, W_try)
            V_new = numer_V / denom_V[None, :]
            
            # Update W
            sum_U2 = np.sum(U_new**2, axis=0)
            sum_V2 = np.sum(V_new**2, axis=0)
            denom_W = sum_U2 * sum_V2
            numer_W = np.einsum('abc,ar,br->cr', T, U_new, V_new)
            W_new = numer_W / denom_W[None, :]
            
            # Check convergence
            diff = np.linalg.norm(U_new - U_try) + np.linalg.norm(V_new - V_try) + np.linalg.norm(W_new - W_try)
            U_try, V_try, W_try = U_new, V_new, W_new
            
            if diff < 1e-8:
                converged = True
                break
        
        # Verify the decomposition
        # Compute the reconstructed tensor and compare with T
        recon = np.einsum('ar,br,cr->abc', U_try, V_try, W_try)
        error = np.linalg.norm(recon - T)
        
        if error < 1e-6:
            best_rank = trial_rank
            best_U, best_V, best_W = U_try, V_try, W_try
            break  # Found a valid decomposition at this rank
    
    # Convert to lists of integers or half-integers if possible
    # For now, just return the matrices as floats
    
    # Round to nearest integer or half-integer
    def round_to_half(x):
        return np.round(x * 2) / 2
    
    U_out = round_to_half(best_U)
    V_out = round_to_half(best_V)
    W_out = round_to_half(best_W)
    
    # Verify after rounding
    recon = np.einsum('ar,br,cr->abc', U_out, V_out, W_out)
    error = np.linalg.norm(recon - T)
    
    if error > 1e-6:
        # Rounding broke it, use original
        U_out = best_U
        V_out = best_V
        W_out = best_W
    
    # Convert to lists
    U_list = U_out.tolist()
    V_list = V_out.tolist()
    W_list = W_out.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
