import numpy as np
import time

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 145.0  # 5s safety margin before 150s kill
    
    # Build the target tensor T of shape (n*m, m*p, p*n)
    # T[i*m + j, j*p + k, k*n + i] = 1
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Precompute flattened views
    T_flat_u = T.reshape(n * m, -1)  # (n*m, m*p*p*n)
    T_flat_v = T.reshape(m * p, -1)  # (m*p, n*m*p*n)
    T_flat_w = T.reshape(p * n, -1)  # (p*n, n*m*m*p)
    
    # Schoolbook decomposition (Rank 48)
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school
    
    # Lower bound: max(n*m, m*p, p*n) = max(12, 16, 12) = 16
    lower_bound = max(n * m, m * p, p * n)
    
    # Try ranks from lower bound up to schoolbook - 1
    for R in range(lower_bound, rank_school):
        if time.time() > deadline - 10:  # Stop if running out of time
            break
        
        # Try multiple random initializations for this rank
        max_trials = 8
        for trial in range(max_trials):
            if time.time() > deadline - 10:
                break
            
            # Random initialization with moderate values
            np.random.seed(42 + R * 100 + trial)  # Reproducible
            U = np.random.randn(n * m, R)
            V = np.random.randn(m * p, R)
            W = np.random.randn(p * n, R)
            
            # Alternating least squares iterations
            max_iter = 150
            prev_error = np.inf
            for it in range(max_iter):
                if time.time() > deadline - 5:
                    break
                
                # Fix V and W, optimize U
                # Build B matrix: (R, m*p*p*n) where B[r] = (V[:, r] ⊗ W[:, r]).reshape(-1)
                B_u = np.array([np.outer(V[:, r], W[:, r]).reshape(-1) for r in range(R)])
                
                try:
                    U_new, _, _, _ = np.linalg.lstsq(B_u.T, T_flat_u, rcond=None)
                    U = U_new
                except:
                    pass
                
                # Similarly for V and W
                B_v = np.array([np.outer(U[:, r], W[:, r]).reshape(-1) for r in range(R)])
                try:
                    V_new, _, _, _ = np.linalg.lstsq(B_v.T, T_flat_v, rcond=None)
                    V = V_new
                except:
                    pass
                
                B_w = np.array([np.outer(U[:, r], V[:, r]).reshape(-1) for r in range(R)])
                try:
                    W_new, _, _, _ = np.linalg.lstsq(B_w.T, T_flat_w, rcond=None)
                    W = W_new
                except:
                    pass
                
                # Compute error efficiently using tensor contraction
                # T_recon[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
                recon = np.einsum('ar,br,cr->abc', U, V, W)
                error = np.linalg.norm(T - recon)
                
                if abs(prev_error - error) < 1e-12:
                    break
                prev_error = error
            
            # Check if we found an exact (or near-exact) decomposition
            if prev_error < 1e-8:
                # Try to round to rational numbers (integers or half-integers)
                # Check if rounding gives an exact decomposition
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                recon_rounded = np.einsum('ar,br,cr->abc', U_rounded, V_rounded, W_rounded)
                error_rounded = np.linalg.norm(T - recon_rounded)
                
                if error_rounded < 1e-8:
                    best_U, best_V, best_W = U_rounded, V_rounded, W_rounded
                    best_rank = R
                    break  # Found a better rank, stop searching for this R
        
        # If we found a better rank, continue searching for lower ranks
        if best_rank < rank_school:
            if best_rank == lower_bound:
                break
    
    # Return the best found
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
