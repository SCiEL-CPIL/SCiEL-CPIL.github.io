import time
import numpy as np

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 145.0  # 5s safety margin before 150s kill
    
    # Build target tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
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
    
    # Lower bound: max(n*m, m*p, p*n) = 16
    lower_bound = max(n * m, m * p, p * n)
    
    # Try ranks from lower_bound+1 up to rank_school-1
    # We'll try a subset of ranks to stay within time budget
    ranks_to_try = list(range(lower_bound + 1, rank_school))
    # Sort by likelihood of success (closer to schoolbook might be easier)
    # Actually, let's try from low to high, but limit total trials
    
    max_trials_per_rank = 3
    max_als_iter = 50
    
    # Precompute tensor reshapes for efficient ALS
    # For optimizing U: T reshaped to (n*m, m*p*p*n)
    T_UM = T.reshape(n * m, m * p * p * n)
    # For optimizing V: T reshaped to (m*p, n*m*p*n)  
    T_VM = T.reshape(m * p, n * m * p * n)
    # For optimizing W: T reshaped to (p*n, n*m*m*p)
    T_WM = T.reshape(p * n, n * m * m * p)
    
    for R in ranks_to_try:
        if time.time() > deadline - 10:
            break
        
        found = False
        for trial in range(max_trials_per_rank):
            if time.time() > deadline - 5:
                break
            
            # Random initialization
            np.random.seed(12345 + R * 1000 + trial)
            U = np.random.randn(n * m, R) * 0.1
            V = np.random.randn(m * p, R) * 0.1
            W = np.random.randn(p * n, R) * 0.1
            
            prev_error = np.inf
            converged = False
            
            for it in range(max_als_iter):
                if time.time() > deadline - 3:
                    break
                
                # Optimize U: for each column r, U[:,r] = argmin ||T - sum_r' U_r' ⊗ V_r' ⊗ W_r'||_F
                # This is equivalent to: U[:,r] = (T - sum_{r'≠r} U_r' ⊗ V_r' ⊗ W_r') best rank-1 in mode 1
                # More efficiently: U = argmin ||T_UM - U @ (V⊗W)^T|| where V⊗W is (R, m*p*p*n)
                
                # Compute V⊗W: shape (R, m*p*p*n)
                VW = np.einsum('jr,kr->rjk', V, W).reshape(R, m * p * p * n)
                
                # Solve: minimize ||T_UM - U @ VW.T||_F
                # This is a linear least squares: U = T_UM @ VW.T @ (VW @ VW.T)^{-1}
                # Or use lstsq
                try:
                    U_new, _, _, _ = np.linalg.lstsq(VW.T, T_UM, rcond=None)
                    U = U_new
                except:
                    pass
                
                # Optimize V
                UW = np.einsum('ir,kr->rik', U, W).reshape(R, n * m * p * n)
                try:
                    V_new, _, _, _ = np.linalg.lstsq(UW.T, T_VM, rcond=None)
                    V = V_new
                except:
                    pass
                
                # Optimize W
                UV = np.einsum('ir,jr->rij', U, V).reshape(R, n * m * m * p)
                try:
                    W_new, _, _, _ = np.linalg.lstsq(UV.T, T_WM, rcond=None)
                    W = W_new
                except:
                    pass
                
                # Compute error
                if it % 5 == 0 or it == max_als_iter - 1:
                    # Compute tensor reconstruction error
                    recon = np.einsum('ir,jr,kr->ijk', U, V, W)
                    error = np.linalg.norm(T - recon)
                    if error < prev_error:
                        prev_error = error
                    if error < 1e-10:
                        converged = True
                        break
                    if abs(prev_error - error) < 1e-12:
                        break
            
            if converged or prev_error < 1e-10:
                # Verify exactness
                recon = np.einsum('ir,jr,kr->ijk', U, V, W)
                if np.allclose(T, recon, atol=1e-9):
                    # Check if entries are close to integer/half-integer
                    all_close = True
                    for mat in [U, V, W]:
                        rounded = np.round(mat * 2) / 2
                        if not np.allclose(mat, rounded, atol=1e-6):
                            all_close = False
                            break
                    if all_close:
                        best_U, best_V, best_W = U, V, W
                        best_rank = R
                        found = True
                        break
                    else:
                        # Still accept if error is tiny, even if not exactly half-integer
                        best_U, best_V, best_W = U, V, W
                        best_rank = R
                        found = True
                        break
        
        if found:
            break
    
    # Return best found
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
