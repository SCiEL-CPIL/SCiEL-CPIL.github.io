import numpy as np
import time

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Precompute the target tensor matrix T_mat
    # T_mat has shape (nm * mp, pn)
    # T_mat[a*mp + b, c] = T[a, b, c]
    # T[a, b, c] = 1 if a = i*m + j, b = j*p + k, c = k*n + i
    # else 0
    
    T_mat = np.zeros((nm * mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                idx_ab = a * mp + b
                T_mat[idx_ab, c] = 1.0
                
    # Search for rank R
    # Lower bound is roughly 20, upper 30
    # We try to find the minimal R
    
    # Time limit check
    start_time = time.time()
    time_limit = 100.0 # Leave some buffer for 120s limit
    
    # We try ranks from 20 to 30
    for R in range(20, 31):
        if time.time() - start_time > time_limit:
            break
            
        # Number of trials depends on R, maybe fewer for higher R
        # But higher R is easier to solve?
        # Actually, higher R means more variables, so solution is more likely to exist and be "clean"?
        # Lower R is harder.
        
        num_trials = 2000 if R < 25 else 1000
        
        for trial in range(num_trials):
            if time.time() - start_time > time_limit:
                break
                
            # Generate random U, V
            # Use small integers to increase chance of rational solutions
            # Entries in {-1, 0, 1} is a good start
            U = np.random.choice([-1, 0, 1], size=(nm, R))
            V = np.random.choice([-1, 0, 1], size=(mp, R))
            
            # Construct B
            # B[:, r] = vec(U[:, r] * V[:, r]^T)
            # U[:, r] is nm x 1
            # V[:, r] is mp x 1
            # Outer product is nm x mp
            # Flatten to nm*mp
            
            B = np.zeros((nm * mp, R))
            for r in range(R):
                outer = np.outer(U[:, r], V[:, r])
                B[:, r] = outer.flatten()
                
            # Check rank
            # If rank < R, we cannot solve uniquely, but we can still try lstsq
            # However, if rank < R, the solution is not unique, 
            # and the "minimum norm" solution might not be integer.
            # But if a solution exists, there might be an integer one.
            # Let's just try to solve.
            
            # Solve B X = T_mat
            # X will be R x pn
            try:
                # Use lstsq for stability
                X, residuals, rank, sv = np.linalg.lstsq(B, T_mat, rcond=None)
            except Exception:
                continue
                
            W = X.T # pn x R
            
            # Check if W has half-integer entries
            # Round to nearest half-integer
            W_check = np.round(W * 2) / 2.0
            U_check = np.round(U * 2) / 2.0
            V_check = np.round(V * 2) / 2.0
            
            # Check exactness
            # Reconstruct T_check
            # T_check[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
            # We can check the matrix product form
            # T_mat_reconstructed = ?
            # Instead of full reconstruction, check B @ W.T == T_mat
            # Note: B @ W.T is (nm*mp) x pn
            # This is exactly the flattened tensor.
            
            T_recon = B @ W.T
            if np.allclose(T_recon, T_mat, atol=1e-9):
                if np.allclose(W, W_check, atol=1e-9) and np.allclose(U, U_check, atol=1e-9) and np.allclose(V, V_check, atol=1e-9):
                    return {
                        "U": U_check.tolist(),
                        "V": V_check.tolist(),
                        "W": W_check.tolist()
                    }
                    
    # Fallback to schoolbook rank 30
    rank = n * m * p
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
