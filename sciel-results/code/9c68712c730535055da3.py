import numpy as np
import time

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Build the target tensor T_slices
    # T_slices[w, u, v] = 1 if u=i*m+j, v=j*p+k, w=k*n+i
    T_slices = np.zeros((p * n, n * m, m * p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T_slices[w, u, v] = 1
    
    # Pre-flatten T_slices for faster solving
    T_flat = T_slices.reshape(p * n, -1)
    
    best_rank = rank_schoolbook
    # Initialize with schoolbook as fallback
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
                
    best_U = U_sb.tolist()
    best_V = V_sb.tolist()
    best_W = W_sb.tolist()

    start_time = time.time()
    time_limit = 115  # Leave 5s margin
    
    # Search ranks from 89 down to 70
    # We prioritize finding *any* valid decomposition lower than 90.
    for R in range(rank_schoolbook - 1, 69, -1):
        if time.time() - start_time > time_limit:
            break
            
        found = False
        # Number of attempts depends on remaining time
        # Estimate cost per attempt: 
        # Building A: R * (n*m*m*p) operations ~ R * 450
        # Solving 18 systems of size 450 x R.
        # Should be fast. Let's try 2000 attempts per rank initially.
        
        max_attempts = 5000
        attempts_made = 0
        
        while attempts_made < max_attempts:
            if time.time() - start_time > time_limit:
                break
                
            attempts_made += 1
            
            # Random half-integer initialization for U and V
            # Entries in {-1, -0.5, 0, 0.5, 1}
            # Using random integers -2 to 2 divided by 2
            U_curr = np.random.randint(-2, 3, size=(n * m, R)) / 2.0
            V_curr = np.random.randint(-2, 3, size=(m * p, R)) / 2.0
            
            # Build A matrix: columns are vec(U_r V_r^T)
            # A shape: (n*m*m*p, R)
            # We can build this efficiently
            A = np.zeros((n * m * m * p, R))
            for r in range(R):
                M_r = np.outer(U_curr[:, r], V_curr[:, r])
                A[:, r] = M_r.flatten()
                
            # Solve for W for each slice w
            # W_sol shape: (p*n, R)
            W_sol = np.zeros((p * n, R))
            
            # Use lstsq for each w
            # To speed up, we can use np.linalg.lstsq on the whole batch?
            # No, lstsq is per right-hand-side.
            # But we can compute the pseudo-inverse of A once?
            # A is 450 x R. R is ~80.
            # Pinv is R x 450.
            # Then W_w = Pinv @ t_w.
            
            try:
                A_pinv = np.linalg.pinv(A)
                for w in range(p * n):
                    W_sol[w] = A_pinv @ T_flat[w]
            except np.linalg.LinAlgError:
                continue
                
            # Check if W_sol is close to half-integers
            W_rounded = np.round(W_sol * 2) / 2.0
            diff = np.abs(W_sol - W_rounded)
            
            if np.max(diff) < 1e-6:
                # Potential match. Verify exactness with rounded U, V, W
                # Reconstruct T
                # T_recon[w] = sum_r W_rounded[w,r] * outer(U_curr[:,r], V_curr[:,r])
                # We can do this efficiently:
                # T_recon_flat = (W_rounded @ A) ? No.
                # A columns are vec(M_r).
                # W_rounded @ A gives (p*n, n*m*m*p)
                # Reshape to (p*n, n*m, m*p)
                T_recon_flat = W_rounded @ A
                T_recon = T_recon_flat.reshape(p * n, n * m, m * p)
                
                # Check against T_slices
                if np.allclose(T_recon, T_slices, atol=1e-9):
                    # Success
                    best_rank = R
                    best_U = U_curr.tolist()
                    best_V = V_curr.tolist()
                    best_W = W_rounded.tolist()
                    found = True
                    break
        
        if found:
            # We found a solution for this R. 
            # Since we iterate downwards, this is the best so far.
            # We can stop or continue to see if we can go lower.
            # Given time limits, stopping at the first found (lowest R tried) is safe.
            # But maybe we can find 75 after finding 85?
            # The loop continues to lower R.
            # So we don't break the outer loop, we just update best.
            # However, if we find 85, and then try 84, 83... it might take too long.
            # Let's just let the loop run until time limit or R=70.
            pass

    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
