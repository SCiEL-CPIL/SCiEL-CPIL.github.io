import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    n, m, p = 2, 3, 8
    nm, mp, pn = n*m, m*p, p*n  # 6, 24, 16
    
    # Build the target tensor
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
                
    # Schoolbook decomposition
    best_rank = n*m*p  # 48
    best_U = None
    best_V = None
    best_W = None
    
    U_sb = np.zeros((nm, best_rank))
    V_sb = np.zeros((mp, best_rank))
    W_sb = np.zeros((pn, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    best_U, best_V, best_W = U_sb, V_sb, W_sb

    # Function to verify a decomposition
    def verify(U, V, W, R):
        # Check if U, V, W are half-integers
        # They should be rounded before calling this, or we check here
        # We assume they are already rounded to 0.5 grid in the caller
        
        # Check all tensor entries
        for ii in range(nm):
            for jj in range(mp):
                for kk in range(pn):
                    val = 0.0
                    for r in range(R):
                        val += U[ii, r] * V[jj, r] * W[kk, r]
                    
                    # Determine expected value
                    expected = 0.0
                    # ii = i*m + j
                    # jj = j*p + k
                    # kk = k*n + i
                    i = ii // m
                    j = ii % m
                    j2 = jj // p
                    k = jj % p
                    k2 = kk // n
                    i2 = kk % n
                    
                    if i == i2 and j == j2 and k == k2:
                        expected = 1.0
                    
                    if abs(val - expected) > 1e-9:
                        return False
        return True

    # Function to try ALS for a specific rank
    def try_als(R, num_trials):
        for trial in range(num_trials):
            if time.time() > deadline:
                return None
            
            np.random.seed(42 + R * 1000 + trial)
            # Initialize with random Gaussian
            U = np.random.randn(nm, R)
            V = np.random.randn(mp, R)
            W = np.random.randn(pn, R)
            
            # ALS iterations
            for it in range(100):
                if time.time() > deadline:
                    return None
                
                # Update U
                for i in range(n):
                    for j in range(m):
                        M_mat = np.zeros((p, R))
                        for k in range(p):
                            M_mat[k, :] = V[j*p+k, :] * W[k*n+i, :]
                        b = np.ones(p)
                        x, _, _, _ = np.linalg.lstsq(M_mat, b, rcond=None)
                        U[i*m+j, :] = x
                
                # Update V
                for j in range(m):
                    for k in range(p):
                        M_mat = np.zeros((n, R))
                        for i in range(n):
                            M_mat[i, :] = U[i*m+j, :] * W[k*n+i, :]
                        b = np.ones(n)
                        x, _, _, _ = np.linalg.lstsq(M_mat, b, rcond=None)
                        V[j*p+k, :] = x
                
                # Update W
                for i in range(n):
                    for k in range(p):
                        M_mat = np.zeros((m, R))
                        for j in range(m):
                            M_mat[j, :] = U[i*m+j, :] * V[j*p+k, :]
                        b = np.ones(m)
                        x, _, _, _ = np.linalg.lstsq(M_mat, b, rcond=None)
                        W[k*n+i, :] = x
                        
                # Check convergence (residual)
                residual = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                            residual += abs(val - 1.0)
                # Check off-diagonals too? 
                # For speed, only check on-diagonal in loop, full check in verify
                if residual < 1e-4:
                    break
            
            if time.time() > deadline:
                return None

            # Polish: Round to half-integers
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            
            if verify(U_r, V_r, W_r, R):
                return (U_r, V_r, W_r)
                
        return None

    # Search strategy:
    # Try low ranks first with limited trials.
    # If found, return.
    # If not, go higher.
    
    # Ranks to try: 24 to 47
    # We prioritize lower ranks.
    
    # To save time, we will try:
    # 24-28 with 20 trials each
    # 29-35 with 10 trials each
    # 36-40 with 5 trials each
    # 41-47 with 2 trials each
    
    rank_ranges = [
        (24, 29, 20),
        (29, 36, 10),
        (36, 41, 5),
        (41, 48, 2)
    ]
    
    best_found = None
    
    for start_r, end_r, trials in rank_ranges:
        for R in range(start_r, end_r):
            if time.time() > deadline:
                break
                
            result = try_als(R, trials)
            if result:
                best_found = (R, result)
                break # Found a valid decomposition for this rank, stop searching higher for this range?
                # Actually, if we find R, we want the MINIMUM.
                # So if we find 25, we should stop and return 25.
                # If we find 30, we should stop and return 30?
                # Yes, because we are iterating from low to high.
                # The first one we find is the lowest in this range.
                # But what if 24 was possible but we failed?
                # We accept the risk.
                break
        
        if best_found:
            break
            
    if best_found:
        R, (U, V, W) = best_found
        best_rank = R
        best_U = U
        best_V = V
        best_W = W

    # Convert to lists
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
