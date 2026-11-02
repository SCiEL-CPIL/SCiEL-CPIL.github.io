import numpy as np
import time

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    # Target Tensor
    # T[alpha, beta, gamma] where alpha=i*m+j, beta=j*p+k, gamma=k*n+i
    T_target = np.zeros((n * m, m * p, p * n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                alpha = i * m + j
                beta = j * p + k
                gamma = k * n + i
                T_target[alpha, beta, gamma] = 1

    # Schoolbook Solution (Rank 50)
    best_rank = n * m * p
    best_U = np.zeros((n * m, best_rank), dtype=np.int64)
    best_V = np.zeros((m * p, best_rank), dtype=np.int64)
    best_W = np.zeros((p * n, best_rank), dtype=np.int64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i * m + j, col] = 1
                best_V[j * p + k, col] = 1
                best_W[k * n + i, col] = 1
                col += 1

    # Search for lower rank
    # Strategy: Fix R-1 random terms, check if residual is rank 1
    # We iterate R downwards from 49 to 30 (or lower if time permits)
    
    # Pre-compute indices for faster access if needed, but einsum is vectorized
    # The main cost is the einsum and the random generation.
    
    for R in range(49, 29, -1):
        if time.time() > deadline:
            break
        
        # Adjust attempts based on time remaining
        # Estimate time per attempt:
        # Random gen: ~10us
        # Einsum: ~20us
        # Residual check: ~10us
        # Total ~50us per attempt.
        # 100s / 50us = 2,000,000 attempts possible in theory.
        # Let's aim for 100,000 attempts per rank, but break early if time is tight.
        
        found = False
        attempts = 0
        # Dynamic attempt count: try to spend ~10s per rank if possible
        # But we have 20 ranks. 110s / 20 = 5.5s per rank.
        # 5.5s / 50us = 110,000 attempts.
        max_attempts = 100000
        
        for _ in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Generate R-1 random terms
            # Using small integers to keep residual small and exact
            # Range -3 to 3 seems good
            U_prev = np.random.randint(-3, 4, (n * m, R - 1))
            V_prev = np.random.randint(-3, 4, (m * p, R - 1))
            W_prev = np.random.randint(-3, 4, (p * n, R - 1))
            
            # Compute partial sum T_calc = sum_{r=0}^{R-2} U_prev[:,r] * V_prev[:,r] * W_prev[:,r]
            T_calc = np.einsum('ar,br,cr->abc', U_prev, V_prev, W_prev, optimize=True)
            
            T_res = T_target - T_calc
            
            # Check if T_res is rank 1
            # If all zero, then R-1 is sufficient
            if not np.any(T_res):
                best_rank = R - 1
                best_U = U_prev
                best_V = V_prev
                best_W = W_prev
                found = True
                break
            
            # Find first non-zero index
            nz = np.nonzero(T_res)
            if len(nz[0]) == 0:
                continue
            
            i0, j0, k0 = nz[0][0], nz[1][0], nz[2][0]
            
            # Extract vectors assuming T_res[i,j,k] = a[i] * b[j] * c[k]
            a = T_res[:, j0, k0]
            b = T_res[i0, :, k0]
            c = T_res[i0, j0, :]
            
            # Check if T_res == a * b * c
            # Reconstruct and compare
            T_check = np.einsum('i,j,k->ijk', a, b, c)
            
            if np.array_equal(T_res, T_check):
                # It is rank 1!
                # The full solution is [U_prev, a], [V_prev, b], [W_prev, c]
                best_rank = R
                best_U = np.hstack((U_prev, a.reshape(-1, 1)))
                best_V = np.hstack((V_prev, b.reshape(-1, 1)))
                best_W = np.hstack((W_prev, c.reshape(-1, 1)))
                found = True
                break
        
        if found:
            # Found a solution for rank R (or R-1)
            # Continue to try lower ranks
            continue
        else:
            # If we failed to find a solution for this R, 
            # it doesn't mean R is impossible, but it's unlikely if we tried enough.
            # We continue to lower R.
            pass

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
