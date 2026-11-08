import numpy as np
import scipy.linalg
import time

DIMENSIONS = [3, 3, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    n, m, p = DIMENSIONS
    # n=3, m=3, p=7
    # U shape: (n*m, R) = (9, R)
    # V shape: (m*p, R) = (21, R)
    # W shape: (p*n, R) = (21, R)
    # Indexing:
    # U[i*m + j, r]
    # V[j*p + k, r]
    # W[k*n + i, r]
    
    # Equation: sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1
    # Scaled by 2: sum_r U2[i*m+j, r] V2[j*p+k, r] W2[k*n+i, r] = 2
    # U2, V2, W2 are integers.
    
    num_eqs = n * m * p
    eq_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                eq_indices.append((i, j, k))
                
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    best_result = {'U': U_school.tolist(), 'V': V_school.tolist(), 'W': W_school.tolist()}
    best_rank = rank_school
    
    # Search for lower rank
    # We try ranks from 62 down to 40
    # For each rank, we try random integer initializations for U2, V2
    # and solve for W2
    
    # To improve search, we can try different distributions
    # 1. Sparse random integers
    # 2. Dense random integers
    
    for rank in range(rank_school - 1, 39, -1):
        if time.time() > deadline:
            break
            
        # Number of attempts per rank
        # We have ~110s total. 
        # If we try 20 ranks, we have ~5s per rank.
        # Each attempt involves solving a linear system of size 63 x (21*rank).
        # For rank=50, 63 x 1050. This is fast.
        # We can afford many attempts.
        
        attempts_per_rank = 2000
        # Adjust attempts based on remaining time?
        # Let's just run a fixed number and check time.
        
        for attempt in range(attempts_per_rank):
            if time.time() > deadline:
                break
                
            # Random initialization
            # Use small integers
            # Range -2 to 2 is a good start
            U2 = np.random.randint(-2, 3, size=(n*m, rank))
            V2 = np.random.randint(-2, 3, size=(m*p, rank))
            
            # Construct A matrix for W2
            # A * W2_flat = b
            # b = 2 * ones(num_eqs)
            b = np.ones(num_eqs) * 2
            
            # A is num_eqs x (p*n*rank)
            # We can build A efficiently
            # Row e corresponds to (i, j, k)
            # Col u corresponds to (k', i', c) -> W2[k'*n+i', c]
            # But W2 depends on k and i from the equation.
            # In the sum, W index is fixed by (k, i) of the equation.
            # So for equation (i, j, k), the unknowns are W2[k*n+i, c] for c in 0..rank-1.
            # The coefficient for W2[k*n+i, c] is U2[i*m+j, c] * V2[j*p+k, c].
            
            # We can build A row by row.
            A = np.zeros((num_eqs, p * n * rank))
            
            for e_idx, (i, j, k) in enumerate(eq_indices):
                u_idx = i * m + j
                v_idx = j * p + k
                w_row_base = (k * n + i) * rank
                
                # Coefficients for this row
                coeffs = U2[u_idx, :] * V2[v_idx, :]
                
                # Non-zero coeffs
                nz = np.nonzero(coeffs)[0]
                if len(nz) > 0:
                    A[e_idx, w_row_base + nz] = coeffs[nz]
            
            # Solve
            try:
                W2_flat, residuals, rank_A, s = scipy.linalg.lstsq(A, b)
            except Exception:
                continue
                
            # Check if W2_flat is close to integers
            W2_rounded = np.round(W2_flat)
            error = np.max(np.abs(W2_flat - W2_rounded))
            
            if error < 1e-6:
                # Verify exactly
                valid = True
                W2 = W2_rounded.reshape((p*n, rank))
                for e_idx, (i, j, k) in enumerate(eq_indices):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    s_val = np.dot(U2[u_idx, :], V2[v_idx, :] * W2[w_idx, :])
                    if s_val != 2:
                        valid = False
                        break
                if valid:
                    # Found a valid decomposition
                    # Convert to floats (divide by 2)
                    U = U2 / 2.0
                    V = V2 / 2.0
                    W = W2 / 2.0
                    
                    result = {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
                    return result
            
    return best_result
