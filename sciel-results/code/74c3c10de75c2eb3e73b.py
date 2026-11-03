import numpy as np
import time

def construct():
    n, m, p = 4, 4, 9
    
    # Build the target tensor T[u,v,w]
    # T[u,v,w] = 1 if (u = i*m+j, v = j*p+k, w = k*n+i) for some i,j,k
    # T[u,v,w] = 0 otherwise
    
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
    
    # Schoolbook solution as fallback (rank = n*m*p = 144)
    U_sb = np.zeros((n*m, n*m*p), dtype=int)
    V_sb = np.zeros((m*p, n*m*p), dtype=int)
    W_sb = np.zeros((p*n, n*m*p), dtype=int)
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i*m + j
                v = j*p + k
                w = k*n + i
                U_sb[u, idx] = 1
                V_sb[v, idx] = 1
                W_sb[w, idx] = 1
                idx += 1
    
    best_solution = (U_sb, V_sb, W_sb)
    best_rank = n * m * p
    
    start_time = time.time()
    deadline = start_time + 120  # 120 second limit
    
    # Precompute all (u,v) pairs
    uv_pairs = [(u, v) for u in range(n*m) for v in range(m*p)]
    num_uv = len(uv_pairs)  # 16 * 36 = 576
    
    # For each w, precompute which (u,v) have T[u,v,w] = 1
    target_indices = {}
    for w in range(p*n):
        target_indices[w] = []
        for u in range(n*m):
            for v in range(m*p):
                if T[u, v, w] != 0:
                    target_indices[w].append((u, v))
    
    # Try different ranks
    # Lower bound considerations: 
    # For matrix multiplication C = AB where A is n×m and B is m×p,
    # the rank is at least max(n*m, m*p, p*n) = max(16, 36, 36) = 36
    # But in practice, much higher ranks are needed.
    # Let's try ranks from a reasonable lower bound up to 143
    
    # Use a smarter search: try several ranks and keep the best found
    ranks_to_try = list(range(40, 144))
    
    for r in ranks_to_try:
        if time.time() > deadline - 30:
            break
        
        # Number of restarts based on time remaining
        time_left = deadline - time.time()
        if time_left < 10:
            break
        
        # Estimate time per restart: 36 lstsq calls of size 576×r
        # Each lstsq is roughly O(576 * r^2) for small r
        # Let's be conservative and do fewer restarts for larger r
        if r < 60:
            max_restarts = int(time_left / 0.5)  # ~0.5s per restart
        elif r < 90:
            max_restarts = int(time_left / 1.0)
        else:
            max_restarts = int(time_left / 2.0)
        
        max_restarts = max(1, min(max_restarts, 500))
        
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
            
            # Generate random U and V with entries in {-1, 0, 1}
            # This helps with integer/half-integer solutions
            U = np.random.randint(-1, 2, (n*m, r), dtype=np.float64)
            V = np.random.randint(-1, 2, (m*p, r), dtype=np.float64)
            
            # For each w, solve for W[w,:]
            W = np.zeros((p*n, r), dtype=np.float64)
            valid = True
            
            for w in range(p*n):
                # Build the linear system A_w * W_w = b_w
                # A_w has one row per (u,v) pair
                # A_w[i, c] = U[u_i, c] * V[v_i, c]
                # b_w[i] = T[u_i, v_i, w]
                
                A_w = np.zeros((num_uv, r), dtype=np.float64)
                b_w = np.zeros(num_uv, dtype=np.float64)
                
                for idx, (u, v) in enumerate(uv_pairs):
                    A_w[idx, :] = U[u, :] * V[v, :]
                    b_w[idx] = T[u, v, w]
                
                # Solve via least squares
                try:
                    W_w, residuals, rank_w, sv = np.linalg.lstsq(A_w, b_w, rcond=None)
                except np.linalg.LinAlgError:
                    valid = False
                    break
                
                # Check if the solution is exact (residual is zero)
                # The residual from lstsq is the sum of squares of the residual vector
                if len(residuals) > 0:
                    if residuals[0] > 1e-10:
                        valid = False
                        break
                else:
                    # If rank < num_uv, check manually
                    residual = A_w @ W_w - b_w
                    if np.max(np.abs(residual)) > 1e-10:
                        valid = False
                        break
                
                W[w, :] = W_w
            
            if valid:
                # Found a valid solution!
                # Round to nearest half-integer for cleanliness
                W_rounded = np.round(W * 2) / 2
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                
                # Verify the rounded solution still works
                valid_rounded = True
                for u in range(n*m):
                    for v in range(m*p):
                        for w in range(p*n):
                            computed = np.dot(U_rounded[u, :] * V_rounded[v, :], W_rounded[w, :])
                            if abs(computed - T[u, v, w]) > 1e-10:
                                valid_rounded = False
                                break
                        if not valid_rounded:
                            break
                    if not valid_rounded:
                        break
                
                if valid_rounded:
                    best_solution = (U_rounded, V_rounded, W_rounded)
                    best_rank = r
                    # Found a solution at this rank, break out of restart loop
                    # and try lower ranks if any remain
                    break
                else:
                    # Use the unrounded solution
                    best_solution = (U, V, W)
                    best_rank = r
                    break
    
    U, V, W = best_solution
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
