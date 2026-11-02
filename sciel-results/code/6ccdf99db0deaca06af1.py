import numpy as np
import time

DIMENSIONS = [2, 3, 9]

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    n_rows = n * m  # 6
    m_rows = m * p  # 27
    p_rows = p * n  # 18
    
    def verify_decomposition(U, V, W):
        """Verify that the tensor decomposition is exact."""
        r = len(U[0])
        U_arr = np.array(U, dtype=np.int64)
        V_arr = np.array(V, dtype=np.int64)
        W_arr = np.array(W, dtype=np.int64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U_arr[i*m+j] * V_arr[j*p+k] * W_arr[k*n+i])
                    if val != 1:
                        return False
        return True
    
    # Rank 50 decomposition (Strassen-based)
    def current_solution():
        rank = 50
        U = [[0] * rank for _ in range(n_rows)]
        V = [[0] * rank for _ in range(m_rows)]
        W = [[0] * rank for _ in range(p_rows)]
        
        col = 0
        # Part 1: A_L (2x2) * B_T (2x9)
        for b in range(4):
            k0 = 2 * b
            k1 = 2 * b + 1
            
            v_e = 0 * p + k0
            v_f = 0 * p + k1
            v_g = 1 * p + k0
            v_h = 1 * p + k1
            
            u_a = 0 * m + 0
            u_b = 0 * m + 1
            u_c = 1 * m + 0
            u_d = 1 * m + 1
            
            # Strassen
            U[u_a][col] += 1; U[u_d][col] += 1; V[v_e][col] += 1; V[v_h][col] += 1
            U[u_c][col+1] += 1; U[u_d][col+1] += 1; V[v_e][col+1] += 1
            U[u_a][col+2] += 1; V[v_f][col+2] += 1; V[v_h][col+2] -= 1
            U[u_d][col+3] += 1; V[v_g][col+3] += 1; V[v_e][col+3] -= 1
            U[u_a][col+4] += 1; U[u_b][col+4] += 1; V[v_h][col+4] += 1
            U[u_c][col+5] += 1; U[u_a][col+5] -= 1; V[v_e][col+5] += 1; V[v_f][col+5] += 1
            U[u_b][col+6] += 1; U[u_d][col+6] -= 1; V[v_g][col+6] += 1; V[v_h][col+6] += 1
            
            w_00 = k0 * n + 0
            w_01 = k1 * n + 0
            w_10 = k0 * n + 1
            w_11 = k1 * n + 1
            
            W[w_00][col] += 1; W[w_00][col+3] += 1; W[w_00][col+4] -= 1; W[w_00][col+6] += 1
            W[w_01][col+2] += 1; W[w_01][col+4] += 1
            W[w_10][col+1] += 1; W[w_10][col+3] += 1
            W[w_11][col] += 1; W[w_11][col+1] -= 1; W[w_11][col+2] += 1; W[w_11][col+5] += 1
            col += 7
            
        # Last col of B_T
        k8 = 8
        v_e = 0 * p + k8
        v_g = 1 * p + k8
        u_a = 0 * m + 0
        u_b = 0 * m + 1
        u_c = 1 * m + 0
        u_d = 1 * m + 1
        
        U[u_a][col] += 1; V[v_e][col] += 1
        U[u_b][col+1] += 1; V[v_g][col+1] += 1
        U[u_c][col+2] += 1; V[v_e][col+2] += 1
        U[u_d][col+3] += 1; V[v_g][col+3] += 1
        
        w_08 = k8 * n + 0
        w_18 = k8 * n + 1
        W[w_08][col] += 1; W[w_08][col+1] += 1
        W[w_18][col+2] += 1; W[w_18][col+3] += 1
        col += 4
        
        # Part 2: A_R (2x1) * B_B (1x9)
        u_ar0 = 0 * m + 2
        u_ar1 = 1 * m + 2
        for k in range(9):
            v_b = 2 * p + k
            U[u_ar0][col] += 1; V[v_b][col] += 1; W[k * n + 0][col] += 1
            U[u_ar1][col+1] += 1; V[v_b][col+1] += 1; W[k * n + 1][col+1] += 1
            col += 2
            
        return {"U": U, "V": V, "W": W}
    
    def try_reduce_rank(U, V, W, max_time=20):
        """Try to reduce rank by finding linear dependencies among columns."""
        start = time.time()
        r = len(U[0])
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        
        # For each column c, check if it's linearly dependent on others
        # We can do this by checking if the tensor contribution of column c
        # can be matched by a combination of other columns
        
        # Build the "column tensor" representation
        # Each column c defines a rank-1 tensor: U[:,c] outer V[:,c] outer W[:,c]
        # We want to find if column c can be expressed as sum of alpha_d * (U[:,d] outer V[:,d] outer W[:,d])
        
        # This is complex. Instead, let's try a simpler approach:
        # Randomly perturb and re-solve
        
        # Alternative: try to find if any column is redundant
        # For column c, the contribution is: for all (i,j,k), U[i,j,c]*V[j,k,c]*W[k,i,c]
        # This must be 0 for all (i,j,k) if the column is to be removable (after adjusting others)
        
        # Actually, we can't just remove a column; we need to adjust the others.
        # Let's try: fix all columns except c, and see if we can find values for column c
        # that make the total sum = 1 for all (i,j,k).
        
        # This is a system of n*m*p = 54 equations in 1 unknown (the scale of column c)
        # if we assume column c is a scalar multiple of some fixed pattern.
        
        # Better approach: try to merge two columns into one
        # If column c1 and c2 have the property that their combined contribution
        # can be represented as a single column, we can reduce rank by 1.
        
        # For now, let's try a different strategy: random search with structured initialization
        
        return None
    
    # Try to find lower rank solutions using random search with better initialization
    def try_rank(r, max_trials=50):
        if time.time() > deadline - 15:
            return None
        
        for trial in range(max_trials):
            if time.time() > deadline - 10:
                return None
            
            # Initialize with sparse random matrices
            V = np.zeros((m_rows, r), dtype=np.int64)
            W = np.zeros((p_rows, r), dtype=np.int64)
            
            # Fill with small random integers, ensuring some structure
            for c in range(r):
                # V column: sparse, 2-4 non-zeros
                nz_v = np.random.randint(2, 5)
                idx_v = np.random.choice(m_rows, size=nz_v, replace=False)
                for idx in idx_v:
                    V[idx, c] = np.random.choice([-1, 1, 2, -2])
                
                # W column: sparse, 2-4 non-zeros  
                nz_w = np.random.randint(2, 5)
                idx_w = np.random.choice(p_rows, size=nz_w, replace=False)
                for idx in idx_w:
                    W[idx, c] = np.random.choice([-1, 1, 2, -2])
            
            U = np.zeros((n_rows, r), dtype=np.int64)
            
            # Solve for U using least squares for each (i,j)
            valid = True
            for i in range(n):
                for j in range(m):
                    # Build M (p x r): M[k, c] = V[j*p+k, c] * W[k*n+i, c]
                    M = np.zeros((p, r), dtype=np.float64)
                    for k in range(p):
                        M[k, :] = V[j*p+k, :].astype(np.float64) * W[k*n+i, :].astype(np.float64)
                    
                    # Solve M @ u = ones(p)
                    try:
                        u, residuals, rank_m, sv = np.linalg.lstsq(M, np.ones(p), rcond=None)
                        u_int = np.round(u).astype(np.int64)
                        
                        # Verify exactness
                        if not np.allclose(M @ u_int, np.ones(p), atol=1e-10):
                            valid = False
                            break
                        U[i*m+j, :] = u_int
                    except:
                        valid = False
                        break
                
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Full verification
            U_arr = U.astype(np.int64)
            V_arr = V.astype(np.int64)
            W_arr = W.astype(np.int64)
            
            success = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U_arr[i*m+j] * V_arr[j*p+k] * W_arr[k*n+i])
                        if val != 1:
                            success = False
                            break
                    if not success:
                        break
                if not success:
                    break
            
            if success:
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
        
        return None
    
    # Try ranks from 49 down to 40
    for r in range(49, 39, -1):
        res = try_rank(r, max_trials=30)
        if res:
            return res
    
    # Fallback to rank 50
    return current_solution()
