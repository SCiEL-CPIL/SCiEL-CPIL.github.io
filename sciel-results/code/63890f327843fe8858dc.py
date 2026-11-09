import time
import numpy as np

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Known rank 17 solution
    rank_17 = 17
    U17 = np.zeros((n * m, rank_17))
    V17 = np.zeros((m * p, rank_17))
    W17 = np.zeros((p * n, rank_17))
    
    # Term 1
    U17[0, 0] = 1; U17[4, 0] = 1
    V17[0, 0] = 1; V17[4, 0] = 1
    W17[0, 0] = 1; W17[3, 0] = 1
    
    # Term 2
    U17[3, 1] = 1; U17[4, 1] = 1
    V17[0, 1] = 1
    W17[1, 1] = 1; W17[3, 1] = -1
    
    # Term 3
    U17[0, 2] = 1
    V17[1, 2] = 1; V17[4, 2] = -1
    W17[2, 2] = 1; W17[3, 2] = 1
    
    # Term 4
    U17[4, 3] = 1
    V17[3, 3] = 1; V17[0, 3] = -1
    W17[0, 3] = 1; W17[1, 3] = 1
    
    # Term 5
    U17[0, 4] = 1; U17[1, 4] = 1
    V17[4, 4] = 1
    W17[0, 4] = -1; W17[2, 4] = 1
    
    # Term 6
    U17[3, 5] = 1; U17[0, 5] = -1
    V17[0, 5] = 1; V17[1, 5] = 1
    W17[3, 5] = 1
    
    # Term 7
    U17[1, 6] = 1; U17[4, 6] = -1
    V17[3, 6] = 1; V17[4, 6] = 1
    W17[0, 6] = 1
    
    # Term 8
    U17[0, 7] = 1
    V17[2, 7] = 1
    W17[4, 7] = 1
    
    # Term 9
    U17[1, 8] = 1
    V17[5, 8] = 1
    W17[4, 8] = 1
    
    # Term 10
    U17[3, 9] = 1
    V17[2, 9] = 1
    W17[5, 9] = 1
    
    # Term 11
    U17[4, 10] = 1
    V17[5, 10] = 1
    W17[5, 10] = 1
    
    # Term 12
    U17[2, 11] = 1
    V17[6, 11] = 1
    W17[0, 11] = 1
    
    # Term 13
    U17[2, 12] = 1
    V17[7, 12] = 1
    W17[2, 12] = 1
    
    # Term 14
    U17[2, 13] = 1
    V17[8, 13] = 1
    W17[4, 13] = 1
    
    # Term 15
    U17[5, 14] = 1
    V17[6, 14] = 1
    W17[1, 14] = 1
    
    # Term 16
    U17[5, 15] = 1
    V17[7, 15] = 1
    W17[3, 15] = 1
    
    # Term 17
    U17[5, 16] = 1
    V17[8, 16] = 1
    W17[5, 16] = 1

    def to_list(mat):
        return mat.tolist()
    
    def verify_solution(U, V, W):
        """Verify that the decomposition exactly reproduces the tensor."""
        R = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i * m + j
                    v = j * p + k
                    w_target = k * n + i
                    # Compute contribution to all w positions
                    contrib = np.zeros(p * n)
                    for r in range(R):
                        uv = U[u, r] * V[v, r]
                        if uv != 0:
                            contrib += uv * W[:, r]
                    # Check that only w_target has value 1
                    for ww in range(p * n):
                        expected = 1.0 if ww == w_target else 0.0
                        if abs(contrib[ww] - expected) > 1e-9:
                            return False
        return True
    
    # Try to find rank 16 solution
    # Strategy: Use the structure from rank 17 and try to merge terms
    # The 7 "Strassen-like" terms (0-6) handle the 2x2 subblock
    # Terms 7-16 handle the remaining entries
    
    # Approach 1: Try to find a rank-16 solution by searching over sparse patterns
    # For rank 16, we have 16 terms. Each term has support on U, V, W.
    # The tensor has 18 non-zero entries.
    
    # Let's try a more directed search:
    # Fix the structure of U and V (sparse, entries in {-1, 0, 1})
    # Then solve for W as a linear system
    
    deadline = start_time + 110.0  # Leave 40s safety margin
    
    R = 16
    np.random.seed(42)
    
    # Precompute the tensor structure
    # For each (u, v), the target w is determined
    tensor_entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                tensor_entries.append((u, v, w))
    
    # Build the system: for each (u,v,w) with T[u,v,w]=1,
    # sum_r U[u,r] * V[v,r] * W[w,r] = 1
    # And for all other (u,v,w'), sum_r U[u,r] * V[v,r] * W[w',r] = 0
    
    # This is a system of 18 * 6 = 108 equations (but many are redundant)
    # Actually, for each (u,v), we have 6 equations (one for each w), but only 1 is =1, rest are =0
    # Total: 18 * 6 = 108 equations, but the 18 "target" ones are =1, the rest 18*5=90 are =0
    # Unknowns: 6 * 16 = 96 (W entries)
    
    # We can write this as: A * W_flat = b
    # where A has shape (108, 96)
    
    # But U and V are also unknowns! This is a bilinear system.
    # Instead, let's fix U and V and solve for W.
    
    # Better approach: iterate over random sparse U, V and solve for W
    
    # The key constraint: for each (u,v), the vector [U[u,:] * V[v,:]] must be such that
    # when multiplied with W, it gives the correct pattern.
    
    # Let's try: for random sparse U, V with entries in {-1, 0, 1},
    # solve the linear system for W and check if W is integer.
    
    num_attempts = 0
    while time.time() < deadline and num_attempts < 50000:
        num_attempts += 1
        
        # Random sparse U, V
        # Density: each entry is 0 with prob ~0.7, ±1 with prob ~0.15 each
        U = np.zeros((n * m, R), dtype=np.int32)
        V = np.zeros((m * p, R), dtype=np.int32)
        
        for r in range(R):
            # Each column of U has 1-3 non-zero entries
            num_nonzero_u = np.random.randint(1, 4)
            indices = np.random.choice(n * m, size=num_nonzero_u, replace=False)
            for idx in indices:
                U[idx, r] = np.random.choice([-1, 1])
            
            # Each column of V has 1-3 non-zero entries
            num_nonzero_v = np.random.randint(1, 4)
            indices = np.random.choice(m * p, size=num_nonzero_v, replace=False)
            for idx in indices:
                V[idx, r] = np.random.choice([-1, 1])
        
        # Build linear system for W
        # Equations: for each (u, v, w), sum_r U[u,r]*V[v,r]*W[w,r] = T[u,v,w]
        # Total equations: n*m * p*n = 18 * 6 = 108
        # Unknowns: p*n * R = 6 * 16 = 96
        
        A_mat = np.zeros((108, 96), dtype=np.float64)
        b_vec = np.zeros(108, dtype=np.float64)
        
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i * m + j
                    v = j * p + k
                    for w in range(p * n):
                        # Coefficient for W[w, r] is U[u,r] * V[v,r]
                        for r in range(R):
                            coeff = U[u, r] * V[v, r]
                            if coeff != 0:
                                A_mat[eq_idx, w * R + r] = coeff
                        # Target value
                        w_target = k * n + i
                        if w == w_target:
                            b_vec[eq_idx] = 1.0
                        else:
                            b_vec[eq_idx] = 0.0
                        eq_idx += 1
        
        # Solve least squares
        try:
            W_sol, residuals, rank_sol, s = np.linalg.lstsq(A_mat, b_vec, rcond=None)
        except:
            continue
        
        W_sol = W_sol.reshape((p * n, R))
        
        # Check if close to integer
        W_int = np.round(W_sol)
        if not np.allclose(W_sol, W_int, atol=1e-8):
            continue
        
        # Verify the solution
        if verify_solution(U, V, W_int):
            return {"U": to_list(U), "V": to_list(V), "W": to_list(W_int)}
    
    # If rank 16 not found, return rank 17
    return {"U": to_list(U17), "V": to_list(V17), "W": to_list(W17)}
