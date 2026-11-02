import numpy as np
import time

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140.0
    
    # Schoolbook decomposition (Rank 48)
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
                
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school
    
    # Try to find a decomposition for M_3,3,3 with rank < 27
    # If found, we can construct M_3,4,4 with rank R_333 + 21
    # R_333 + 21 < 48 => R_333 < 27
    
    def find_m333_decomposition(target_rank, time_limit):
        # Returns (U, V, W) for M_3,3,3 of shape (9, R), (9, R), (9, R)
        # or None
        n3, m3, p3 = 3, 3, 3
        T = np.zeros((n3*m3, m3*p3, p3*n3))
        for i in range(n3):
            for j in range(m3):
                for k in range(p3):
                    T[i*m3+j, j*p3+k, k*n3+i] = 1.0
        
        # Flatten T to vector
        T_flat = T.flatten()
        
        def error_func(params):
            # params: vector of length 3 * 9 * R
            R = target_rank
            U = params[0:9*R].reshape(9, R)
            V = params[9*R:18*R].reshape(9, R)
            W = params[18*R:27*R].reshape(9, R)
            
            # Compute sum U[:,r] outer V[:,r] outer W[:,r]
            # This is a 3D tensor
            # We can compute the contraction
            # T_est[i,j,k] = sum_r U[i,j] V[j,k] W[k,i]
            # Note: indices are flattened
            # U index: i*3+j
            # V index: j*3+k
            # W index: k*3+i
            
            # Efficient computation:
            # For each r, we add U_r outer V_r outer W_r
            # This is slow in Python loops.
            # We can use einsum?
            # T_est = np.einsum('ir,jr,kr->ijk', U, V, W) ? No, indices are coupled.
            
            # Let's reshape U, V, W to 3D
            U3 = U.reshape(3, 3, R)
            V3 = V.reshape(3, 3, R)
            W3 = W.reshape(3, 3, R)
            
            # T_est[i,j,k] = sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
            # This is not a standard einsum because of the cyclic indices.
            # We can loop over i, j, k? 27 iterations.
            T_est = np.zeros((3,3,3))
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        # sum_r U[i,j,r] V[j,k,r] W[k,i,r]
                        # Extract vectors
                        u_vec = U3[i,j,:]
                        v_vec = V3[j,k,:]
                        w_vec = W3[k,i,:]
                        T_est[i,j,k] = np.dot(u_vec, v_vec * w_vec)
                        
            return T_est.flatten() - T_flat

        # Try a few random starts
        np.random.seed(42)
        for attempt in range(50):
            if time.time() > start_time + time_limit:
                break
            
            R = target_rank
            # Initialize with small random values
            params = np.random.randn(27 * R) * 0.1
            
            # Use scipy.optimize
            from scipy.optimize import least_squares
            try:
                res = least_squares(error_func, params, method='lm', max_nfev=1000)
                if res.success and res.cost < 1e-6:
                    # Try to snap to half-integers
                    U = res.x[0:9*R].reshape(9, R)
                    V = res.x[9*R:18*R].reshape(9, R)
                    W = res.x[18*R:27*R].reshape(9, R)
                    
                    U_snap = np.round(U * 2) / 2
                    V_snap = np.round(V * 2) / 2
                    W_snap = np.round(W * 2) / 2
                    
                    # Verify
                    err = np.linalg.norm(error_func(np.concatenate([U_snap.flatten(), V_snap.flatten(), W_snap.flatten()])))
                    if err < 1e-9:
                        return U_snap, V_snap, W_snap
            except Exception:
                pass
                
        return None

    # Try ranks for M_3,3,3
    # We want R_333 + 21 < 48 => R_333 <= 26
    for R333 in [25, 26]:
        if time.time() > deadline - 20:
            break
            
        print(f"Trying M_3,3,3 rank {R333}")
        result = find_m333_decomposition(R333, 10.0)
        if result is not None:
            U333, V333, W333 = result
            # Construct M_3,4,4
            # A = [A1 (3x3), A2 (3x1)]
            # B = [[B11 (3x3), B12 (3x1)], [B21 (1x3), B22 (1x1)]]
            # C = A1 B11 + A1 B12 + A2 B21 + A2 B22
            
            # U for M_3,4,4: shape (12, R_total)
            # V for M_3,4,4: shape (16, R_total)
            # W for M_3,4,4: shape (12, R_total)
            
            R_total = R333 + 9 + 9 + 3
            
            U_full = np.zeros((12, R_total))
            V_full = np.zeros((16, R_total))
            W_full = np.zeros((12, R_total))
            
            # Part 1: M_3,3,3 (A1 B11)
            # U333 rows: i*3+j for i,j in 0..2
            # V333 rows: j*3+k for j,k in 0..2
            # W333 rows: k*3+i for k,i in 0..2
            
            # In M_3,4,4:
            # U rows: i*4+j
            # V rows: j*4+k
            # W rows: k*3+i
            
            # Map U333 to U_full
            for i in range(3):
                for j in range(3):
                    U_full[i*4+j, 0:R333] = U333[i*3+j, :]
                    
            # Map V333 to V_full
            for j in range(3):
                for k in range(3):
                    V_full[j*4+k, 0:R333] = V333[j*3+k, :]
                    
            # Map W333 to W_full
            for k in range(3):
                for i in range(3):
                    W_full[k*3+i, 0:R333] = W333[k*3+i, :]
            
            # Part 2: M_3,3,1 (A1 B12)
            # A1 is 3x3, B12 is 3x1
            # C_part is 3x1
            # Rank 9
            # U: rows i*4+j (i,j in 0..2)
            # V: rows j*4+3 (j in 0..2)  (B12 column index is 3)
            # W: rows k*3+0 (k in 0..2)  (C row index is k, col index is 0? No, C is 3x4. B12 is col 3. So C col 3.)
            # Wait, C = A B.
            # C[:, 3] = A1 B12 + A2 B22.
            # So W index for C col 3 is k*3+3? No, W index is k*n+i?
            # W index is k*n+i. Here n=3. So k*3+i.
            # For C col 3, the index is... wait.
            # C is 3x4. Indices (i, c) where i in 0..2, c in 0..3.
            # W index is k*n+i? No.
            # The tensor T[i*m+j, j*p+k, k*n+i].
            # i is row of A (0..2).
            # j is col of A / row of B (0..3).
            # k is col of B (0..3).
            # W index is k*n+i.
            # For B12, j=0..2, k=3.
            # So V index is j*4+3.
            # W index is 3*3+i = 9+i.
            
            # So for Part 2:
            # U: i*4+j
            # V: j*4+3
            # W: 9+i
            
            for r in range(9):
                i = r // 3
                j = r % 3
                U_full[i*4+j, R333 + r] = 1
                V_full[j*4+3, R333 + r] = 1
                W_full[9+i, R333 + r] = 1
            
            # Part 3: M_3,1,3 (A2 B21)
            # A2 is 3x1, B21 is 1x3
            # C_part is 3x3
            # Rank 9
            # U: i*4+3 (j=3)
            # V: 3*4+k (j=3)
            # W: k*3+i
            
            for r in range(9):
                i = r // 3
                k = r % 3
                U_full[i*4+3, R333 + 9 + r] = 1
                V_full[3*4+k, R333 + 9 + r] = 1
                W_full[k*3+i, R333 + 9 + r] = 1
            
            # Part 4: M_3,1,1 (A2 B22)
            # A2 is 3x1, B22 is 1x1
            # C_part is 3x1
            # Rank 3
            # U: i*4+3
            # V: 3*4+3
            # W: 9+i
            
            for r in range(3):
                i = r
                U_full[i*4+3, R333 + 18 + r] = 1
                V_full[3*4+3, R333 + 18 + r] = 1
                W_full[9+i, R333 + 18 + r] = 1
            
            # Verify the full decomposition
            # T_full[i*4+j, j*4+k, k*3+i] should be 1 if i==k else 0?
            # No, T[i*m+j, j*p+k, k*n+i] = 1.
            # Here n=3, m=4, p=4.
            # T[i*4+j, j*4+k, k*3+i] = 1.
            
            # Check
            U_check = U_full
            V_check = V_full
            W_check = W_full
            is_valid = True
            for i in range(3):
                for j in range(4):
                    for k in range(4):
                        val = 0
                        for r in range(R_total):
                            val += U_check[i*4+j, r] * V_check[j*4+k, r] * W_check[k*3+i, r]
                        expected = 1.0 if i == k else 0.0
                        if abs(val - expected) > 1e-9:
                            is_valid = False
                            break
                if not is_valid:
                    break
            if not is_valid:
                break
            if not is_valid:
                break
            if not is_valid:
                break
                
            if is_valid:
                best_U, best_V, best_W = U_full, V_full, W_full
                best_rank = R_total
                break
                
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
