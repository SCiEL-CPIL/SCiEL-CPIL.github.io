# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 4
    dim_U = n * m  # 6
    dim_V = m * p  # 12
    dim_W = p * n  # 8
    
    # Schoolbook solution as fallback
    R_school = n * m * p
    U_school = [[0] * R_school for _ in range(n * m)]
    V_school = [[0] * R_school for _ in range(m * p)]
    W_school = [[0] * R_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
                
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = R_school

    # Attempt to find Rank 15
    # Using a fast local search with NumPy acceleration
    target_rank = 15
    start_time = time.time()
    time_limit = 100  # Leave 20s for safety

    def check_solution_np(U, V, W, n, m, p, R):
        """Fast check using NumPy."""
        U_np = np.array(U)
        V_np = np.array(V)
        W_np = np.array(W)
        
        # We need: sum_r W[k*n+i, r] U[i1*m+j1, r] V[j2*p+k2, r] = delta
        # Reshape U to (n, m, R), V to (m, p, R), W to (p, n, R)
        U_reshaped = U_np.reshape(n, m, R)
        V_reshaped = V_np.reshape(m, p, R)
        W_reshaped = W_np.reshape(p, n, R)
        
        # Compute the tensor T[i, k, i1, j1, j2, k2]
        # T[i, k, i1, j1, j2, k2] = sum_r W_reshaped[k, i, r] U_reshaped[i1, j1, r] V_reshaped[j2, k2, r]
        
        # This is a contraction over r.
        # We can check specific slices.
        # For fixed i, k, i1, j1, j2, k2:
        # We need to check if the sum is 1 if i==i1, k==k2, j1==j2, else 0.
        
        # Efficient check:
        # Construct the target tensor T_target (sparse)
        # And compute T_computed for the non-zero parts of T_target?
        # No, we must ensure off-diagonals are 0.
        
        # Let's compute the full contraction for a few random indices? No, must be exact.
        # The tensor size is 2*4*2*3*3*4 = 576.
        # We can compute all 576 values using broadcasting.
        
        # U_reshaped: (2, 3, R)
        # V_reshaped: (3, 4, R)
        # W_reshaped: (4, 2, R)
        
        # T[i, k, i1, j1, j2, k2] = sum_r W[k, i, r] U[i1, j1, r] V[j2, k2, r]
        
        # We can form this as:
        # M1[i1, j1, k2, j2, r] = U[i1, j1, r] * V[j2, k2, r]  -- Note V indices are (j, k)
        # Actually V is (m, p) so V[j2, k2, r].
        # M1 = np.einsum('ijr,jkr->ijkjr', U_reshaped, V_reshaped) # Shape (n, m, p, m, R)
        # T[i, k, i1, j1, j2, k2] = sum_r W[k, i, r] * M1[i1, j1, k2, j2, r]
        # T = np.einsum('kir,ijkjr->ikijjk', W_reshaped, M1) # Shape (p, n, n, m, m, p)
        
        try:
            M1 = np.einsum('ijr,jkr->ijkjr', U_reshaped, V_reshaped)
            T = np.einsum('kir,ijkjr->ikijjk', W_reshaped, M1)
        except Exception:
            return False

        # Check T
        # T[k, i, i1, j1, j2, k2] should be 1 if k==k2, i==i1, j1==j2, else 0.
        # Note: My einsum output order is 'ikijjk' -> indices (k, i, i1, j1, j2, k2)
        # Wait, W is (p, n, R) -> indices (k, i, r).
        # U is (n, m, R) -> indices (i1, j1, r).
        # V is (m, p, R) -> indices (j2, k2, r).
        
        # Let's re-verify the einsum string.
        # U: i1, j1, r
        # V: j2, k2, r
        # W: k, i, r
        # M1 = U * V -> i1, j1, j2, k2, r
        # T = W * M1 -> k, i, i1, j1, j2, k2
        
        # Target:
        # T[k, i, i1, j1, j2, k2] = 1 if k==k2 and i==i1 and j1==j2 else 0.
        
        # Create target tensor
        T_target = np.zeros((p, n, n, m, m, p))
        for k in range(p):
            for i in range(n):
                for j in range(m):
                    T_target[k, i, i, j, j, k] = 1.0
        
        if not np.allclose(T, T_target, atol=1e-9):
            return False
        return True

    def solve_W_np(U, V, n, m, p, R):
        """Solve for W given U, V using NumPy."""
        U_np = np.array(U)
        V_np = np.array(V)
        
        U_reshaped = U_np.reshape(n, m, R)
        V_reshaped = V_np.reshape(m, p, R)
        
        # M1[i1, j1, j2, k2, r] = U[i1, j1, r] * V[j2, k2, r]
        M1 = np.einsum('ijr,jkr->ijkjr', U_reshaped, V_reshaped)
        
        # We need W[k, i, r] such that:
        # sum_r W[k, i, r] M1[i1, j1, j2, k2, r] = T_target[k, i, i1, j1, j2, k2]
        
        # Flatten M1 and T_target to form a linear system A W_flat = b
        # A[idx, r] = M1[flattened_idx, r]
        # b[idx] = T_target[flattened_idx]
        
        # Number of equations: p*n*n*m*m*p
        num_eq = p * n * n * m * m * p
        # Number of unknowns: p*n*R
        
        # We can solve this by iterating over (k, i) since the system is block-diagonal?
        # No, W[k, i, r] appears in equations for all i1, j1, j2, k2.
        # But for a fixed (k, i), the unknowns are W[k, i, 0..R-1].
        # The equations involving W[k, i, *] are those where the target index is (k, i, i1, j1, j2, k2).
        # There are n*m*m*p such equations for each (k, i).
        # So we have (p*n) independent systems of size (n*m*m*p) x R.
        
        W_flat = np.zeros((p * n * R))
        
        for k in range(p):
            for i in range(n):
                # Extract equations for this (k, i)
                # Indices in T_target: [k, i, i1, j1, j2, k2]
                # Corresponding M1 indices: [i1, j1, j2, k2, r]
                
                # Build A_block (num_eq_block x R)
                # num_eq_block = n * m * m * p
                A_block = np.zeros((n * m * m * p, R))
                b_block = np.zeros(n * m * m * p)
                
                idx = 0
                for i1 in range(n):
                    for j1 in range(m):
                        for j2 in range(m):
                            for k2 in range(p):
                                # M1[i1, j1, j2, k2, :]
                                A_block[idx, :] = M1[i1, j1, j2, k2, :]
                                if i == i1 and k == k2 and j1 == j2:
                                    b_block[idx] = 1.0
                                else:
                                    b_block[idx] = 0.0
                                idx += 1
                
                # Solve least squares
                W_block, _, _, _ = np.linalg.lstsq(A_block, b_block, rcond=None)
                W_flat[k * n * R + i * R : k * n * R + i * R + R] = W_block
        
        # Reshape W_flat to (p, n, R)
        W = W_flat.reshape(p, n, R)
        return W.tolist()

    def search_rank_fast(R, n, m, p, max_iters=2000, time_limit=90):
        start_time = time.time()
        for iter in range(max_iters):
            if time.time() - start_time > time_limit:
                break
                
            # Random U, V
            U = [[random.randint(-2, 2) for _ in range(R)] for _ in range(n * m)]
            V = [[random.randint(-2, 2) for _ in range(R)] for _ in range(m * p)]
            
            W = solve_W_np(U, V, n, m, p, R)
            
            # Snap to half-integers
            W_snapped = [[0] * R for _ in range(p * n)]
            for i in range(p * n):
                for r in range(R):
                    W_snapped[i][r] = round(2 * W[i][r]) / 2.0
            
            if check_solution_np(U, V, W_snapped, n, m, p, R):
                return U, V, W_snapped
                
        return None

    res = search_rank_fast(target_rank, n, m, p, max_iters=1000, time_limit=100)
    if res:
        U, V, W = res
        best_result = {"U": U, "V": V, "W": W}
        best_rank = target_rank

    return best_result

# EVOLVE-BLOCK-END
