import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 10s safety margin

    n, m, p = 3, 4, 8

    # Schoolbook rank
    rank_sb = n * m * p  # 96
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, idx] = 1
                V_sb[j * p + k, idx] = 1
                W_sb[k * n + i, idx] = 1
                idx += 1
                
    best = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = rank_sb

    def verify(U, V, W, rank, tol=1e-9):
        U_np = np.array(U)
        V_np = np.array(V)
        W_np = np.array(W)
        
        # Reshape
        U_3d = U_np.reshape(n, m, rank)
        V_3d = V_np.reshape(m, p, rank)
        W_3d = W_np.reshape(p, n, rank)
        
        rng = np.random.default_rng(12345)
        for _ in range(3):
            A = rng.standard_normal((n, m))
            B = rng.standard_normal((m, p))
            C = A @ B
            
            M = U_3d * A[:, :, np.newaxis]
            N = V_3d * B[:, :, np.newaxis]
            S = np.einsum('ijc,jkc->ikc', M, N)
            S_T = np.transpose(S, (1, 0, 2))
            C_dec = np.sum(S_T * W_3d, axis=2)
            
            if np.max(np.abs(C - C_dec)) > tol:
                return False
        return True

    def to_clean_list(mat_np):
        # Convert to list of lists with integers or half-integers
        result = []
        for row in mat_np:
            new_row = []
            for v in row:
                if abs(v - round(v)) < 1e-9:
                    new_row.append(int(round(v)))
                elif abs(v * 2 - round(v * 2)) < 1e-9:
                    new_row.append(round(v * 2) / 2)
                else:
                    new_row.append(v)
            result.append(new_row)
        return result

    # We want to find a rank r < 96
    # We can try to solve for U, V, W given a random W, or vice versa.
    # The bilinear map is: C = sum_c W_c * (U_c * A) * V_c
    # This is equivalent to finding a rank-r decomposition of the tensor T_ijk = sum_c U_ic V_jk W_ki
    # Wait, the indices are:
    # C[i, k] = sum_c W[k, i, c] * ( sum_j U[i, j, c] A[i, j] V[j, k, c] )
    # T[i, j, k] = sum_c U[i, j, c] V[j, k, c] W[k, i, c]
    # Here T[i, j, k] is 1 if we are doing standard multiplication (which it is).
    # So we are looking for a rank-r decomposition of the matrix multiplication tensor M_{n, m, p}.
    #
    # Strategy:
    # Fix U and V randomly? No, that's underdetermined.
    # Fix W?
    # If we fix U and V, the equation is linear in W.
    # T_flat = A_matrix @ W_flat
    # where A_matrix is constructed from U and V.
    # A_matrix[idx_ijk, idx_kic] = U[i, j, c] * V[j, k, c]
    # If we fix U and V, we can solve for W via least squares.
    # Then we check if W is "close to integer/half-integer" or if the residual is small.
    # The problem asks for integer/half-integer entries.
    # If we find a decomposition with real entries, can we round it?
    # Usually, if the decomposition is "tight" (rank is minimal or near minimal), the coefficients might be integers.
    # For example, Strassen's algorithm uses integers.
    # So we can try random U, V, solve for W, and check if W can be rounded to half-integers and still satisfy the equation.
    # Or better: Solve for U, V, W simultaneously? That's nonlinear.
    # Let's stick to: Fix U, V (random), solve for W.
    # The solution W is unique if A_matrix has full column rank (which is likely for random U, V if r is not too large).
    # If W has small values or structure, maybe we can find an integer solution?
    # Actually, if we find ANY solution (U, V, W) that works, it doesn't have to be integer.
    # BUT the task says: "integer/half-integer entries".
    # This is a strong constraint.
    # So we must find U, V, W with entries in {0, 0.5, 1, 1.5, ...}.
    # This suggests the rank is likely small enough that such a structure exists, OR the problem implies that if a real solution exists, we can transform it? No, "exactly decompose".
    # If the minimal rank requires irrational numbers, we can't do it?
    # However, for small matrices, minimal rank decompositions often exist with integers.
    # Let's assume such a decomposition exists with small integers.
    
    # We can use a search strategy:
    # 1. Try ranks r from 80 to 96.
    # 2. For each r, try random integer/half-integer U, V.
    #    Actually, generating random integer U, V is a huge space.
    #    Instead, let's generate random real U, V, solve for W.
    #    If W is close to half-integers, we round W to W'.
    #    Then we can re-solve for U, V? Or just check if (U, V, W') works?
    #    If we fix U, V and round W, the equation T = A(W) won't hold anymore unless A was designed to make W integer.
    
    # Better approach:
    # We know that for 2x2x2, rank 7 exists (Strassen) with integers.
    # For 3x4x8, maybe we can use a block decomposition?
    # Or just brute force the "fix U, V, solve W" and hope W comes out nice?
    # If W is not nice, the residual of the rounded W will be large.
    
    # Let's try a smarter initialization.
    # Start with schoolbook and try to merge columns?
    
    # Or, use the fact that we can permute indices.
    
    # Let's try the "Fix U, V, solve W" method with a twist:
    # We will search for U, V such that the resulting W is close to half-integers.
    # This is a non-convex optimization.
    
    # Alternative:
    # If we find a real solution (U, V, W) with rank r, we can check if it's valid.
    # The constraint "integer/half-integer" is for the final output.
    # If we find a real solution that is NOT integer, we fail.
    # So we MUST find an integer solution.
    
    # Is it possible that NO integer solution exists for the minimal rank?
    # If so, the score will be determined by the smallest r for which an integer solution exists.
    
    # Let's assume that for this size, an integer solution exists for the minimal rank.
    
    # Let's try to construct U, V, W with entries in {-1, 0, 1}.
    # This is a SAT-like problem but we can use linear algebra.
    
    # Method:
    # 1. Generate random sparse U, V with entries in {-1, 0, 1}.
    # 2. Solve for W.
    # 3. Check if W is in {-1, 0, 1} (or half-integers).
    # 4. If yes, verify.
    
    # This might work for small ranks.
    
    def try_find_decomposition(r, max_attempts=200):
        rng = np.random.default_rng(r)
        A_mat = np.zeros((n * m * p, p * n * r))
        
        # Precompute indices for speed
        # A_mat[i*m*p + j*p + k, k*n + i + c*r] ... wait
        # T_flat index: i * m * p + j * p + k
        # W_flat index: (k * n + i) * r + c
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                return None
            
            # Generate sparse U, V
            # Density 0.3
            U_rand = (rng.random((n * m, r)) < 0.3).astype(float)
            V_rand = (rng.random((m * p, r)) < 0.3).astype(float)
            # Random signs
            U_rand[U_rand == 0] = 0
            U_rand[U_rand != 0] = rng.choice([-1, 1], size=np.sum(U_rand != 0))
            V_rand[V_rand == 0] = 0
            V_rand[V_rand != 0] = rng.choice([-1, 1], size=np.sum(V_rand != 0))
            
            # Build A_mat
            # This is slow to build from scratch every time.
            # But n*m*p = 96, p*n*r = 24*r.
            # For r=80, size is 96 x 1920.
            # Building it takes O(n*m*p*r) = 96*80 = 7680 ops. Fast enough.
            
            A_mat.fill(0)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        idx_t = i * m * p + j * p + k
                        for c in range(r):
                            idx_w = (k * n + i) * r + c
                            A_mat[idx_t, idx_w] = U_rand[i * m + j, c] * V_rand[j * p + k, c]
                            
            T_flat = np.ones(n * m * p)
            
            try:
                W_flat, residuals, rank_A, sv = np.linalg.lstsq(A_mat, T_flat, rcond=None)
            except:
                continue
                
            if len(residuals) == 0 or residuals[0] > 1e-9:
                continue
                
            W_rand = W_flat.reshape((p * n, r))
            
            # Check if W is close to half-integers
            W_rounded = np.round(W_rand * 2) / 2
            if np.max(np.abs(W_rand - W_rounded)) < 1e-6:
                # Check U, V are already integer
                # Verify
                U_list = U_rand.tolist()
                V_list = V_rand.tolist()
                W_list = to_clean_list(W_rounded)
                
                if verify(U_list, V_list, W_list, r):
                    return {"U": U_list, "V": V_list, "W": W_list}
            
        return None

    # Search ranks
    # Lower bound for 3x4x8?
    # 2x2x2 is 7.
    # 3x3x3 is 23.
    # 4x4x4 is 49?
    # Let's guess around 80-90.
    # Start from 80 up to 95.
    
    for r in range(80, 96):
        if time.time() > deadline - 10:
            break
            
        res = try_find_decomposition(r, max_attempts=100)
        if res:
            best = res
            best_rank = r
            break
            
    # Final cleaning
    def clean_matrix(mat):
        result = []
        for row in mat:
            new_row = []
            for v in row:
                if isinstance(v, (int, np.integer)):
                    new_row.append(int(v))
                elif isinstance(v, float):
                    if abs(v - round(v)) < 1e-9:
                        new_row.append(int(round(v)))
                    elif abs(v * 2 - round(v * 2)) < 1e-9:
                        new_row.append(round(v * 2) / 2)
                    else:
                        new_row.append(v)
                else:
                    new_row.append(v)
            result.append(new_row)
        return result

    U_final = clean_matrix(best["U"])
    V_final = clean_matrix(best["V"])
    W_final = clean_matrix(best["W"])
    
    return {"U": U_final, "V": V_final, "W": W_final}
