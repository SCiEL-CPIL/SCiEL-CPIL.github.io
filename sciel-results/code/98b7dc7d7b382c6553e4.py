"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    deadline = time.time() + 100  # safety margin
    n, m, p = DIMENSIONS
    
    # Strassen terms for 2x2x2
    # Local indices 0..3 map to (0,0), (0,1), (1,0), (1,1)
    terms = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    
    def build_28_rank():
        rank = 28
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        def get_global_A(block, li, lj):
            i = li
            if block in [0, 2]:
                j = lj
            else:
                j = 2 + lj
            return i * 4 + j
            
        def get_global_B(block, li, lj):
            j_local = li
            r_local = lj
            if block == 0:
                j = j_local
                r = r_local
            elif block == 1:
                j = 2 + j_local
                r = r_local
            elif block == 2:
                j = j_local
                r = 2 + r_local
            else:
                j = 2 + j_local
                r = 2 + r_local
            return j * 4 + r
            
        def get_global_C(block, li, lj):
            i_local = li
            r_local = lj
            if block in [0, 1]:
                i = i_local
                r = r_local
            else:
                i = i_local
                r = 2 + r_local
            return r * 2 + i

        for b in range(4):
            for s in range(7):
                col = b * 7 + s
                u_coeffs, v_coeffs, w_coeffs = terms[s]
                
                for local_idx, coeff in u_coeffs.items():
                    li = local_idx // 2
                    lj = local_idx % 2
                    g_idx = get_global_A(b, li, lj)
                    U[g_idx][col] += coeff
                    
                for local_idx, coeff in v_coeffs.items():
                    li = local_idx // 2
                    lj = local_idx % 2
                    g_idx = get_global_B(b, li, lj)
                    V[g_idx][col] += coeff
                    
                for local_idx, coeff in w_coeffs.items():
                    li = local_idx // 2
                    lj = local_idx % 2
                    g_idx = get_global_C(b, li, lj)
                    W[g_idx][col] += coeff

        return U, V, W

    def verify_decomp(U, V, W):
        """Verify that the decomposition correctly computes matrix multiplication."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        rank = U_arr.shape[1]
        
        np.random.seed(12345)
        for trial in range(3):
            A_test = np.random.randn(n, m)
            B_test = np.random.randn(m, p)
            C_test = A_test @ B_test  # shape (n, p)
            
            # C_flat index: C[r, i] stored at r*n + i (per W convention W[k*n+i, r])
            # Wait: W has shape (p*n, rank), index is k*n+i where k=r (row of C) and i=col of C
            # So C is stored as C[r, i] at flat index r*n + i
            # C_test[i, r] is the (i,r) entry. We need C[r,i] for the flat storage.
            # Actually C_test is n x p, so C_test[i, r] where i in [0,n), r in [0,p)
            # The flat index for C[r, i] is r*n + i
            # So C_flat[r*n + i] should equal C_test[i, r]
            
            C_flat = np.zeros(p * n)
            for k in range(rank):
                # alpha_k = sum over A entries: U[am_idx, k] * A[am_idx]
                # A_flat[am_idx] where am_idx = i*m + j for A[i,j]
                A_flat = A_test.reshape(-1)  # shape (n*m,), A_flat[i*m+j] = A_test[i,j]
                B_flat = B_test.reshape(-1)  # shape (m*p,), B_flat[j*p+r] = B_test[j,r]
                
                alpha = float(np.dot(U_arr[:, k], A_flat))
                beta = float(np.dot(V_arr[:, k], B_flat))
                
                C_flat += W_arr[:, k] * alpha * beta
            
            # C_flat[r*n + i] should equal C_test[i, r]
            C_expected_flat = np.zeros(p * n)
            for i in range(n):
                for r in range(p):
                    C_expected_flat[r * n + i] = C_test[i, r]
            
            err = np.max(np.abs(C_flat - C_expected_flat))
            if err > 1e-10:
                return False, err
        return True, 0.0

    U, V, W = build_28_rank()
    
    # Verify the 28-rank decomposition
    valid, err = verify_decomp(U, V, W)
    if not valid:
        # Fallback to naive rank-32
        rank_naive = n * m * p
        U_naive = [[0]*rank_naive for _ in range(n*m)]
        V_naive = [[0]*rank_naive for _ in range(m*p)]
        W_naive = [[0]*rank_naive for _ in range(p*n)]
        for i in range(n):
            for j in range(m):
                for r in range(p):
                    k = i * m * p + j * p + r
                    U_naive[i*m+j][k] = 1
                    V_naive[j*p+r][k] = 1
                    W_naive[r*n+i][k] = 1
        return {"U": U_naive, "V": V_naive, "W": W_naive}
    
    # Try to reduce rank by checking if we can merge or eliminate terms
    # Approach: try to find if any subset of columns can be replaced
    # by a linear combination of others (i.e., the tensor has lower rank)
    
    # Method: try rank 27 by checking if one column is redundant
    # For each column k, try to see if we can express the k-th term
    # as a combination of the others
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank0 = 28
    
    # Build the tensor T as a 3D array
    # T[am, bm, cm] = sum_k U[am,k] * V[bm,k] * W[cm,k]
    # This is expensive for large dimensions, but n*m=8, m*p=16, p*n=8
    # So T is 8x16x8 = 1024 entries, each a sum of 28 products
    
    T = np.zeros((n*m, m*p, p*n))
    for k in range(rank0):
        T += np.einsum('a,b,c->abc', U_arr[:,k], V_arr[:,k], W_arr[:,k])
    
    # Now try to find if the tensor can be decomposed with fewer terms
    # Try: for each column k, remove it and try to fit the residual
    # with a linear combination of the other columns
    
    best_rank = rank0
    best_U, best_V, best_W = U, V, W
    
    for k_remove in range(rank0):
        if time.time() > deadline - 20:
            break
        
        # Remove column k_remove
        mask = [i for i in range(rank0) if i != k_remove]
        U_rem = U_arr[:, mask]
        V_rem = V_arr[:, mask]
        W_rem = W_arr[:, mask]
        
        # T_rem = sum_{k in mask} U[:,k] ⊗ V[:,k] ⊗ W[:,k]
        T_rem = np.zeros((n*m, m*p, p*n))
        for idx, k in enumerate(mask):
            T_rem += np.einsum('a,b,c->abc', U_arr[:,k], V_arr[:,k], W_arr[:,k])
        
        # Residual: T_res = T - T_rem = U[:,k_remove] ⊗ V[:,k_remove] ⊗ W[:,k_remove]
        T_res = T - T_rem
        
        # Can T_res be written as sum_{j in mask} c_j * U[:,j] ⊗ V[:,j] ⊗ W[:,j]?
        # This is: T_res[am,bm,cm] = sum_j c_j * U[am,j] * V[bm,j] * W[cm,j]
        # Flatten: T_res_flat = sum_j c_j * (U[:,j] ⊗ V[:,j] ⊗ W[:,j])_flat
        # This is a linear system: T_res_flat = M @ c
        # where M[flat_idx, j] = U[am,j]*V[bm,j]*W[cm,j] for the (am,bm,cm) at flat_idx
        
        T_res_flat = T_res.reshape(-1)
        # M has shape (n*m * m*p * p*n, rank0-1)
        M = np.zeros((n*m * m*p * p*n, rank0 - 1))
        for j_idx, j in enumerate(mask):
            M[:, j_idx] = np.einsum('a,b,c->abc', U_arr[:,j], V_arr[:,j], W_arr[:,j]).reshape(-1)
        
        # Solve least squares: M @ c ≈ T_res_flat
        c, residuals, rank_m, sv = np.linalg.lstsq(M, T_res_flat, rcond=None)
        
        # Check if the fit is good
        T_fitted = M @ c
        fit_err = np.max(np.abs(T_fitted - T_res_flat))
        
        if fit_err < 1e-10:
            # We can eliminate column k_remove!
            new_rank = rank0 - 1
            new_U = U_arr[:, mask].tolist()
            new_V = V_arr[:, mask].tolist()
            new_W = W_arr[:, mask].tolist()
            
            # Verify
            valid2, err2 = verify_decomp(new_U, new_V, new_W)
            if valid2:
                best_rank = new_rank
                best_U, best_V, best_W = new_U, new_V, new_W
                # Try to remove more columns
                # Re-set up for further reduction
                U_arr = np.array(new_U, dtype=float)
                V_arr = np.array(new_V, dtype=float)
                W_arr = np.array(new_W, dtype=float)
                rank0 = new_rank
                T = np.zeros((n*m, m*p, p*n))
                for k in range(rank0):
                    T += np.einsum('a,b,c->abc', U_arr[:,k], V_arr[:,k], W_arr[:,k])
                # Restart the removal loop
                k_remove = -1
                continue
    
    # Also try: can we find a completely different 27-rank or lower decomposition?
    # Try a different block structure: instead of 4 independent 2x2 Strassen blocks,
    # use a shared structure
    
    # Alternative: 2x4x4 as two 2x4x2 multiplications
    # C[:,0:2] = A * B[:,0:2], C[:,2:4] = A * B[:,2:4]
    # Each 2x4x2: A (2x4) * B_half (4x2)
    # Split A into A_left (2x2) and A_right (2x2)
    # C_half = A_left * B_half[0:2,:] + A_right * B_half[2:4,:]
    # Each 2x2x2 has rank 7, so 14 per half, 28 total
    
    # Same as before. Let's try yet another approach.
    
    # What about using a 4x4 Strassen decomposition but projecting?
    # The 4x4x4 Strassen has rank 49. But we only need 2x4x4.
    # Maybe we can project the 49 terms down to fewer?
    
    # For now, return the best we found
    if best_rank < 28:
        return {"U": best_U, "V": best_V, "W": best_W}
    else:
        return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
