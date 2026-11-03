"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import linprog

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    # Start with the known rank-55 Strassen-based decomposition
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part A: j=0 (first column of A, first row of B)
    # A[i,0] * B[0,k] -> C[i,k]
    for i in range(n):
        for k in range(p):
            u_idx = i * m + 0
            v_idx = 0 * p + k
            w_idx = k * n + i
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1
            
    # Part B: j=1,2 (last 2 cols of A, last 2 rows of B)
    # 5 blocks of 2x2 Strassen
    for b in range(5):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        w_00 = k0 * n + 0
        w_01 = k1 * n + 0
        w_10 = k0 * n + 1
        w_11 = k1 * n + 1
        
        u_11 = 0 * m + 1
        u_12 = 0 * m + 2
        u_21 = 1 * m + 1
        u_22 = 1 * m + 2
        
        v_11 = 1 * p + k0
        v_12 = 1 * p + k1
        v_21 = 2 * p + k0
        v_22 = 2 * p + k1
        
        # M1 = (a11 + a22)(b11 + b22)
        U[u_11][col] += 1
        U[u_22][col] += 1
        V[v_11][col] += 1
        V[v_22][col] += 1
        W[w_00][col] += 1
        W[w_11][col] += 1
        col += 1
        
        # M2 = (a21 + a22)b11
        U[u_21][col] += 1
        U[u_22][col] += 1
        V[v_11][col] += 1
        W[w_10][col] += 1
        W[w_11][col] += -1
        col += 1
        
        # M3 = a11(b12 - b22)
        U[u_11][col] += 1
        V[v_12][col] += 1
        V[v_22][col] += -1
        W[w_01][col] += 1
        W[w_11][col] += 1
        col += 1
        
        # M4 = a22(b21 - b11)
        U[u_22][col] += 1
        V[v_21][col] += 1
        V[v_11][col] += -1
        W[w_00][col] += 1
        W[w_10][col] += 1
        col += 1
        
        # M5 = (a11 + a12)b22
        U[u_11][col] += 1
        U[u_12][col] += 1
        V[v_22][col] += 1
        W[w_00][col] += -1
        W[w_01][col] += 1
        col += 1
        
        # M6 = (a21 - a11)(b11 + b12)
        U[u_21][col] += 1
        U[u_11][col] += -1
        V[v_11][col] += 1
        V[v_12][col] += 1
        W[w_11][col] += 1
        col += 1
        
        # M7 = (a12 - a22)(b21 + b22)
        U[u_12][col] += 1
        U[u_22][col] += -1
        V[v_21][col] += 1
        V[v_22][col] += 1
        W[w_00][col] += 1
        col += 1
    
    # Convert to numpy arrays for efficient operations
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    dim_u = n * m  # 6
    dim_v = m * p  # 30
    dim_w = p * n  # 20
    
    # Build the true tensor
    T_true = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                alpha = i * m + j
                beta = j * p + k
                gamma = k * n + i
                T_true[alpha, beta, gamma] = 1
    
    def verify_decomposition(U_mat, V_mat, W_mat, tol=1e-10):
        """Verify that the decomposition exactly matches T_true."""
        if time.time() > deadline - 5:
            return False
        T_rec = np.zeros((dim_u, dim_v, dim_w))
        for c in range(U_mat.shape[1]):
            T_rec += np.einsum('i,j,k->ijk', U_mat[:, c], V_mat[:, c], W_mat[:, c])
        return np.allclose(T_rec, T_true, atol=tol)
    
    current_U = U_np.copy()
    current_V = V_np.copy()
    current_W = W_np.copy()
    current_rank = rank
    
    # Attempt peeling: check each column for linear dependence
    # Vectorized approach using numpy
    
    def check_dependence_vectorized(U_mat, V_mat, W_mat, c):
        """Check if component c is a linear combination of others. Returns True if dependent."""
        other_cols = [i for i in range(U_mat.shape[1]) if i != c]
        if len(other_cols) < 2:
            return False, None
        
        # Build A matrix: A[flat_idx, oc_idx] = U[i,oc]*V[j,oc]*W[k,oc]
        # b[flat_idx] = U[i,c]*V[j,c]*W[k,c]
        
        # Use einsum to build the tensor for each other column
        # Shape: (dim_u, dim_v, dim_w, num_other)
        other_U = U_mat[:, other_cols]  # (dim_u, num_other)
        other_V = V_mat[:, other_cols]  # (dim_v, num_other)
        other_W = W_mat[:, other_cols]  # (dim_w, num_other)
        
        # Build A: for each (i,j,k), A[i,j,k,oc] = U[i,oc]*V[j,oc]*W[k,oc]
        # This is (dim_u, dim_v, dim_w, num_other)
        A_tensor = np.einsum('io,jo,ko->ijk o', other_U, other_V, other_W)
        A_mat = A_tensor.reshape(-1, len(other_cols))
        
        # Build b: b[i,j,k] = U[i,c]*V[j,c]*W[k,c]
        b_tensor = np.einsum('i,j,k->ijk', U_mat[:, c], V_mat[:, c], W_mat[:, c])
        b_vec = b_tensor.reshape(-1)
        
        b_norm = np.linalg.norm(b_vec)
        if b_norm < 1e-10:
            return False, None
        
        # Solve least squares
        try:
            lambda_sol, residuals, rank_ls, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
        except Exception:
            return False, None
        
        if residuals is not None and len(residuals) > 0:
            residual_norm = np.sqrt(residuals[0])
            if residual_norm / b_norm < 1e-8:
                return True, lambda_sol
        return False, None
    
    peeled = True
    while peeled and time.time() < deadline - 15:
        peeled = False
        for c in range(current_rank):
            if time.time() > deadline - 15:
                break
            
            try:
                is_dep, _ = check_dependence_vectorized(current_U, current_V, current_W, c)
                if is_dep:
                    current_U = np.delete(current_U, c, axis=1)
                    current_V = np.delete(current_V, c, axis=1)
                    current_W = np.delete(current_W, c, axis=1)
                    current_rank -= 1
                    peeled = True
                    break
            except Exception:
                continue
    
    # Verify the peeled decomposition
    if current_rank < rank:
        if verify_decomposition(current_U, current_V, current_W):
            # Convert back to list format
            U_final = current_U.tolist()
            V_final = current_V.tolist()
            W_final = current_W.tolist()
            return {"U": U_final, "V": V_final, "W": W_final}
    
    # Try a more aggressive approach: iterative peeling with multiple passes
    # and also try to find components that can be merged (sum to zero contribution)
    
    # Additional pass: try to remove components by checking if their contribution
    # can be absorbed. Use a different criterion: check if the component's
    # "fingerprint" (pattern of non-zeros in U, V, W) allows merging.
    
    peeled2 = True
    while peeled2 and time.time() < deadline - 10:
        peeled2 = False
        for c in range(current_rank):
            if time.time() > deadline - 10:
                break
            
            # Try removing component c and see if we can compensate
            # by adjusting other components. This is equivalent to the linear
            # dependence check but we use a different numerical approach.
            
            # Alternative: check if the rank-1 tensor of component c is in the
            # span of the other rank-1 tensors. We already do this above.
            # Instead, let's try a different heuristic: if component c has
            # a very sparse pattern, it might be mergeable.
            
            u_c = current_U[:, c]
            v_c = current_V[:, c]
            w_c = current_W[:, c]
            
            # Count non-zeros
            nz_u = np.count_nonzero(u_c)
            nz_v = np.count_nonzero(v_c)
            nz_w = np.count_nonzero(w_c)
            
            # If the component is very sparse (e.g., only 1 non-zero in one factor),
            # it might be easier to merge
            if nz_u == 1 or nz_v == 1 or nz_w == 1:
                # Try to find another component that can absorb this one
                for d in range(current_rank):
                    if d == c:
                        continue
                    if time.time() > deadline - 10:
                        break
                    
                    # Check if we can express component c as a scaled version
                    # of component d plus adjustments
                    u_d = current_U[:, d]
                    v_d = current_V[:, d]
                    w_d = current_W[:, d]
                    
                    # If the supports are similar, try to merge
                    # Simple heuristic: if they share the same non-zero pattern
                    support_u_c = np.nonzero(u_c)[0]
                    support_u_d = np.nonzero(u_d)[0]
                    support_v_c = np.nonzero(v_c)[0]
                    support_v_d = np.nonzero(v_d)[0]
                    support_w_c = np.nonzero(w_c)[0]
                    support_w_d = np.nonzero(w_d)[0]
                    
                    # If supports are subsets, they might be mergeable
                    if (set(support_u_c).issubset(set(support_u_d)) and
                        set(support_v_c).issubset(set(support_v_d)) and
                        set(support_w_c).issubset(set(support_w_d))):
                        # Try to find a scaling factor
                        # For each non-zero in c, check if u_c[i]*v_c[j]*w_c[k] 
                        # can be matched by u_d[i]*v_d[j]*w_d[k]
                        # This requires u_c[i]/u_d[i] * v_c[j]/v_d[j] * w_c[k]/w_d[k] = const
                        # for all non-zero (i,j,k)
                        
                        # Get the non-zero indices
                        nz_idx_u = np.nonzero(u_c)[0]
                        nz_idx_v = np.nonzero(v_c)[0]
                        nz_idx_w = np.nonzero(w_c)[0]
                        
                        if len(nz_idx_u) == 0 or len(nz_idx_v) == 0 or len(nz_idx_w) == 0:
                            continue
                        
                        # For a 1-1-1 pattern, just check if the product matches
                        if len(nz_idx_u) == 1 and len(nz_idx_v) == 1 and len(nz_idx_w) == 1:
                            i, j, k = nz_idx_u[0], nz_idx_v[0], nz_idx_w[0]
                            if abs(u_d[i]) > 1e-10 and abs(v_d[j]) > 1e-10 and abs(w_d[k]) > 1e-10:
                                scale = (u_c[i] * v_c[j] * w_c[k]) / (u_d[i] * v_d[j] * w_d[k])
                                # Check if this scale works for all entries
                                valid = True
                                for ii in nz_idx_u:
                                    for jj in nz_idx_v:
                                        for kk in nz_idx_w:
                                            expected = u_c[ii] * v_c[jj] * w_c[kk]
                                            actual = scale * u_d[ii] * v_d[jj] * w_d[kk]
                                            if abs(expected - actual) > 1e-8 * max(1, abs(expected)):
                                                valid = False
                                                break
                                        if not valid:
                                            break
                                    if not valid:
                                        break
                                if valid:
                                    # Merge: add scaled d to c, then remove c
                                    # Actually, we want to absorb c into d
                                    # new_d = d + (1/scale) * c ... wait
                                    # We want: c = scale * d (on the support)
                                    # So we can just remove c and scale d appropriately?
                                    # No, we need to be more careful.
                                    # The contribution of c is u_c ⊗ v_c ⊗ w_c
                                    # The contribution of d is u_d ⊗ v_d ⊗ w_d
                                    # If u_c ⊗ v_c ⊗ w_c = scale * (u_d ⊗ v_d ⊗ w_d),
                                    # then we can remove c and scale d by ... 
                                    # Actually, we can't just scale d because d contributes
                                    # to other terms too. We need to add the correction.
                                    # 
                                    # Better: remove c, and for each entry where c was non-zero,
                                    # adjust d's factors. But this doesn't work directly.
                                    # 
                                    # The correct approach: if c = scale * d (as rank-1 tensors),
                                    # then we can remove c and that's it, because c's contribution
                                    # is already captured by d (scaled). But d's scale is fixed.
                                    # We need: sum of all components = original.
                                    # If we remove c, we lose its contribution.
                                    # We can add it to d by modifying d's factors, but that
                                    # changes d's contribution to all entries.
                                    #
                                    # This merge only works if c and d have identical support
                                    # and the ratios are constant. In that case, we can
                                    # combine them into a single component.
                                    pass
                    
            # Continue with the standard peeling approach for now
            is_dep, _ = check_dependence_vectorized(current_U, current_V, current_W, c)
            if is_dep:
                current_U = np.delete(current_U, c, axis=1)
                current_V = np.delete(current_V, c, axis=1)
                current_W = np.delete(current_W, c, axis=1)
                current_rank -= 1
                peeled2 = True
                break
    
    # Verify the final decomposition
    if current_rank < rank:
        if verify_decomposition(current_U, current_V, current_W):
            U_final = current_U.tolist()
            V_final = current_V.tolist()
            W_final = current_W.tolist()
            return {"U": U_final, "V": V_final, "W": W_final}
    
    # Fallback: verify original and return
    if verify_decomposition(U_np, V_np, W_np):
        return {"U": U, "V": V, "W": W}
    
    # Final fallback to schoolbook
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col_s = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col_s] = 1
                V_school[j * p + k][col_s] = 1
                W_school[k * n + i][col_s] = 1
                col_s += 1
    return {"U": U_school, "V": V_school, "W": W_school}
# EVOLVE-BLOCK-END
