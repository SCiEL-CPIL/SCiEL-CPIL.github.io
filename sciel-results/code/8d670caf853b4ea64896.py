"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _build_tensor(n, m, p):
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1
    return T

def _verify_decomposition(U, V, W, T, tol=1e-9):
    rank = U.shape[1]
    result = np.zeros(T.shape)
    for r in range(rank):
        result += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
    return np.allclose(result, T, atol=tol)

def _get_rank17_solution():
    n, m, p = 2, 3, 3
    n_a, n_b, n_c = 6, 9, 6
    rank = 17
    U = np.zeros((n_a, rank))
    V = np.zeros((n_b, rank))
    W = np.zeros((n_c, rank))
    
    # Strassen for 2x2x2 block (rows 0,1 of A; cols 0,1 of B; output 2x2)
    # A indices: 0=(0,0), 1=(0,1), 3=(1,0), 4=(1,1)
    # B indices: 0=(0,0), 1=(0,1), 3=(1,0), 4=(1,1)
    # W indices: 0=(0,0), 1=(1,0), 2=(0,1), 3=(1,1)
    U[0,0]=1; U[4,0]=1; V[0,0]=1; V[4,0]=1; W[0,0]=1; W[3,0]=1
    U[3,1]=1; U[4,1]=1; V[0,1]=1; W[1,1]=1; W[3,1]=-1
    U[0,2]=1; V[1,2]=1; V[4,2]=-1; W[2,2]=1; W[3,2]=1
    U[4,3]=1; V[3,3]=1; V[0,3]=-1; W[0,3]=1; W[1,3]=1
    U[0,4]=1; U[1,4]=1; V[4,4]=1; W[0,4]=-1; W[2,4]=1
    U[3,5]=1; U[0,5]=-1; V[0,5]=1; V[1,5]=1; W[3,5]=1
    U[1,6]=1; U[4,6]=-1; V[3,6]=1; V[4,6]=1; W[0,6]=1
    
    # Schoolbook for remaining terms
    U[0,7]=1; V[2,7]=1; W[4,7]=1   # A(0,0)*B(0,2) -> C(0,2)
    U[1,8]=1; V[5,8]=1; W[4,8]=1   # A(0,1)*B(1,2) -> C(0,2)
    U[3,9]=1; V[2,9]=1; W[5,9]=1   # A(1,0)*B(0,2) -> C(1,2)
    U[4,10]=1; V[5,10]=1; W[5,10]=1 # A(1,1)*B(1,2) -> C(1,2)
    U[2,11]=1; V[6,11]=1; W[0,11]=1 # A(0,2)*B(2,0) -> C(0,0)
    U[2,12]=1; V[7,12]=1; W[2,12]=1 # A(0,2)*B(2,1) -> C(0,1)
    U[2,13]=1; V[8,13]=1; W[4,13]=1 # A(0,2)*B(2,2) -> C(0,2)
    U[5,14]=1; V[6,14]=1; W[1,14]=1 # A(1,2)*B(2,0) -> C(1,0)
    U[5,15]=1; V[7,15]=1; W[3,15]=1 # A(1,2)*B(2,1) -> C(1,1)
    U[5,16]=1; V[8,16]=1; W[5,16]=1 # A(1,2)*B(2,2) -> C(1,2)
    
    return U, V, W

def _als_polish(U, V, W, T, max_iter=50, deadline=None):
    """Alternating least squares to polish a rank-16 candidate."""
    n_a, n_b, n_c = T.shape
    rank = U.shape[1]
    T_flat = T.flatten()
    
    for it in range(max_iter):
        if deadline and time.time() > deadline:
            break
        # Solve for U: fix V, W
        # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # For each a: T[a,:,:].flatten() = sum_r U[a,r] * (V[:,r]*W[:,r]).flatten()
        B = np.zeros((rank, n_b * n_c))
        for r in range(rank):
            B[r, :] = (V[:, r] * W[:, r]).flatten()
        A_mat = np.zeros((n_a, n_b * n_c))
        for a in range(n_a):
            A_mat[a, :] = T[a, :, :].flatten()
        U_sol, _, _, _ = np.linalg.lstsq(B.T, A_mat.T, rcond=None)
        U = U_sol.T
        
        # Solve for V
        B = np.zeros((rank, n_a * n_c))
        for r in range(rank):
            B[r, :] = (U[:, r] * W[:, r]).flatten()
        A_mat = np.zeros((n_b, n_a * n_c))
        for b in range(n_b):
            A_mat[b, :] = T[:, b, :].flatten()
        V_sol, _, _, _ = np.linalg.lstsq(B.T, A_mat.T, rcond=None)
        V = V_sol.T
        
        # Solve for W
        B = np.zeros((rank, n_a * n_b))
        for r in range(rank):
            B[r, :] = (U[:, r] * V[:, r]).flatten()
        A_mat = np.zeros((n_c, n_a * n_b))
        for c in range(n_c):
            A_mat[c, :] = T[:, :, c].flatten()
        W_sol, _, _, _ = np.linalg.lstsq(B.T, A_mat.T, rcond=None)
        W = W_sol.T
    
    return U, V, W

def _try_merge_pair(U17, V17, W17, T, r1, r2, deadline):
    """
    Try to merge terms r1 and r2 into a single term.
    The merged term can have up to 3 non-zero entries in each factor.
    We solve for the merged term's entries and check if the remaining 15 terms
    can be scaled to complete the decomposition.
    """
    n_a, n_b, n_c = U17.shape[0], V17.shape[0], W17.shape[0]
    T_flat = T.flatten()
    num_entries = T_flat.size
    rank17 = U17.shape[1]
    
    other_indices = [r for r in range(rank17) if r != r1 and r != r2]
    
    # The merged term will have unknown entries.
    # Strategy: try all combinations of support sizes (su, sv, sw) where su+sv+sw is small
    # For each support choice, solve for the values.
    
    # Get the "free" tensor: T - sum of other terms (with unknown scalings)
    # Actually, let's set up a linear system.
    # Let the merged term be (u, v, w) with unknowns.
    # Let the other terms have unknown scalings alpha_0, ..., alpha_14.
    # Total unknowns: |sup_u| + |sup_v| + |sup_w| + 15
    # Equations: num_entries (324)
    
    # Try small supports for the merged term
    # The merged term should capture the "difference" between terms r1 and r2
    
    # First, compute what the other 15 terms contribute (with unit scaling)
    # and what T - that looks like. Then see if a rank-1 tensor can match it.
    
    # Actually, a cleaner approach: 
    # Fix the other 15 terms to their original values (scaling = 1).
    # Compute residual = T - sum_{other} (U[:,r] V[:,r] W[:,r])
    # Check if residual is rank-1 (i.e., can be written as u v w^T)
    
    residual = T.copy()
    for r in other_indices:
        residual -= np.einsum('a,b,c->abc', U17[:,r], V17[:,r], W17[:,r])
    
    # Check if residual is rank-1
    # Reshape to matrix and check rank
    # residual[a,b,c] - try all flattenings
    mat1 = residual.reshape(n_a, n_b * n_c)
    rank1 = np.linalg.matrix_rank(mat1, tol=1e-10)
    
    if rank1 == 1:
        # Extract u, v, w from the rank-1 matrix
        # mat1 = u @ (v*w).T
        u, s, vh = np.linalg.svd(mat1, full_matrices=False)
        u_vec = u[:, 0] * s[0]
        vw_vec = vh[0]
        # Now vw_vec[b*n_c + c] = v[b]*w[c]
        # Check if vw_vec is rank-1 as a matrix
        vw_mat = vw_vec.reshape(n_b, n_c)
        rank2 = np.linalg.matrix_rank(vw_mat, tol=1e-10)
        if rank2 == 1:
            v, s2, wh = np.linalg.svd(vw_mat, full_matrices=False)
            v_vec = v[:, 0] * s2[0]
            w_vec = wh[0]
            
            # Build the rank-16 solution
            U16 = np.zeros((n_a, 16))
            V16 = np.zeros((n_b, 16))
            W16 = np.zeros((n_c, 16))
            for j, r in enumerate(other_indices):
                U16[:, j] = U17[:, r]
                V16[:, j] = V17[:, r]
                W16[:, j] = W17[:, r]
            U16[:, 15] = u_vec
            V16[:, 15] = v_vec
            W16[:, 15] = w_vec
            
            if _verify_decomposition(U16, V16, W16, T):
                return U16, V16, W16
    
    # Also try with different scalings of the other terms
    # This is more complex; skip for now
    
    return None

def _try_absorb_single(U17, V17, W17, T, remove_idx, deadline):
    """Try to absorb term remove_idx into the other 16 terms by adding one entry."""
    n_a, n_b, n_c = U17.shape[0], V17.shape[0], W17.shape[0]
    T_flat = T.flatten()
    num_entries = T_flat.size
    rank17 = U17.shape[1]
    keep_indices = [r for r in range(rank17) if r != remove_idx]
    
    # Try: no modification
    A = np.zeros((num_entries, 16))
    for j, r in enumerate(keep_indices):
        A[:, j] = np.einsum('a,b,c->abc', U17[:, r], V17[:, r], W17[:, r]).flatten()
    try:
        alpha, _, _, _ = np.linalg.lstsq(A, T_flat, rcond=None)
        if np.allclose(A @ alpha, T_flat, atol=1e-9):
            U16 = np.zeros((n_a, 16))
            V16 = np.zeros((n_b, 16))
            W16 = np.zeros((n_c, 16))
            for j, r in enumerate(keep_indices):
                U16[:, j] = alpha[j] * U17[:, r]
                V16[:, j] = alpha[j] * V17[:, r]
                W16[:, j] = alpha[j] * W17[:, r]
            return U16, V16, W16
    except:
        pass
    
    # Try adding one entry to U, V, or W of each term
    base_supU = [set(np.nonzero(U17[:, r])[0]) for r in keep_indices]
    base_supV = [set(np.nonzero(V17[:, r])[0]) for r in keep_indices]
    base_supW = [set(np.nonzero(W17[:, r])[0]) for r in keep_indices]
    
    for j in range(16):
        if time.time() > deadline:
            return None
        for new_a in range(n_a):
            if new_a not in base_supU[j]:
                supU = list(base_supU[j]) + [new_a]
                supV = list(base_supV[j])
                supW = list(base_supW[j])
                U_test = np.zeros((n_a, 16))
                V_test = np.zeros((n_b, 16))
                W_test = np.zeros((n_c, 16))
                for jj, rr in enumerate(keep_indices):
                    if jj == j:
                        for idx in supU:
                            U_test[idx, jj] = 1
                        for idx in supV:
                            V_test[idx, jj] = 1
                        for idx in supW:
                            W_test[idx, jj] = 1
                    else:
                        U_test[:, jj] = U17[:, rr]
                        V_test[:, jj] = V17[:, rr]
                        W_test[:, jj] = W17[:, rr]
                A_test = np.zeros((num_entries, 16))
                for jj in range(16):
                    A_test[:, jj] = np.einsum('a,b,c->abc', U_test[:, jj], V_test[:, jj], W_test[:, jj]).flatten()
                try:
                    alpha_test, _, _, _ = np.linalg.lstsq(A_test, T_flat, rcond=None)
                    if np.allclose(A_test @ alpha_test, T_flat, atol=1e-9):
                        U16 = np.zeros((n_a, 16))
                        V16 = np.zeros((n_b, 16))
                        W16 = np.zeros((n_c, 16))
                        for jj in range(16):
                            U16[:, jj] = alpha_test[jj] * U_test[:, jj]
                            V16[:, jj] = alpha_test[jj] * V_test[:, jj]
                            W16[:, jj] = alpha_test[jj] * W_test[:, jj]
                        return U16, V16, W16
                except:
                    pass
        for new_b in range(n_b):
            if new_b not in base_supV[j]:
                supU = list(base_supU[j])
                supV = list(base_supV[j]) + [new_b]
                supW = list(base_supW[j])
                U_test = np.zeros((n_a, 16))
                V_test = np.zeros((n_b, 16))
                W_test = np.zeros((n_c, 16))
                for jj, rr in enumerate(keep_indices):
                    if jj == j:
                        for idx in supU:
                            U_test[idx, jj] = 1
                        for idx in supV:
                            V_test[idx, jj] = 1
                        for idx in supW:
                            W_test[idx, jj] = 1
                    else:
                        U_test[:, jj] = U17[:, rr]
                        V_test[:, jj] = V17[:, rr]
                        W_test[:, jj] = W17[:, rr]
                A_test = np.zeros((num_entries, 16))
                for jj in range(16):
                    A_test[:, jj] = np.einsum('a,b,c->abc', U_test[:, jj], V_test[:, jj], W_test[:, jj]).flatten()
                try:
                    alpha_test, _, _, _ = np.linalg.lstsq(A_test, T_flat, rcond=None)
                    if np.allclose(A_test @ alpha_test, T_flat, atol=1e-9):
                        U16 = np.zeros((n_a, 16))
                        V16 = np.zeros((n_b, 16))
                        W16 = np.zeros((n_c, 16))
                        for jj in range(16):
                            U16[:, jj] = alpha_test[jj] * U_test[:, jj]
                            V16[:, jj] = alpha_test[jj] * V_test[:, jj]
                            W16[:, jj] = alpha_test[jj] * W_test[:, jj]
                        return U16, V16, W16
                except:
                    pass
        for new_c in range(n_c):
            if new_c not in base_supW[j]:
                supU = list(base_supU[j])
                supV = list(base_supV[j])
                supW = list(base_supW[j]) + [new_c]
                U_test = np.zeros((n_a, 16))
                V_test = np.zeros((n_b, 16))
                W_test = np.zeros((n_c, 16))
                for jj, rr in enumerate(keep_indices):
                    if jj == j:
                        for idx in supU:
                            U_test[idx, jj] = 1
                        for idx in supV:
                            V_test[idx, jj] = 1
                        for idx in supW:
                            W_test[idx, jj] = 1
                    else:
                        U_test[:, jj] = U17[:, rr]
                        V_test[:, jj] = V17[:, rr]
                        W_test[:, jj] = W17[:, rr]
                A_test = np.zeros((num_entries, 16))
                for jj in range(16):
                    A_test[:, jj] = np.einsum('a,b,c->abc', U_test[:, jj], V_test[:, jj], W_test[:, jj]).flatten()
                try:
                    alpha_test, _, _, _ = np.linalg.lstsq(A_test, T_flat, rcond=None)
                    if np.allclose(A_test @ alpha_test, T_flat, atol=1e-9):
                        U16 = np.zeros((n_a, 16))
                        V16 = np.zeros((n_b, 16))
                        W16 = np.zeros((n_c, 16))
                        for jj in range(16):
                            U16[:, jj] = alpha_test[jj] * U_test[:, jj]
                            V16[:, jj] = alpha_test[jj] * V_test[:, jj]
                            W16[:, jj] = alpha_test[jj] * W_test[:, jj]
                        return U16, V16, W16
                except:
                    pass
    return None

def _random_rank16_search(T, deadline, num_trials=30):
    """Randomized ALS search for rank-16 decomposition."""
    n_a, n_b, n_c = T.shape
    T_flat = T.flatten()
    
    for trial in range(num_trials):
        if time.time() > deadline:
            return None
        rng = np.random.RandomState(42 + trial)
        U = rng.randn(n_a, 16)
        V = rng.randn(n_b, 16)
        W = rng.randn(n_c, 16)
        
        U, V, W = _als_polish(U, V, W, T, max_iter=30, deadline=deadline)
        
        if _verify_decomposition(U, V, W, T, tol=1e-8):
            return U, V, W
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    n, m, p = DIMENSIONS
    n_a = n * m  # 6
    n_b = m * p  # 9
    n_c = p * n  # 6
    T = _build_tensor(n, m, p)
    
    U17, V17, W17 = _get_rank17_solution()
    
    # 1. Try merging pairs of terms
    for r1 in range(17):
        if time.time() > deadline:
            break
        for r2 in range(r1 + 1, 17):
            if time.time() > deadline:
                break
            result = _try_merge_pair(U17, V17, W17, T, r1, r2, deadline)
            if result is not None:
                U, V, W = result
                if _verify_decomposition(U, V, W, T):
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # 2. Try absorbing single terms
    for remove_idx in range(17):
        if time.time() > deadline:
            break
        result = _try_absorb_single(U17, V17, W17, T, remove_idx, deadline)
        if result is not None:
            U, V, W = result
            if _verify_decomposition(U, V, W, T):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # 3. Randomized ALS search
    result = _random_rank16_search(T, deadline, num_trials=20)
    if result is not None:
        U, V, W = result
        if _verify_decomposition(U, V, W, T, tol=1e-8):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # 4. Fallback to verified Rank 17
    if not _verify_decomposition(U17, V17, W17, T):
        rank_fb = n * m * p
        U_fb = np.zeros((n_a, rank_fb))
        V_fb = np.zeros((n_b, rank_fb))
        W_fb = np.zeros((n_c, rank_fb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j, col] = 1
                    V_fb[j*p+k, col] = 1
                    W_fb[k*n+i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    return {"U": U17.tolist(), "V": V17.tolist(), "W": W17.tolist()}
# EVOLVE-BLOCK-END
