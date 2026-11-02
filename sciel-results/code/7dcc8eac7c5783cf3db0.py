import time
import numpy as np

DIMENSIONS = [2, 2, 10]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    
    # Strassen-based rank-35 solution
    strassen_terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 1]),
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 1, 0, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]
    
    rank = 7 * (p // 2)
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for k in range(p // 2):
        for a_vec, b_vec, c_vec in strassen_terms:
            for idx in range(n * m):
                U[idx, col] = a_vec[idx]
            V[0 * p + 2 * k, col] = b_vec[0]
            V[0 * p + 2 * k + 1, col] = b_vec[1]
            V[1 * p + 2 * k, col] = b_vec[2]
            V[1 * p + 2 * k + 1, col] = b_vec[3]
            W[2 * k * n + 0, col] = c_vec[0]
            W[2 * k * n + 1, col] = c_vec[1]
            W[(2 * k + 1) * n + 0, col] = c_vec[2]
            W[(2 * k + 1) * n + 1, col] = c_vec[3]
            col += 1
    
    # Verify the solution
    def verify(U, V, W, n, m, p, tol=1e-6):
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    T[i*m + k, k*p + j, j*n + i] = 1.0
        result = np.zeros((n*m, m*p, p*n))
        for k in range(U.shape[1]):
            result += np.einsum('a,b,c->abc', U[:, k], V[:, k], W[:, k])
        return np.allclose(result, T, atol=tol)
    
    if not verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank_sb = n * m * p
        U_sb = np.zeros((n*m, rank_sb))
        V_sb = np.zeros((m*p, rank_sb))
        W_sb = np.zeros((p*n, rank_sb))
        col = 0
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    U_sb[i*m + k, col] = 1
                    V_sb[k*p + j, col] = 1
                    W_sb[j*n + i, col] = 1
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    # --- PSLQ-based dependency detection ---
    # Build the 40x35 coefficient matrix: each column is the tensor entry
    # (flattened) for one Strassen term. If there are linear dependencies
    # among columns, we can reduce rank.
    # Tensor size: (n*m, m*p, p*n) = (4, 20, 20) = 1600 entries
    # But we only need the "active" entries: for each term, the support
    # is determined by the non-zero pattern.
    
    # Actually, let's build the full 1600x35 matrix and check rank.
    # The tensor T has T[i*m+k, k*p+j, j*n+i] = 1 for all i,k,j.
    # Each Strassen term contributes a rank-1 tensor u ⊗ v ⊗ w.
    
    def build_tensor_matrix(U_mat, V_mat, W_mat, n, m, p):
        """Build the 1600 x rank matrix where each column is the flattened
        rank-1 tensor for that term."""
        num_terms = U_mat.shape[1]
        total = n * m * m * p * p * n
        M = np.zeros((total, num_terms))
        for t in range(num_terms):
            u = U_mat[:, t]
            v = V_mat[:, t]
            w = W_mat[:, t]
            # Flatten u ⊗ v ⊗ w
            M[:, t] = np.outer(np.outer(u, v), w).flatten()
        return M
    
    M = build_tensor_matrix(U, V, W, n, m, p)
    
    # Compute rank of M via SVD
    try:
        if time.time() < deadline - 5.0:
            s, _, _ = np.linalg.svd(M, full_matrices=False)
            # Count significant singular values
            tol_sv = 1e-10 * max(M.shape) * s[0] if s[0] > 0 else 1e-10
            rank_actual = np.sum(s > tol_sv)
            
            if rank_actual < rank:
                # There are dependencies! Find the null space basis.
                # The null space vectors give linear combinations of terms
                # that sum to zero. We can eliminate terms using these.
                # 
                # Strategy: use QR with column pivoting on M^T to find
                # a set of rank_actual linearly independent columns,
                # then express the remaining columns as combinations.
                # But for exact integer/half-integer results, we need
                # to be careful.
                #
                # Alternative: use the null space to identify which terms
                # can be removed. If the null space has dimension d, we
                # can potentially remove up to d terms.
                #
                # For now, let's try a greedy approach: remove terms one
                # at a time if the remaining set still spans the tensor.
                
                # Get null space via SVD
                null_space = V_mat_right = np.linalg.svd(M, full_matrices=True)[2]
                # The last (rank - rank_actual) right singular vectors span the null space
                null_dim = rank - rank_actual
                if null_dim > 0:
                    null_vecs = null_space[:, rank - null_dim:]
                    
                    # Try to find integer/rational null space vectors
                    # and use them to eliminate terms.
                    # For each null vector, find the term with largest
                    # coefficient and eliminate it.
                    
                    keep = list(range(rank))
                    for nv in null_vecs:
                        # Scale to make entries rational-ish
                        # Find the index with largest absolute value
                        max_idx = np.argmax(np.abs(nv))
                        if nv[max_idx] == 0:
                            continue
                        # Try to eliminate this term
                        if max_idx in keep:
                            # Check if removing it still works
                            new_keep = [k for k in keep if k != max_idx]
                            # Verify the reduced decomposition
                            U_red = U[:, new_keep]
                            V_red = V[:, new_keep]
                            W_red = W[:, new_keep]
                            if verify(U_red, V_red, W_red, n, m, p):
                                keep = new_keep
                    
                    if len(keep) < rank:
                        U = U[:, keep]
                        V = V[:, keep]
                        W = W[:, keep]
                        rank = len(keep)
    except Exception:
        pass
    
    # Final verification
    if not verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank_sb = n * m * p
        U_sb = np.zeros((n*m, rank_sb))
        V_sb = np.zeros((m*p, rank_sb))
        W_sb = np.zeros((p*n, rank_sb))
        col = 0
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    U_sb[i*m + k, col] = 1
                    V_sb[k*p + j, col] = 1
                    W_sb[j*n + i, col] = 1
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
