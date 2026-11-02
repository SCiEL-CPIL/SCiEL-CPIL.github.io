"""Matrix multiplication 2x5x5 - rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 5, 5]

_START = time.time()
_DEADLINE = _START + 120.0


def _remaining():
    return _DEADLINE - time.time()


# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS

    terms = []

    def add_term(u_entries, v_entries, w_entries):
        terms.append((u_entries, v_entries, w_entries))

    def add_schoolbook(i_start, i_end, j_start, j_end, k_start, k_end):
        for i in range(i_start, i_end):
            for j in range(j_start, j_end):
                for k in range(k_start, k_end):
                    add_term([(i, j, 1)], [(j, k, 1)], [(k, i, 1)])

    def add_strassen(i_start, j_start, k_start):
        i0, i1 = i_start, i_start + 1
        j0, j1 = j_start, j_start + 1
        k0, k1 = k_start, k_start + 1
        # M1 = (A00 + A11)(B00 + B11)
        add_term([(i0, j0, 1), (i1, j1, 1)],
                 [(j0, k0, 1), (j1, k1, 1)],
                 [(k0, i0, 1), (k1, i1, 1)])
        # M2 = (A10 + A11) B00
        add_term([(i1, j0, 1), (i1, j1, 1)],
                 [(j0, k0, 1)],
                 [(k0, i1, 1), (k1, i1, -1)])
        # M3 = A00 (B01 - B11)
        add_term([(i0, j0, 1)],
                 [(j0, k1, 1), (j1, k1, -1)],
                 [(k1, i0, 1), (k1, i1, 1)])
        # M4 = A11 (B10 - B00)
        add_term([(i1, j1, 1)],
                 [(j1, k0, 1), (j0, k0, -1)],
                 [(k0, i0, 1), (k0, i1, 1)])
        # M5 = (A00 + A01) B11
        add_term([(i0, j0, 1), (i0, j1, 1)],
                 [(j1, k1, 1)],
                 [(k0, i0, -1), (k1, i0, 1)])
        # M6 = (A10 - A00)(B00 + B01)
        add_term([(i1, j0, 1), (i0, j0, -1)],
                 [(j0, k0, 1), (j0, k1, 1)],
                 [(k1, i1, 1)])
        # M7 = (A01 - A11)(B10 + B11)
        add_term([(i0, j1, 1), (i1, j1, -1)],
                 [(j1, k0, 1), (j1, k1, 1)],
                 [(k0, i0, 1)])

    # Partition m = (2,2,1), p = (2,2,1); n = 2.
    add_strassen(0, 0, 0)
    add_strassen(0, 0, 2)
    add_schoolbook(0, 2, 0, 2, 4, 5)
    add_strassen(0, 2, 0)
    add_strassen(0, 2, 2)
    add_schoolbook(0, 2, 2, 4, 4, 5)
    add_schoolbook(0, 2, 4, 5, 0, 2)
    add_schoolbook(0, 2, 4, 5, 2, 4)
    add_schoolbook(0, 2, 4, 5, 4, 5)

    def _decompose_terms():
        """Return (U, V, W) as lists of lists from current term set (exact floats)."""
        rank = len(terms)
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        for c in range(rank):
            a, b, w = terms[c]
            for (i, j, val) in a:
                U[i * m + j][c] = val
            for (j, k, val) in b:
                V[j * p + k][c] = val
            for (k, i, val) in w:
                W[k * n + i][c] = val
        return U, V, W

    def _tensor_from(U, V, W):
        rank = len(W[0]) if W else 0
        T = {}
        for c in range(rank):
            for ii in range(n * m):
                u = U[ii][c]
                if u == 0.0:
                    continue
                r, s = divmod(ii, m)
                for jj in range(m * p):
                    v = V[jj][c]
                    if v == 0.0:
                        continue
                    t, uidx = divmod(jj, p)
                    if s != t:
                        continue
                    for kk in range(p * n):
                        w = W[kk][c]
                        if w == 0.0:
                            continue
                        key = (r, uidx, divmod(kk, n)[1])
                        T[key] = T.get(key, 0.0) + u * v * w
        return T

    def _exact(T, tol=1e-9):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if abs(T.get((i, j, k), 0.0) - 1.0) > tol:
                        return False
        for (i, j, k), val in T.items():
            if i < 0 or i >= n or j < 0 or j >= m or k < 0 or k >= p:
                if abs(val) > tol:
                    return False
        return True

    def _svd_reduce(U, V, W, tol=1e-10):
        """
        Use SVD to find linear dependencies among rank vectors.
        If the U matrix (n*m x rank) has rank < rank, we can reduce.
        Similarly for V and W.
        Returns reduced (U, V, W) if exact, else None.
        """
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        rank_orig = U_arr.shape[1]
        if rank_orig == 0:
            return U, V, W
        
        # Check if we can reduce by finding null space of U^T, V^T, W^T
        # The decomposition is T = sum_c u_c ⊗ v_c ⊗ w_c
        # If there's a linear dependency among the u_c's, we can potentially reduce.
        
        # Strategy: Find the actual rank of each matrix
        U_rank = np.linalg.matrix_rank(U_arr, tol=tol)
        V_rank = np.linalg.matrix_rank(V_arr, tol=tol)
        W_rank = np.linalg.matrix_rank(W_arr, tol=tol)
        
        # The effective rank is bounded by min of these, but we need to be careful
        # because the tensor decomposition requires all three to be consistent.
        
        # Try to reduce using SVD of U
        if U_rank < rank_orig:
            # Find basis for column space of U
            U_svd = np.linalg.svd(U_arr, full_matrices=False)
            # Keep only significant singular vectors
            sig_idx = np.where(U_svd[1] > tol)[0]
            if len(sig_idx) < rank_orig:
                # Project V and W onto the reduced space
                U_reduced = U_arr[:, sig_idx]
                # We need to find new V and W such that the tensor is preserved
                # This is tricky - the reduction isn't simply dropping columns
                
                # Alternative: use the null space
                # If U * alpha = 0 for some alpha, then sum_c alpha_c * u_c ⊗ v_c ⊗ w_c = 0
                # This means we can add alpha_c * (v_c, w_c) combinations that cancel out
        
        # Simpler approach: try to find if any column is a linear combination of others
        # For each matrix, check for linear dependencies
        
        def find_redundant_columns(mat, tol=1e-10):
            """Find columns that are linear combinations of others."""
            mat_arr = np.array(mat, dtype=float)
            n_cols = mat_arr.shape[1]
            redundant = set()
            
            for c in range(n_cols):
                # Check if column c is in span of other columns
                other_cols = [j for j in range(n_cols) if j != c]
                if not other_cols:
                    continue
                
                other_mat = mat_arr[:, other_cols]
                col_c = mat_arr[:, c]
                
                # Solve other_mat * x = col_c
                try:
                    # Use least squares
                    if np.linalg.matrix_rank(other_mat, tol=tol) < len(other_cols):
                        # Underdetermined, use lstsq
                        x, residuals, _, _ = np.linalg.lstsq(other_mat, col_c, rcond=tol)
                        if residuals.size > 0 and residuals[0] > tol * (np.linalg.norm(col_c) + 1e-30):
                            continue
                    else:
                        x = np.linalg.solve(other_mat, col_c)
                        if np.linalg.norm(other_mat @ x - col_c) > tol * (np.linalg.norm(col_c) + 1e-30):
                            continue
                    
                    # Column c is redundant
                    redundant.add(c)
                except np.linalg.LinAlgError:
                    continue
            
            return redundant
        
        # Try reducing U
        U_redundant = find_redundant_columns(U, tol)
        V_redundant = find_redundant_columns(V, tol)
        W_redundant = find_redundant_columns(W, tol)
        
        # If any matrix has redundant columns, we can potentially reduce
        # But we need to be careful: reducing one matrix doesn't automatically
        # reduce the rank unless the dependencies are consistent.
        
        # For now, if we find that all three have the same number of redundant columns
        # and they correspond to the same indices, we can drop those columns.
        
        if U_redundant or V_redundant or W_redundant:
            # Find common redundant indices
            common = U_redundant & V_redundant & W_redundant
            if common:
                keep = [i for i in range(rank_orig) if i not in common]
                U_new = [[U[i][j] for j in keep] for i in range(len(U))]
                V_new = [[V[i][j] for j in keep] for i in range(len(V))]
                W_new = [[W[i][j] for j in keep] for i in range(len(W))]
                
                # Verify exactness
                if _exact(_tensor_from(U_new, V_new, W_new)):
                    return U_new, V_new, W_new
        
        return None

    def _pairwise_merge_reduce(U, V, W, tol=1e-10):
        """
        Try to merge pairs of terms by checking if their combined contribution
        can be expressed with fewer terms.
        """
        rank = len(U[0]) if U else 0
        if rank < 2:
            return None
        
        # For each pair (c1, c2), check if we can replace them with a single term
        # or if one can be absorbed into the other.
        
        # Simple approach: check if u_c1 is proportional to u_c2, etc.
        def is_proportional(a, b, tol=1e-10):
            a_arr = np.array(a, dtype=float)
            b_arr = np.array(b, dtype=float)
            if np.allclose(a_arr, 0, atol=tol):
                return np.allclose(b_arr, 0, atol=tol)
            if np.allclose(b_arr, 0, atol=tol):
                return False
            
            # Find scale factor
            mask = np.abs(a_arr) > tol
            if not np.any(mask):
                return False
            
            scales = b_arr[mask] / a_arr[mask]
            if not np.allclose(scales, scales[0], atol=tol, rtol=tol):
                return False
            
            scale = scales[0]
            return np.allclose(b_arr, scale * a_arr, atol=tol, rtol=tol)
        
        # Try to find if any two terms can be merged
        merged = False
        for c1 in range(rank):
            for c2 in range(c1 + 1, rank):
                # Check if u_c1 proportional to u_c2
                u1 = U[:, c1] if isinstance(U[0], list) else [U[i][c1] for i in range(len(U))]
                u2 = U[:, c2] if isinstance(U[0], list) else [U[i][c2] for i in range(len(U))]
                v1 = V[:, c1] if isinstance(V[0], list) else [V[i][c1] for i in range(len(V))]
                v2 = V[:, c2] if isinstance(V[0], list) else [V[i][c2] for i in range(len(V))]
                w1 = W[:, c1] if isinstance(W[0], list) else [W[i][c1] for i in range(len(W))]
                w2 = W[:, c2] if isinstance(W[0], list) else [W[i][c2] for i in range(len(W))]
                
                if is_proportional(u1, u2, tol) and is_proportional(v1, v2, tol) and is_proportional(w1, w2, tol):
                    # These two terms are proportional, can merge
                    # Drop c2
                    keep = [i for i in range(rank) if i != c2]
                    U_new = [[U[i][j] for j in keep] for i in range(len(U))]
                    V_new = [[V[i][j] for j in keep] for i in range(len(V))]
                    W_new = [[W[i][j] for j in keep] for i in range(len(W))]
                    
                    if _exact(_tensor_from(U_new, V_new, W_new)):
                        return U_new, V_new, W_new
                    merged = True
                    break
            if merged:
                break
        
        return None

    U0, V0, W0 = _decompose_terms()
    if not _exact(_tensor_from(U0, V0, W0)):
        return {"U": U0, "V": V0, "W": W0}

    best_result = (U0, V0, W0)
    best_rank = len(U0[0])
    
    if _remaining() > 2.0:
        # Try SVD-based reduction
        reduced = _svd_reduce(U0, V0, W0)
        if reduced is not None:
            rU, rV, rW = reduced
            if len(rW[0]) < best_rank and _exact(_tensor_from(rU, rV, rW)):
                best_result = (rU, rV, rW)
                best_rank = len(rW[0])
    
    if _remaining() > 2.0:
        # Try pairwise merge reduction
        reduced = _pairwise_merge_reduce(best_result[0], best_result[1], best_result[2])
        if reduced is not None:
            rU, rV, rW = reduced
            if len(rW[0]) < best_rank and _exact(_tensor_from(rU, rV, rW)):
                best_result = (rU, rV, rW)
                best_rank = len(rW[0])
    
    # Iterative reduction: keep trying to reduce
    for _ in range(5):
        if _remaining() < 1.0:
            break
        
        rU, rV, rW = best_result
        reduced = _pairwise_merge_reduce(rU, rV, rW)
        if reduced is not None:
            nrU, nrV, nrW = reduced
            if len(nrW[0]) < best_rank and _exact(_tensor_from(nrU, nrV, nrW)):
                best_result = (nrU, nrV, nrW)
                best_rank = len(nrW[0])
            else:
                break
        else:
            break
    
    return {"U": best_result[0], "V": best_result[1], "W": best_result[2]}
# EVOLVE-BLOCK-END
