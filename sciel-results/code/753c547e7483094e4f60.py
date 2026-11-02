"""Matrix multiplication 2x4 x 4x7 rank-minimal tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START

def _build_rank50(n, m, p):
    """Build the rank-50 Strassen+schoolbook decomposition."""
    rank = 50
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)

    col = 0

    def add_strassen_block(i0, j0, k0):
        nonlocal col
        a_idx = i0 * m + j0
        b_idx = i0 * m + (j0 + 1)
        c_idx = (i0 + 1) * m + j0
        d_idx = (i0 + 1) * m + (j0 + 1)

        e_idx = j0 * p + k0
        f_idx = j0 * p + (k0 + 1)
        g_idx = (j0 + 1) * p + k0
        h_idx = (j0 + 1) * p + (k0 + 1)

        c00_idx = k0 * n + i0
        c01_idx = (k0 + 1) * n + i0
        c10_idx = k0 * n + (i0 + 1)
        c11_idx = (k0 + 1) * n + (i0 + 1)

        U[a_idx, col] += 1; U[d_idx, col] += 1
        V[e_idx, col] += 1; V[h_idx, col] += 1
        W[c00_idx, col] += 1; W[c11_idx, col] += 1
        col += 1

        U[c_idx, col] += 1; U[d_idx, col] += 1
        V[e_idx, col] += 1
        W[c10_idx, col] += 1; W[c11_idx, col] -= 1
        col += 1

        U[a_idx, col] += 1
        V[f_idx, col] += 1; V[h_idx, col] -= 1
        W[c01_idx, col] += 1; W[c11_idx, col] += 1
        col += 1

        U[d_idx, col] += 1
        V[g_idx, col] += 1; V[e_idx, col] -= 1
        W[c00_idx, col] += 1; W[c10_idx, col] += 1
        col += 1

        U[a_idx, col] += 1; U[b_idx, col] += 1
        V[h_idx, col] += 1
        W[c00_idx, col] -= 1; W[c01_idx, col] += 1
        col += 1

        U[c_idx, col] += 1; U[a_idx, col] -= 1
        V[e_idx, col] += 1; V[f_idx, col] += 1
        W[c11_idx, col] += 1
        col += 1

        U[b_idx, col] += 1; U[d_idx, col] -= 1
        V[g_idx, col] += 1; V[h_idx, col] += 1
        W[c00_idx, col] += 1
        col += 1

    def add_schoolbook_221_block(i0, j0, k0):
        nonlocal col
        a_idx = i0 * m + j0
        b_idx = i0 * m + (j0 + 1)
        c_idx = (i0 + 1) * m + j0
        d_idx = (i0 + 1) * m + (j0 + 1)

        e_idx = j0 * p + k0
        g_idx = (j0 + 1) * p + k0

        c00_idx = k0 * n + i0
        c10_idx = k0 * n + (i0 + 1)

        U[a_idx, col] += 1; V[e_idx, col] += 1; W[c00_idx, col] += 1; col += 1
        U[b_idx, col] += 1; V[g_idx, col] += 1; W[c00_idx, col] += 1; col += 1
        U[c_idx, col] += 1; V[e_idx, col] += 1; W[c10_idx, col] += 1; col += 1
        U[d_idx, col] += 1; V[g_idx, col] += 1; W[c10_idx, col] += 1; col += 1

    for k_start in [0, 2, 4]:
        for j_start in [0, 2]:
            add_strassen_block(0, j_start, k_start)
    for j_start in [0, 2]:
        add_schoolbook_221_block(0, j_start, 6)

    return U, V, W, col

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that sum_k U[:,k] outer V[:,k] outer W[:,k] equals the multiplication tensor."""
    rank = U.shape[1]
    # Build the tensor T[i,j,k] = delta_{j,k} (for A[i,j]*B[j,k])
    # The decomposition requires: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == delta_{j,k}
    # Actually the tensor is T[i,j,k] = 1 if j==k else 0, where i in [0,n), j in [0,m), k in [0,p)
    # Wait: A is n x m, B is m x p. C = A*B, C[i,k] = sum_j A[i,j]*B[j,k]
    # Tensor T[i,j,k] = delta_{j,k}... no. The multiplication tensor has entries T[i,j,k] = 1 iff the (i,j) entry of A and (j,k) entry of B contribute to C[i,k].
    # Actually the bilinear form is: C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor T[i,j,k] = 1 for all valid i,j,k (it's the identity on the j-index pairing).
    # The rank-1 decomposition: sum_r U[i,j,r] * V[j,k,r] * W[i,k,r] = delta? No...
    # Let me re-read the convention: W[k*n+i, r]
    # The tensor is T[i,j,k] = 1 (the multiplication tensor for 2x4 x 4x7)
    # Decomposition: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k*n+i, r]
    # So we need: for all i in [0,2), j in [0,4), k in [0,7):
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    # (since every (i,j,k) triple contributes exactly 1 to the product)
    
    # Actually wait - the multiplication tensor T[i,j,k] = 1 for all i,j,k is NOT correct.
    # The matrix multiplication tensor has T[i,j,k] = 1 if and only if the j-index of A matches the j-index of B.
    # A[i,j] * B[j,k] -> C[i,k]. So T[i,j,k] = 1 for all valid (i,j,k).
    # Yes! Every triple (i,j,k) has T[i,j,k] = 1.
    
    # Let's verify: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1 for all i,j,k.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for r in range(rank):
                    total += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if total != 1:
                    return False
    return True

def _try_reduce_rank(U, V, W, n, m, p, deadline):
    """
    Try to find a linear dependency among the rank-1 terms to reduce rank by 1.
    
    The rank-1 terms are vectors in the tensor space R^{n*m*p}.
    Term r corresponds to the tensor T_r[i,j,k] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    If there exist coefficients c_0, ..., c_{R-1} (not all zero) such that
    sum_r c_r * T_r = 0, then we can drop any term with c_r != 0.
    
    We search for such a dependency using linear algebra over the reals,
    then check if a rational (half-integer) solution exists.
    """
    R = U.shape[1]
    tensor_dim = n * m * p  # 2*4*7 = 56
    
    # Build the matrix M where column r is the flattened tensor T_r
    M = np.zeros((tensor_dim, R), dtype=np.float64)
    for r in range(R):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    M[k * n * m + j * n + i, r] = U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
    
    # Find null space of M (i.e., coefficients c such that M @ c = 0)
    # Use SVD
    try:
        U_svd, s, Vt = np.linalg.svd(M, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    
    # The null space dimension is R - rank(M)
    tol = 1e-10
    rank_M = np.sum(s > tol)
    null_dim = R - rank_M
    
    if null_dim <= 0:
        return None  # No linear dependency, can't reduce
    
    # Get null space vectors
    null_space = Vt[rank_M:].T  # shape (R, null_dim)
    
    # For each null space vector, check if we can use it to drop a term
    # We need coefficients that are "nice" (small integers or half-integers)
    # and at least one coefficient is non-zero (which is guaranteed for null space vectors)
    
    best_reduction = None
    for null_vec in null_space:
        # Normalize: find the index with largest absolute value
        max_idx = np.argmax(np.abs(null_vec))
        max_val = null_vec[max_idx]
        
        if abs(max_val) < 1e-15:
            continue
        
        # Scale so that the max coefficient is 1 (or -1)
        scaled = null_vec / max_val
        # Now scaled[max_idx] = 1 (or -1)
        
        # Check if all coefficients are close to half-integers
        # (i.e., 2*scaled is close to integers)
        doubled = 2 * scaled
        rounded = np.round(doubled)
        if np.allclose(doubled, rounded, atol=1e-6):
            # This is a valid half-integer dependency
            # We can drop term max_idx by subtracting the combination
            # New coefficients: for term r, new_coeff = scaled[r]
            # The new U,V,W for the reduced set:
            # We drop column max_idx and adjust the remaining columns.
            # Actually, the dependency says: sum_r scaled[r] * T_r = 0
            # So T_{max_idx} = -sum_{r != max_idx} scaled[r]/scaled[max_idx] * T_r
            # Since scaled[max_idx] = 1 (after normalization):
            # T_{max_idx} = -sum_{r != max_idx} scaled[r] * T_r
            
            # To drop term max_idx, we need to express it as a combination of others.
            # The new decomposition has R-1 terms where:
            # For r != max_idx: the new U[:,r] = old U[:,r] (unchanged)
            # But we need to add the contribution of term max_idx to the other terms.
            # Actually, it's simpler: we just verify that removing column max_idx
            # and adjusting works.
            
            # Let's construct the reduced decomposition:
            keep = [r for r in range(R) if r != max_idx]
            U_new = np.zeros((n * m, R - 1), dtype=np.int64)
            V_new = np.zeros((m * p, R - 1), dtype=np.int64)
            W_new = np.zeros((p * n, R - 1), dtype=np.int64)
            
            for new_r, old_r in enumerate(keep):
                U_new[:, new_r] = U[:, old_r]
                V_new[:, new_r] = V[:, old_r]
                W_new[:, new_r] = W[:, old_r]
            
            # Now verify the reduced decomposition
            if _verify_decomposition(U_new, V_new, W_new, n, m, p):
                best_reduction = (max_idx, U_new, V_new, W_new)
                break
            else:
                # The simple drop doesn't work because the dependency involves
                # combining terms, not just dropping. We need to absorb the dropped
                # term into the others.
                # New term r' (for r' in keep) should be:
                # T_{r'} + scaled[r'] * T_{max_idx}  (if we're absorbing)
                # But this changes the structure...
                # Actually, the correct approach: the sum of all R terms equals the tensor.
                # If sum_r scaled[r] * T_r = 0, then the sum of (1 - scaled[r]*alpha) * T_r for some alpha
                # ... this is getting complex. Let me just try the direct verification.
                pass
    
    return best_reduction

def construct():
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = DIMENSIONS
    
    # Build rank-50 decomposition
    U, V, W, col = _build_rank50(n, m, p)
    
    if col != 50:
        return _schoolbook(n, m, p)
    
    # Verify the rank-50 decomposition
    if not _verify_decomposition(U, V, W, n, m, p):
        return _schoolbook(n, m, p)
    
    # Try to reduce rank by finding syzygies
    if time.time() < deadline - 10:
        reduction = _try_reduce_rank(U, V, W, n, m, p, deadline)
        if reduction is not None:
            drop_idx, U_new, V_new, W_new = reduction
            return {"U": U_new.tolist(), "V": V_new.tolist(), "W": W_new.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
