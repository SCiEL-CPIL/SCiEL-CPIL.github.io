import numpy as np
import time

DIMENSIONS = [3, 4, 4]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-6):
    """Check if U,V,W exactly decompose the tensor (within tol)."""
    rank = U.shape[1]
    T = np.zeros((n, m, p), dtype=np.float64)
    for r in range(rank):
        u = U[:, r].reshape(n, m)
        v = V[:, r].reshape(m, p)
        w = W[:, r].reshape(p, n)
        T += np.einsum('ij,jk,ki->ijk', u, v, w)
    return np.allclose(T, 1.0, atol=tol, rtol=0)

def _construct_rank44_decomposition():
    """
    Construct a rank-44 decomposition using Strassen for 2x2x2 blocks
    and schoolbook for 1x2x2 blocks.
    Returns U, V, W as numpy arrays of shape (n*m, 44), (m*p, 44), (p*n, 44).
    """
    n, m, p = 3, 4, 4
    rank = 44
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    def add_strassen(rA, cA, rB, cB, rC):
        """Add Strassen decomposition for 2x2x2 block."""
        nonlocal col
        # M1 = (A11+A22)(B11+B22)
        U[(rA+0)*m + (cA+0), col] = 1
        U[(rA+1)*m + (cA+1), col] = 1
        V[(rB+0)*p + (cB+0), col] = 1
        V[(rB+1)*p + (cB+1), col] = 1
        W[(cB+0)*n + (rC+0), col] = 1
        W[(cB+1)*n + (rC+1), col] = 1
        col += 1
        
        # M2 = (A21+A22)B11
        U[(rA+1)*m + (cA+0), col] = 1
        U[(rA+1)*m + (cA+1), col] = 1
        V[(rB+0)*p + (cB+0), col] = 1
        W[(cB+0)*n + (rC+1), col] = 1
        W[(cB+1)*n + (rC+1), col] = -1
        col += 1
        
        # M3 = A11(B12-B22)
        U[(rA+0)*m + (cA+0), col] = 1
        V[(rB+0)*p + (cB+1), col] = 1
        V[(rB+1)*p + (cB+1), col] = -1
        W[(cB+1)*n + (rC+0), col] = 1
        W[(cB+1)*n + (rC+1), col] = 1
        col += 1
        
        # M4 = A22(B21-B11)
        U[(rA+1)*m + (cA+1), col] = 1
        V[(rB+1)*p + (cB+0), col] = 1
        V[(rB+0)*p + (cB+0), col] = -1
        W[(cB+0)*n + (rC+0), col] = 1
        W[(cB+0)*n + (rC+1), col] = 1
        col += 1
        
        # M5 = (A11+A12)B22
        U[(rA+0)*m + (cA+0), col] = 1
        U[(rA+0)*m + (cA+1), col] = 1
        V[(rB+1)*p + (cB+1), col] = 1
        W[(cB+1)*n + (rC+0), col] = 1
        W[(cB+0)*n + (rC+0), col] = -1
        col += 1
        
        # M6 = (A21-A11)(B11+B12)
        U[(rA+1)*m + (cA+0), col] = 1
        U[(rA+0)*m + (cA+0), col] = -1
        V[(rB+0)*p + (cB+0), col] = 1
        V[(rB+0)*p + (cB+1), col] = 1
        W[(cB+1)*n + (rC+1), col] = 1
        col += 1
        
        # M7 = (A12-A22)(B21+B22)
        U[(rA+0)*m + (cA+1), col] = 1
        U[(rA+1)*m + (cA+1), col] = -1
        V[(rB+1)*p + (cB+0), col] = 1
        V[(rB+1)*p + (cB+1), col] = 1
        W[(cB+0)*n + (rC+0), col] = 1
        col += 1

    def add_schoolbook_1x2x2(rA, cA, rB, cB, rC):
        """Add schoolbook for 1x2x2 block."""
        nonlocal col
        # C0 = A0*B00 + A1*B10
        U[(rA)*m + (cA+0), col] = 1
        V[(rB+0)*p + (cB+0), col] = 1
        W[(cB+0)*n + (rC), col] = 1
        col += 1
        
        # A1*B10 -> C0
        U[(rA)*m + (cA+1), col] = 1
        V[(rB+1)*p + (cB+0), col] = 1
        W[(cB+0)*n + (rC), col]  = 1
        col += 1
        
        # A0*B01 -> C1
        U[(rA)*m + (cA+0), col] = 1
        V[(rB+0)*p + (cB+1), col] = 1
        W[(cB+1)*n + (rC), col] = 1
        col += 1
        
        # A1*B11 -> C1
        U[(rA)*m + (cA+1), col] = 1
        V[(rB+1)*p + (cB+1), col] = 1
        W[(cB+1)*n + (rC), col] = 1
        col += 1

    # C[0:2, 0:2] = A[0:2, 0:2]*B[0:2, 0:2] + A[0:2, 2:4]*B[2:4, 0:2]
    add_strassen(0, 0, 0, 0, 0)
    add_strassen(0, 2, 2, 0, 0)
    
    # C[0:2, 2:4] = A[0:2, 0:2]*B[0:2, 2:4] + A[0:2, 2:4]*B[2:4, 2:4]
    add_strassen(0, 0, 0, 2, 0)
    add_strassen(0, 2, 2, 2, 0)
    
    # C[2, 0:2] = A[2, 0:2]*B[0:2, 0:2] + A[2, 2:4]*B[2:4, 0:2]
    add_schoolbook_1x2x2(2, 0, 0, 0, 2)
    add_schoolbook_1x2x2(2, 2, 2, 0, 2)
    
    # C[2, 2:4] = A[2, 0:2]*B[0:2, 2:4] + A[2, 2:4]*B[2:4, 2:4]
    add_schoolbook_1x2x2(2, 0, 0, 2, 2)
    add_schoolbook_1x2x2(2, 2, 2, 2, 2)
    
    return U, V, W

def _try_reduce_rank(U, V, W, n, m, p, max_reduction=5, deadline=None):
    """
    Try to reduce rank by finding linear dependencies among tensor terms.
    Each term r defines a rank-1 tensor T_r = u_r ⊗ v_r ⊗ w_r.
    If a linear combination of terms sums to zero, we can eliminate one term.
    
    We check: for each term r, can it be expressed as a linear combination of others?
    This is equivalent to checking if the 44 rank-1 tensors are linearly independent
    in the space R^(n*m*p).
    
    Actually, a more practical approach: try to find subsets of terms that can be
    combined. But the most direct approach is to check linear independence of the
    44 vectors in R^(n*m*p) and if dependent, reduce.
    
    However, tensor rank reduction is more subtle. Let me try a different approach:
    attempt to find a lower-rank decomposition by solving a system.
    
    For now, let's try: check if any single term is a linear combination of others.
    If term r = sum_s c_s * term_s, then we can eliminate term r.
    """
    if deadline is not None and time.time() > deadline:
        return U, V, W
    
    rank = U.shape[1]
    nmp = n * m * p
    
    # Build the matrix of all 44 rank-1 tensors flattened
    # M has shape (nmp, rank) where M[:, r] = vec(u_r ⊗ v_r ⊗ w_r)
    M = np.zeros((nmp, rank))
    for r in range(rank):
        u = U[:, r].reshape(n, m)
        v = V[:, r].reshape(m, p)
        w = W[:, r].reshape(p, n)
        # The rank-1 tensor is T[i,j,k] = u[i,j] * v[j,k] * w[k,i]
        T = np.einsum('ij,jk,ki->ijk', u, v, w)
        M[:, r] = T.flatten()
    
    # Check rank of M
    # If rank(M) < rank, there are linear dependencies
    # Use SVD to find the null space
    try:
        U_svd, S, Vt = np.linalg.svd(M, full_matrices=False)
    except np.linalg.LinAlgError:
        return U, V, W
    
    # Numerical rank
    tol = 1e-10
    num_rank = np.sum(S > tol)
    
    if num_rank >= rank:
        # All terms are linearly independent, no reduction possible via this method
        return U, V, W
    
    # There are dependencies. The null space of M^T gives combinations that sum to zero.
    # Vt has shape (num_rank, rank) for the right singular vectors.
    # The null space vectors are the last (rank - num_rank) rows of Vt.
    
    # Actually, let's think differently. If we have a dependency:
    # sum_r c_r * T_r = 0, then we can express T_{r0} = -sum_{r!=r0} (c_r/c_{r0}) T_r
    # and eliminate term r0.
    
    # Find null space of M^T (shape rank x nmp)
    # Null space vectors c satisfy M^T c = 0, i.e., sum_r c_r M[:,r] = 0
    
    # From SVD: M = U_svd S Vt
    # M^T = Vt^T S U_svd^T
    # Null space of M^T is spanned by columns of V corresponding to zero singular values
    
    # Vt has shape (min(nmp,rank), rank). The last (rank - num_rank) columns of V are null space.
    # Actually Vt is (num_svd, rank) where num_svd = min(nmp, rank)
    # The null space of M^T is the orthogonal complement of row space of M^T = column space of M
    # Which is spanned by the last (rank - num_rank) rows of Vt... no.
    
    # Let me just use lstsq or null_space approach
    # Actually, simpler: try to eliminate terms one at a time.
    
    # For each term r, check if M[:, r] is in the span of {M[:, s] : s != r}
    # This is a rank check: rank(M without col r) vs rank(M)
    
    best_U, best_V, best_W = U, V, W
    best_rank = rank
    
    # Try eliminating each term
    for r_elim in range(rank):
        if deadline is not None and time.time() > deadline:
            break
        
        # Check if term r_elim is in span of others
        M_other = np.delete(M, r_elim, axis=1)
        
        # Check if M[:, r_elim] is in column space of M_other
        # Solve M_other * x = M[:, r_elim]
        try:
            x, residuals, rank_other, _ = np.linalg.lstsq(M_other, M[:, r_elim], rcond=1e-10)
        except np.linalg.LinAlgError:
            continue
        
        # Check residual
        residual = np.linalg.norm(M_other @ x - M[:, r_elim])
        if residual < 1e-8:
            # Term r_elim is a linear combination of others
            # We can eliminate it. But we need to update U, V, W.
            # The new decomposition has rank-1 terms, but we need to express
            # the eliminated term's contribution in terms of the others.
            # Actually, if sum_s x_s T_s = T_r_elim, then T_r_elim can be removed
            # and the other terms are unchanged. But wait, that means the total sum
            # changes. We need: sum_all T_r = sum_{s != r_elim} T_s + T_r_elim
            # = sum_{s != r_elim} T_s + sum_s x_s T_s
            # = sum_{s != r_elim} (1 + x_s) T_s + x_{r_elim} T_{r_elim}... 
            # No, x is defined for s != r_elim only.
            # T_r_elim = sum_{s != r_elim} x_s T_s
            # So total = sum_{s != r_elim} T_s + sum_{s != r_elim} x_s T_s = sum_{s != r_elim} (1+x_s) T_s
            # But this changes the coefficients of U, V, W for those terms.
            # However, U, V, W are not linear in this way because each term is a
            # product u⊗v⊗w. You can't just scale the terms independently.
            
            # This approach doesn't directly work for tensor rank reduction because
            # the terms are not just vectors but rank-1 tensors with specific structure.
            # A linear dependency among the flattened tensors doesn't directly give
            # a lower-rank tensor decomposition.
            
            # Let's skip this and try a different approach.
            pass
    
    # Alternative approach: try random linear combinations of terms to see if we
    # can find a representation with fewer terms.
    # This is essentially trying to find a lower-rank decomposition.
    
    # For a more practical approach, let's try to see if we can merge terms.
    # Two terms can be "merged" if their u, v, w vectors have special structure.
    
    # For now, return the original if no simple reduction found
    return best_U, best_V, best_W

def _construct():
    """Main construction with rank reduction attempt."""
    n, m, p = 3, 4, 4
    deadline = time.time() + 100  # Safety margin
    
    # Build rank-44 decomposition
    U, V, W = _construct_rank44_decomposition()
    
    # Verify
    if not _verify_decomposition(U, V, W, n, m, p, tol=1e-6):
        # Fallback to schoolbook
        rank = n * m * p
        U_fb = np.zeros((n * m, rank))
        V_fb = np.zeros((m * p, rank))
        W_fb = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j, col] = 1
                    V_fb[j * p + k, col] = 1
                    W_fb[k * n + i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    # Try to reduce rank
    if time.time() < deadline:
        U_red, V_red, W_red = _try_reduce_rank(U, V, W, n, m, p, max_reduction=5, deadline=deadline)
        if U_red.shape[1] < U.shape[1]:
            # Verify reduced version
            if _verify_decomposition(U_red, V_red, W_red, n, m, p, tol=1e-6):
                U, V, W = U_red, V_red, W_red
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def construct():
    return _construct()
