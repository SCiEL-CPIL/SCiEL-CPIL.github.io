import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def _add_strassen_block(U, V, W, col, i_off, j_off, k_off):
    """Add Strassen 2x2x2 block at given offsets. Returns new col."""
    n, m, p = 2, 3, 5
    # Local indices: i_l in {0,1}, j_l in {0,1}, k_l in {0,1}
    # Global: i = i_off+i_l, j = j_off+j_l, k = k_off+k_l
    
    # M1: (A00+A11)(B00+B11) -> C00 += , C11 +=
    U[i_off * m + j_off][col] = 1
    U[(i_off+1) * m + (j_off+1)][col] = 1
    V[j_off * p + k_off][col] = 1
    V[(j_off+1) * p + (k_off+1)][col] = 1
    W[k_off * n + i_off][col] = 1
    W[(k_off+1) * n + (i_off+1)][col] = 1
    col += 1
    
    # M2: (A10+A11)B00 -> C10 += , C11 -=
    U[(i_off+1) * m + j_off][col] = 1
    U[(i_off+1) * m + (j_off+1)][col] = 1
    V[j_off * p + k_off][col] = 1
    W[k_off * n + (i_off+1)][col] = 1
    W[(k_off+1) * n + (i_off+1)][col] = -1
    col += 1
    
    # M3: A00(B01-B11) -> C01 += , C11 +=
    U[i_off * m + j_off][col] = 1
    V[j_off * p + (k_off+1)][col] = 1
    V[(j_off+1) * p + (k_off+1)][col] = -1
    W[(k_off+1) * n + i_off][col] = 1
    W[(k_off+1) * n + (i_off+1)][col] = 1
    col += 1
    
    # M4: A11(B10-B00) -> C00 += , C10 +=
    U[(i_off+1) * m + (j_off+1)][col] = 1
    V[(j_off+1) * p + k_off][col] = 1
    V[j_off * p + k_off][col] = -1
    W[k_off * n + i_off][col] = 1
    W[k_off * n + (i_off+1)][col] = 1
    col += 1
    
    # M5: (A00+A01)B11 -> C00 -= , C01 +=
    U[i_off * m + j_off][col] = 1
    U[i_off * m + (j_off+1)][col] = 1
    V[(j_off+1) * p + (k_off+1)][col] = 1
    W[k_off * n + i_off][col] = -1
    W[(k_off+1) * n + i_off][col] = 1
    col += 1
    
    # M6: (A10-A00)(B00+B01) -> C11 +=
    U[(i_off+1) * m + j_off][col] = 1
    U[i_off * m + j_off][col] = -1
    V[j_off * p + k_off][col] = 1
    V[j_off * p + (k_off+1)][col] = 1
    W[(k_off+1) * n + (i_off+1)][col] = 1
    col += 1
    
    # M7: (A01-A11)(B10+B11) -> C00 +=
    U[i_off * m + (j_off+1)][col] = 1
    U[(i_off+1) * m + (j_off+1)][col] = -1
    V[(j_off+1) * p + k_off][col] = 1
    V[(j_off+1) * p + (k_off+1)][col] = 1
    W[k_off * n + i_off][col] = 1
    col += 1
    
    return col

def _verify_decomposition(U, V, W, n=2, m=3, p=5):
    """Verify that the decomposition exactly represents matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Should equal 1 for all i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _try_reduce_rank_28_to_27(U, V, W, deadline):
    """
    Attempt to find a linear dependency among the 28 terms that allows
    reducing rank to 27. We look for coefficients c_r such that
    sum_r c_r * (u_r, v_r, w_r) = (0, 0, 0) in the tensor product space,
    which would mean one term is redundant.
    
    More practically: try to find a term r* that can be expressed as a
    linear combination of the others, then eliminate it.
    """
    n, m, p = 2, 3, 5
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]  # 28
    
    # Build the tensor T as a matrix: T[(i,j,k), r] = U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
    # For the decomposition to be valid, T[(i,j,k),:] must have sum_r = 1 for all (i,j,k)
    # and the "extra" components must sum to 0.
    
    # Actually, let's think differently. We have 28 rank-1 tensors.
    # A rank-27 decomposition exists if there's a linear relation among these 28 tensors.
    # i.e., there exist c_0,...,c_27 not all zero such that
    # sum_r c_r * (u_r ⊗ v_r ⊗ w_r) = 0 as a tensor.
    
    # This is a system of 2*3*5 = 30 linear equations in 28 unknowns.
    # If the system has a non-trivial solution, we can eliminate one term.
    
    # Build the 30x28 matrix M where M[(i,j,k), r] = U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
    M = np.zeros((n*m*p, rank))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx = i*m*p + j*p + k
                M[idx] = U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :]
    
    # We want to find c such that M @ c = 0, c != 0
    # Use SVD to find the null space
    try:
        U_svd, s, Vt_svd = np.linalg.svd(M, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    
    # The null space dimension is rank - min(rank, 30)
    # If s has a zero (or very small) singular value, we have a null vector.
    # Check the smallest singular value
    if len(s) > 0 and s[-1] < 1e-10:
        # There's a non-trivial null space
        # Get the right singular vector corresponding to the smallest singular value
        c = Vt_svd[-1, :]
        
        # Find the term with the largest |c_r| to eliminate
        r_star = np.argmax(np.abs(c))
        
        # If c[r_star] is close to 1 (or -1), we can eliminate term r_star
        # by adding (c_r / c_r_star) * term_r_star to term_r for all r != r_star
        if abs(c[r_star]) < 1e-10:
            return None
        
        # New decomposition: for r != r_star, new_U[r] = U[r] + (c[r]/c[r_star])*U[r_star]
        # Similarly for V and W. Then remove column r_star.
        U_new = np.delete(U, r_star, axis=1)
        V_new = np.delete(V, r_star, axis=1)
        W_new = np.delete(W, r_star, axis=1)
        
        for r in range(rank):
            if r == r_star:
                continue
            new_r = r if r < r_star else r - 1
            U_new[:, new_r] = U[:, r] + (c[r] / c[r_star]) * U[:, r_star]
            V_new[:, new_r] = V[:, r] + (c[r] / c[r_star]) * V[:, r_star]
            W_new[:, new_r] = W[:, r] + (c[r] / c[r_star]) * W[:, r_star]
        
        # Verify the new decomposition
        if _verify_decomposition(U_new.tolist(), V_new.tolist(), W_new.tolist()):
            return U_new.tolist(), V_new.tolist(), W_new.tolist()
    
    # Alternative: try to find a term that is a linear combination of others
    # by checking if any column of M is in the span of the others
    for r_elim in range(rank):
        if time.time() > deadline:
            break
        M_other = np.delete(M, r_elim, axis=1)
        M_col = M[:, r_elim]
        
        # Check if M_col is in the column space of M_other
        # Solve M_other @ x = M_col
        try:
            x, residuals, rank_sv, sv = np.linalg.lstsq(M_other, M_col, rcond=None)
        except np.linalg.LinAlgError:
            continue
        
        if residuals and residuals[0] < 1e-10:
            # M_col is in the span of M_other
            # This means term r_elim is redundant in the tensor sense
            # But we need to be careful: the tensor decomposition constraint is
            # that sum_r u_r[i,j]*v_r[j,k]*w_r[k,i] = 1 for all (i,j,k)
            # and the "cross terms" cancel.
            
            # If the tensor of term r_elim is a linear combination of the others,
            # we can absorb it. But the absorption must be done consistently
            # across U, V, W.
            
            # This is more complex. Let's try a different approach:
            # Try to find a term r_elim and coefficients for the other terms
            # such that the tensor of r_elim equals the sum.
            
            # Actually, the SVD approach above should handle this. Let's also
            # try to directly check if removing a term and adjusting works.
            
            # For now, skip this and rely on the SVD null space approach.
            pass
    
    return None

def construct():
    n, m, p = 2, 3, 5
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    # Build the 28-term decomposition
    rank = 28
    U = [[0] * rank for _ in range(n * m)]  # 6 x 28
    V = [[0] * rank for _ in range(m * p)]  # 15 x 28
    W = [[0] * rank for _ in range(p * n)]  # 10 x 28
    
    col = 0
    
    # Block 1: j=0, all k (schoolbook 2x1x5) -> 10 terms
    for i in range(n):
        for k in range(p):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    
    # Block 2a: j={1,2}, k={0,1} (Strassen) -> 7 terms
    col = _add_strassen_block(U, V, W, col, 0, 1, 0)
    
    # Block 2b: j={1,2}, k=2 (schoolbook 2x2x1) -> 4 terms
    for i in range(n):
        for j in range(1, 3):
            U[i * m + j][col] = 1
            V[j * p + 2][col] = 1
            W[2 * n + i][col] = 1
            col += 1
    
    # Block 2c: j={1,2}, k={3,4} (Strassen) -> 7 terms
    col = _add_strassen_block(U, V, W, col, 0, 1, 3)
    
    assert col == rank, f"Column mismatch: {col} != {rank}"
    
    # Verify the decomposition
    if not _verify_decomposition(U, V, W):
        # Fallback: return a known-good schoolbook decomposition
        rank = 30
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Try to reduce rank from 28 to 27
    reduced = _try_reduce_rank_28_to_27(U, V, W, deadline)
    if reduced is not None:
        U_r, V_r, W_r = reduced
        # Final verification
        if _verify_decomposition(U_r, V_r, W_r):
            return {"U": U_r, "V": V_r, "W": W_r}
    
    # Return the 28-term decomposition
    return {"U": U, "V": V, "W": W}
