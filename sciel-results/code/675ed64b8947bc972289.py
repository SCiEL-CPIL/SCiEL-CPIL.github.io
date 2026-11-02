"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import numpy as np

DIMENSIONS = [2, 4, 6]
_START_TIME = time.time()
_DEADLINE = _START_TIME + 115  # 5s safety margin before 120s kill

# EVOLVE-BLOCK-START

def _build_tensor(n, m, p):
    """Build the matrix-multiplication tensor T[i,j,k] where T[i,m*j+k, p*n+i] = 1
    Actually: T[a, b, c] = 1 iff a = i, b = m*j+k, c = p*n+i... 
    Let's use the standard: T[i, j, k] for the tensor where
    (A*B)_{i,k} = sum_j A_{i,j} B_{j,k}
    Tensor T[i, j, k] = delta (it's 1 for all valid i,j,k in the product sense)
    
    Better formulation: The multiplication tensor has indices (i, j, k) where
    i in [0,n), j in [0,m), k in [0,p)
    T[i, j, k] = 1 for all i,j,k (it's the identity tensor for this multiplication)
    
    Wait, that's not right either. Let me think carefully.
    
    The tensor for matrix multiplication C = A*B where A is n×m, B is m×p, C is n×p:
    C_{i,k} = sum_j A_{i,j} B_{j,k}
    
    The tensor T has three modes:
    - Mode 1: (i, j) for A, size n*m
    - Mode 2: (j, k) for B, size m*p  
    - Mode 3: (i, k) for C, size n*p
    
    T[(i,j), (j',k), (i',k')] = 1 if i=i' and j=j' and k=k', else 0
    
    So T is essentially a "diagonal" tensor of size (n*m) × (m*p) × (n*p) with
    exactly n*m*p ones on the "diagonal" where all three indices match appropriately.
    """
    n, m, p = DIMENSIONS
    # T[a, b, c] = 1 iff:
    #   a = i*m + j, b = j'*p + k, c = i'*n + k'
    #   and i = i', j = j', k = k'
    # So: a = i*m+j, b = j*p+k, c = i*n+k where i,j,k are the same
    T = np.zeros((n*m, m*p, n*p), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = i * n + k
                T[a, b, c] = 1.0
    return T


def _check_decomposition(U, V, W, T, tol=1e-9):
    """Check if sum_r U[:,r] * V[:,r] * W[:,r] == T"""
    rank = U.shape[1]
    # Compute the tensor as outer sum
    # T_reconstructed[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # Use einsum for efficiency
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(T_recon, T, atol=tol)


def _try_remove_term(U, V, W, T, col, tol=1e-9):
    """Try removing column 'col' by adjusting other columns.
    Returns (success, new_U, new_V, new_W) or (False, None, None, None)
    
    Strategy: If we remove column r, we need to absorb its contribution into
    other columns. This is possible if the outer product U[:,r]⊗V[:,r]⊗W[:,r]
    can be expressed as a linear combination of the remaining outer products.
    
    A simpler approach: try setting the column to zero and check if the
    tensor can still be recovered by adjusting a few other columns.
    
    Actually, the cleanest approach: for each pair of columns (r, s), check if
    we can merge them. Or: for each column r, check if removing it and solving
    for adjustments to one other column s works.
    """
    rank = U.shape[1]
    if rank <= 1:
        return False, None, None, None
    
    n_rows_U, n_rows_V, n_rows_W = U.shape[0], V.shape[0], W.shape[0]
    
    # Try removing column 'col' and see if the residual can be absorbed
    # by modifying one other column.
    # Residual after removing col: R = U[:,col]*V[:,col]*W[:,col]
    # We need: R = sum_{s != col} delta_U_s * V[:,s] * W[:,s] + ... 
    # This is complex. Let's use a simpler strategy:
    
    # Strategy: For each other column s, check if we can find scalars such that
    # U[:,col]*V[:,col]*W[:,col] = alpha * U[:,s]*V[:,s]*W[:,s]
    # This only works for very specific cases.
    
    # Better strategy: Try to find if column 'col' is a linear combination
    # in the tensor sense. We'll use a rank-reduction approach:
    # 
    # The tensor T = sum_r u_r ⊗ v_r ⊗ w_r
    # If we can find coefficients such that for some r:
    # u_r ⊗ v_r ⊗ w_r = sum_{s≠r} c_s * (u_s' ⊗ v_s' ⊗ w_s')
    # where u_s', v_s', w_s' are modified versions...
    #
    # Simplest effective approach: try removing each column and checking
    # if the remaining rank-1 tensors span the original tensor.
    # We can do this by checking if the tensor without column r has the
    # same "support" structure.
    
    # Most practical: try to eliminate a column by solving a small linear system
    # For column r to be removable, we need:
    # u_r ⊗ v_r ⊗ w_r ∈ span{u_s ⊗ v_s ⊗ w_s : s ≠ r}
    #
    # This is a subspace membership test. We can check by seeing if
    # u_r ⊗ v_r ⊗ w_r is in the span of the other rank-1 tensors.
    
    # Build the matrix of other terms flattened
    other_cols = [c for c in range(rank) if c != col]
    if not other_cols:
        return False, None, None, None
    
    # Flatten all rank-1 tensors
    target = np.kron(np.kron(U[:, col], V[:, col]), W[:, col])
    basis = np.array([np.kron(np.kron(U[:, c], V[:, c]), W[:, c]) for c in other_cols])
    
    # Check if target is in span of basis
    try:
        # Use least squares
        coeffs, residuals, rank_basis, sv = np.linalg.lstsq(basis.T, target, rcond=None)
        recon = basis.T @ coeffs
        if np.allclose(recon, target, atol=tol):
            # Column col is redundant! Remove it.
            new_U = np.delete(U, col, axis=1)
            new_V = np.delete(V, col, axis=1)
            new_W = np.delete(W, col, axis=1)
            return True, new_U, new_V, new_W
    except:
        pass
    
    return False, None, None, None


def _try_merge_columns(U, V, W, T, col1, col2, tol=1e-9):
    """Try to merge two columns into one by finding a rank-1 tensor
    that replaces both. This requires:
    u1⊗v1⊗w1 + u2⊗v2⊗w2 = u⊗v⊗w for some u,v,w
    
    This is possible if the sum of two rank-1 tensors has rank 1,
    which happens when they are proportional.
    """
    # Check if the two rank-1 tensors are proportional
    t1 = np.kron(np.kron(U[:, col1], V[:, col1]), W[:, col1])
    t2 = np.kron(np.kron(U[:, col2], V[:, col2]), W[:, col2])
    
    # Check proportionality
    if np.allclose(t1, 0, atol=tol) or np.allclose(t2, 0, atol=tol):
        return False, None, None, None
    
    # Check if t1 = alpha * t2 for some alpha
    # Use first non-zero element
    mask = np.abs(t1) > tol
    if not np.any(mask):
        return False, None, None, None
    
    idx = np.argmax(mask)
    alpha = t1[idx] / t2[idx]
    
    if np.allclose(t1, alpha * t2, atol=tol):
        # They are proportional, so their sum is also rank-1
        # New term: (1+alpha) * t2 (or t1)
        # We can just keep one and scale it
        new_U = np.delete(U, col2, axis=1)
        new_V = np.delete(V, col2, axis=1)
        new_W = np.delete(W, col2, axis=1)
        # Scale the remaining column by (1 + alpha)
        new_U[:, col1] *= (1 + alpha)
        return True, new_U, new_V, new_W
    
    return False, None, None, None


def construct():
    n, m, p = DIMENSIONS
    rank0 = n * m * p  # 48
    
    # Build initial schoolbook decomposition
    U = np.zeros((n * m, rank0), dtype=np.float64)
    V = np.zeros((m * p, rank0), dtype=np.float64)
    W = np.zeros((p * n, rank0), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    # Build the target tensor
    T = _build_tensor(n, m, p)
    
    # Verify initial decomposition
    if not _check_decomposition(U, V, W, T):
        # Fallback to schoolbook as list
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank0
    
    # Iteratively try to reduce rank
    # Phase 1: Try removing redundant columns
    current_U, current_V, current_W = U.copy(), V.copy(), W.copy()
    current_rank = rank0
    
    max_iterations = 100
    iter_count = 0
    
    while time.time() < _DEADLINE and current_rank > 10 and iter_count < max_iterations:
        iter_count += 1
        found_reduction = False
        
        # Try removing each column
        for col_idx in range(current_rank):
            if time.time() > _DEADLINE:
                break
            success, new_U, new_V, new_W = _try_remove_term(
                current_U, current_V, current_W, T, col_idx
            )
            if success:
                # Verify the new decomposition
                if _check_decomposition(new_U, new_V, new_W, T):
                    current_U, current_V, current_W = new_U, new_V, new_W
                    current_rank -= 1
                    best_U, best_V, best_W = current_U.copy(), current_V.copy(), current_W.copy()
                    best_rank = current_rank
                    found_reduction = True
                    break  # Restart column scan after a reduction
        
        if not found_reduction:
            # Try merging proportional columns
            merged = False
            for c1 in range(current_rank):
                if time.time() > _DEADLINE:
                    break
                for c2 in range(c1 + 1, current_rank):
                    if time.time() > _DEADLINE:
                        break
                    success, new_U, new_V, new_W = _try_merge_columns(
                        current_U, current_V, current_W, T, c1, c2
                    )
                    if success:
                        if _check_decomposition(new_U, new_V, new_W, T):
                            current_U, current_V, current_W = new_U, new_V, new_W
                            current_rank -= 1
                            best_U, best_V, best_W = current_U.copy(), current_V.copy(), current_W.copy()
                            best_rank = current_rank
                            merged = True
                            break
                if merged:
                    break
            
            if not merged:
                break  # No more reductions possible
    
    # Final check
    if _check_decomposition(best_U, best_V, best_W, T):
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        # Fallback to original
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
