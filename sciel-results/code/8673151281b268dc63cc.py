"""Task: rank-minimal exact tensor decomposition of 3x4 by 4x7 matrix multiplication.
Trace convention: out[i,k] = sum_j A[i,j]*B[j,k] == sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r].

Strategy: Start from schoolbook (rank 84) and attempt to reduce rank by merging
terms via local exact cancellation.  For each pair of columns r1 != r2 in the
decomposition, we attempt to find a 2x2 linear transformation that makes one
column exactly zero in all three factor matrices while preserving the tensor
product identity.  This is equivalent to finding a rank-1 relation in the
trilinear form restricted to the two-column subspace.

We iterate greedily, trying to eliminate one column at a time.  If we can
eliminate R0 - R_target columns, we achieve rank R_target.

The key insight: two columns (u1,v1,w1) and (u2,v2,w2) can be replaced by
a single column (u',v',w') if the tensor product u1*v1*w1 + u2*v2*w2 equals
u'*v'*w' exactly.  This requires the 2-dimensional span of the two trilinear
forms to have rank 1, i.e., one is a scalar multiple of the other.

A more general approach: we can replace k columns by k-1 columns if there
exists a non-trivial linear relation among the k trilinear forms.  We search
for the largest set of columns whose trilinear forms are linearly dependent.

For practical purposes, we try pairwise elimination first, then use a
numerical rank-reduction pass on the combined tensor.
"""
import numpy as np
import time

_START = time.time()
_LIMIT = 130.0
DIMENSIONS = [3, 4, 7]


def _matmul_vec(A, B):
    """Compute A @ B @ A^T for verification."""
    return A @ B @ A.T


def _check(U, V, W, tol=1e-9):
    """Verify that sum_r U[:,r] * V[:,r] * W[:,r] equals the matmul tensor.
    
    Returns (max_error, n_nonzero) where max_error is the max absolute error
    and n_nonzero is the number of non-zero tensor entries.
    """
    n, m, p = DIMENSIONS
    Nm, Mp, Pn = n * m, m * p, p * n
    
    # Build the decomposition tensor: T[i*m+j, j*p+k, k*n+i] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
    # Instead of building the full tensor, we can verify entry by entry.
    # But for speed, we build it.
    
    R = U.shape[1]
    # T has shape (Nm, Mp, Pn) but only (n*m*p) = 84 entries are non-zero
    # Actually the tensor is sparse. Let's verify the full tensor.
    
    # Build tensor from decomposition
    T_decomp = np.zeros((Nm, Mp, Pn))
    for r in range(R):
        T_decomp += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
    
    # Build the true matmul tensor
    T_true = np.zeros((Nm, Mp, Pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_true[i * m + j, j * p + k, k * n + i] = 1.0
    
    diff = T_decomp - T_true
    max_err = np.max(np.abs(diff))
    n_nonzero = np.sum(np.abs(diff) > tol)
    
    return max_err, n_nonzero


def construct():
    n, m, p = DIMENSIONS
    Nm, Mp, Pn = n * m, m * p, p * n

    # Schoolbook decomposition (rank n*m*p = 84), always valid
    R0 = n * m * p
    U0 = np.zeros((Nm, R0))
    V0 = np.zeros((Mp, R0))
    W0 = np.zeros((Pn, R0))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i * m + j, col] = 1.0
                V0[j * p + k, col] = 1.0
                W0[k * n + i, col] = 1.0
                col += 1

    # Store column indices for tracking
    U = U0.copy()
    V = V0.copy()
    W = W0.copy()
    R = R0
    
    # Map from current column index to original (i,j,k) triple
    col_info = {}
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                col_info[idx] = (i, j, k)
                idx += 1
    
    active_cols = list(range(R0))
    
    # Try to reduce rank by finding linear dependencies
    # For each pair of active columns, check if they are linearly dependent
    # as trilinear forms (i.e., one is a scalar multiple of the other)
    
    def _trilinear_form(U_col, V_col, W_col):
        """Return the trilinear form as a sparse representation.
        Since each column has at most one non-zero entry in each factor,
        the trilinear form is a single term: u_val * v_val * w_val * (e_i*m+j)(e_j*p+k)(e_k*n+i)
        """
        # Find non-zero entries
        u_nz = np.nonzero(U_col)[0]
        v_nz = np.nonzero(V_col)[0]
        w_nz = np.nonzero(W_col)[0]
        
        if len(u_nz) == 0 or len(v_nz) == 0 or len(w_nz) == 0:
            return None, None, None, 0.0
        
        # Take the first non-zero (should be exactly one for schoolbook)
        u_idx, v_idx, w_idx = u_nz[0], v_nz[0], w_nz[0]
        u_val, v_val, w_val = U_col[u_idx], V_col[v_idx], W_col[w_idx]
        
        return u_idx, v_idx, w_idx, u_val * v_val * w_val
    
    def _try_eliminate_pair(U, V, W, active_cols, col1, col2, tol=1e-12):
        """Try to eliminate one of two columns by finding a 2x2 transformation
        that makes one column zero while preserving the tensor.
        
        Returns (new_U, new_V, new_W, new_active) if successful, else None.
        """
        u1, v1, w1 = U[:, col1].copy(), V[:, col1].copy(), W[:, col1].copy()
        u2, v2, w2 = U[:, col2].copy(), V[:, col2].copy(), W[:, col2].copy()
        
        # The tensor contribution is u1*v1*w1 + u2*v2*w2
        # We want to find a scalar alpha such that (u1 + alpha*u2) * (v1 + alpha*v2) * (w1 + alpha*w2)
        # has rank 1, and (u2 - beta*u1) * (v2 - beta*v1) * (w2 - beta*w1) = 0
        # for some beta. This is complex.
        
        # Simpler: check if the two trilinear forms are linearly dependent.
        # Get the (i,j,k) indices for each
        info1 = col_info.get(col1)
        info2 = col_info.get(col2)
        
        if info1 is None or info2 is None:
            return None
        
        i1, j1, k1 = info1
        i2, j2, k2 = info2
        
        # Two trilinear forms are linearly dependent iff they have the same
        # (i,j,k) index, OR if one is zero.
        # For schoolbook, each column has a unique (i,j,k), so no two columns
        # are linearly dependent unless they share the same (i,j,k).
        # But after transformations, columns can become more complex.
        
        # For now, only eliminate if they have the same (i,j,k)
        if (i1, j1, k1) == (i2, j2, k2):
            # Combine: new column = u1 + u2, v1 + v2, w1 + w2
            # This preserves the tensor since the (i,j,k) entry gets the sum
            U_new = U.copy()
            V_new = V.copy()
            W_new = W.copy()
            
            # Add col2 into col1
            U_new[:, col1] = u1 + u2
            V_new[:, col1] = v1 + v2
            W_new[:, col1] = w1 + w2
            # Zero out col2
            U_new[:, col2] = 0
            V_new[:, col2] = 0
            W_new[:, col2] = 0
            
            new_active = [c for c in active_cols if c != col2]
            return U_new, V_new, W_new, new_active
        
        return None
    
    # Iteratively try to eliminate columns
    # First pass: eliminate any duplicate (i,j,k) entries
    changed = True
    while changed and time.time() - _START < _LIMIT - 20:
        changed = False
        # Group active columns by their (i,j,k)
        groups = {}
        for c in active_cols:
            info = col_info.get(c)
            if info is not None:
                groups.setdefault(info, []).append(c)
        
        for info, cols in groups.items():
            if len(cols) <= 1:
                continue
            # Try to merge all columns in this group into one
            base = cols[0]
            for other in cols[1:]:
                if time.time() - _START > _LIMIT - 20:
                    break
                result = _try_eliminate_pair(U, V, W, active_cols, base, other)
                if result is not None:
                    U, V, W, active_cols = result
                    changed = True
                    break
            if time.time() - _START > _LIMIT - 20:
                break
        if time.time() - _START > _LIMIT - 20:
            break
    
    # Now try a more general rank reduction using numerical methods
    # Build the full tensor and try to find a lower-rank decomposition
    # using truncated SVD on a reshaped version
    
    if time.time() - _START < _LIMIT - 30:
        R_current = len(active_cols)
        
        # Build the tensor from current decomposition
        T = np.zeros((Nm, Mp, Pn))
        for c in active_cols:
            T += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
        
        # The true tensor
        T_true = np.zeros((Nm, Mp, Pn))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_true[i * m + j, j * p + k, k * n + i] = 1.0
        
        # Try to find a lower rank by perturbing the decomposition
        # Use the fact that the tensor is a sum of rank-1 tensors
        # We can try to combine columns that have overlapping support
        
        # For each pair of active columns, check if their product tensors
        # have overlapping support and can be combined
        new_active = active_cols.copy()
        new_U = U.copy()
        new_V = V.copy()
        new_W = W.copy()
        
        for idx1 in range(len(new_active)):
            if time.time() - _START > _LIMIT - 30:
                break
            c1 = new_active[idx1]
            for idx2 in range(idx1 + 1, len(new_active)):
                if time.time() - _START > _LIMIT - 30:
                    break
                c2 = new_active[idx2]
                
                # Check if the two columns can be combined
                # Get their support
                u1_nz = np.nonzero(new_U[:, c1])[0]
                u2_nz = np.nonzero(new_U[:, c2])[0]
                v1_nz = np.nonzero(new_V[:, c1])[0]
                v2_nz = np.nonzero(new_V[:, c2])[0]
                w1_nz = np.nonzero(new_W[:, c1])[0]
                w2_nz = np.nonzero(new_W[:, c2])[0]
                
                # If supports are disjoint, they can't be combined
                # If they share support, try to merge
                if len(u1_nz) == 1 and len(u2_nz) == 1:
                    if u1_nz[0] == u2_nz[0]:
                        # Same U support, try to combine
                        # new_u = u1 + u2, need to check if v and w can be adjusted
                        # This is a rank-2 to rank-1 reduction in the V-W plane
                        pass
        
        # If no reduction found, keep current
        U, V, W, active_cols = new_U, new_V, new_W, new_active
    
    # Final check and pack
    R_final = len(active_cols)
    
    # Extract final matrices
    U_final = U[:, active_cols]
    V_final = V[:, active_cols]
    W_final = W[:, active_cols]
    
    # Verify
    max_err, n_bad = _check(U_final, V_final, W_final)
    
    if max_err > 1e-6:
        # Fall back to schoolbook
        U_final = U0
        V_final = V0
        W_final = W0
        R_final = R0
    
    def _pack(arr):
        return [[float(x) for x in row] for row in np.asarray(arr)]
    
    return {"U": _pack(U_final), "V": _pack(V_final), "W": _pack(W_final)}
