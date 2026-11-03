"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import combinations

DIMENSIONS = [3, 3, 7]
DEADLINE = 140.0  # Safety margin before 150s kill

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Build the target tensor
    # T[i*m+j, j*p+k, k*n+i] = 1
    tensor = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                tensor[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Schoolbook decomposition
    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        return U, V, W
    
    U, V, W = schoolbook()
    best_rank = n * m * p
    best_U, best_V, best_W = U, V, W
    
    # Fast verify using tensor contraction
    def verify_fast(U_mat, V_mat, W_mat, tol=1e-9):
        """Check if U,V,W decompose the tensor using vectorized ops."""
        r = U_mat.shape[1]
        # Reconstruct tensor: sum_c U[:,c] V[:,c]^T W[:,c]^T
        # Use einsum for speed
        recon = np.einsum('ac,bc,dc->abd', U_mat, V_mat, W_mat)
        if np.max(np.abs(recon - tensor)) > tol:
            return False
        return True
    
    # Check schoolbook
    assert verify_fast(U, V, W), "Schoolbook failed verification"
    
    # Strategy: greedy column removal
    # For each column, try to remove it and see if the remaining columns
    # can be adjusted (via least squares on the tensor) to still represent it.
    
    # More efficient: for each column c, compute the residual if we zero out column c.
    # Then check if this residual can be expressed as a sum of rank-1 tensors
    # using only the other columns. This is equivalent to checking if the residual
    # tensor lies in the span of {U[:,c'] V[:,c']^T W[:,c']^T : c' != c}.
    
    # A simpler practical approach: iteratively try to remove columns.
    # For each column c, remove it, then project the tensor onto the subspace
    # spanned by the remaining rank-1 tensors and check the residual norm.
    
    # Even simpler: just try removing each column and verify.
    # This works if the decomposition is over-complete.
    
    # Let's try a smarter approach: use the structure.
    # The tensor has the property that for each j, the "slice" T[:, j*p:(j+1)*p, :]
    # has a special form.
    
    # Actually, let's try: for each pair of columns that share the same j-index,
    # see if we can merge them.
    
    # Column c corresponds to (i_c, j_c, k_c) where:
    # c = i_c * m + j_c * p + k_c (in the schoolbook ordering)
    
    def col_to_ijk(c):
        """Convert column index to (i, j, k) in schoolbook ordering."""
        # Schoolbook ordering: for i in range(n): for j in range(m): for k in range(p)
        # So c = i * m * p + j * p + k
        i = c // (m * p)
        rem = c % (m * p)
        j = rem // p
        k = rem % p
        return i, j, k
    
    # Try removing columns greedily
    current_U = U.copy()
    current_V = V.copy()
    current_W = W.copy()
    current_rank = best_rank
    
    # Build a list of active columns
    active_cols = list(range(current_rank))
    
    # Try to remove columns one by one
    improved = True
    while improved and (time.time() - start_time) < DEADLINE - 20:
        improved = False
        for c in list(active_cols):
            if len(active_cols) <= 1:
                break
            # Try removing column c
            new_active = [x for x in active_cols if x != c]
            if not new_active:
                break
            
            new_U = current_U[:, new_active]
            new_V = current_V[:, new_active]
            new_W = current_W[:, new_active]
            
            if verify_fast(new_U, new_V, new_W, tol=1e-8):
                # Successfully removed column c
                current_U = new_U
                current_V = new_V
                current_W = new_W
                active_cols = new_active
                current_rank = len(active_cols)
                improved = True
                if current_rank < best_rank:
                    best_rank = current_rank
                    best_U = current_U.copy()
                    best_V = current_V.copy()
                    best_W = current_W.copy()
                break  # Restart the loop since columns changed
    
    # Now try a more aggressive approach: for each column, try to absorb it
    # into others by solving a linear system.
    # For column c, we want to find coefficients alpha_{c'} for c' != c such that:
    # sum_{c' != c} alpha_{c'} * U[:,c'] V[:,c']^T W[:,c']^T = U[:,c] V[:,c]^T W[:,c]^T
    # This is a linear system in the alpha_{c'} variables.
    
    # The tensor U[:,c] V[:,c]^T W[:,c]^T is a rank-1 tensor.
    # We're asking if it's in the span of the other rank-1 tensors.
    
    # Flatten the tensor and solve least squares
    target_flat = (current_U[:, c] * current_V[:, c] * current_W[:, c]).ravel() if False else None
    
    # Actually, let's vectorize the check properly
    def can_remove_column(U_mat, V_mat, W_mat, c, tol=1e-8):
        """Check if column c can be removed by adjusting others."""
        r = U_mat.shape[1]
        if r <= 1:
            return False
        
        # The tensor to represent: T_c = U[:,c] V[:,c]^T W[:,c]^T
        T_c = np.einsum('a,b,d->abd', U_mat[:, c], V_mat[:, c], W_mat[:, c])
        
        # The other columns span a subspace of tensors
        # We need to check if T_c is in span{U[:,c'] V[:,c']^T W[:,c']^T : c' != c}
        
        # Build the matrix: for each c' != c, the flattened tensor
        other_cols = [x for x in range(r) if x != c]
        if not other_cols:
            return False
        
        # Flatten each rank-1 tensor
        basis = np.array([
            np.einsum('a,b,d->abd', U_mat[:, c], V_mat[:, c], W_mat[:, c]).ravel()
            for c in other_cols
        ]).T  # shape: (nm*mp*pn, len(other_cols))
        
        target = T_c.ravel()
        
        # Solve least squares
        try:
            coeffs, residuals, rank_b, sv = np.linalg.lstsq(basis, target, rcond=None)
            recon = basis @ coeffs
            if np.linalg.norm(recon - target) < tol * np.linalg.norm(target) + 1e-10:
                return True
        except Exception:
            pass
        
        return False
    
    # Try to remove columns using the linear algebra approach
    improved = True
    while improved and (time.time() - start_time) < DEADLINE - 15:
        improved = False
        for c in list(active_cols):
            if len(active_cols) <= 1:
                break
            if can_remove_column(current_U, current_V, current_W, c):
                # Remove column c
                new_active = [x for x in active_cols if x != c]
                current_U = current_U[:, new_active]
                current_V = current_V[:, new_active]
                current_W = current_W[:, new_active]
                active_cols = new_active
                current_rank = len(active_cols)
                improved = True
                if current_rank < best_rank:
                    best_rank = current_rank
                    best_U = current_U.copy()
                    best_V = current_V.copy()
                    best_W = current_W.copy()
                break
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - 5:
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
