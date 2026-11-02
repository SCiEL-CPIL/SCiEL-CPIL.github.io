"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START

def _build_schoolbook(n, m, p):
    """Return (U, V, W) as lists of lists, schoolbook rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check tensor decomposition: T[i,j,k,r] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    For matrix mult tensor, T[i,j,k,r] should equal 1 iff r==k*n+i and the index (i,j) maps to the right.
    Actually the standard tensor: T[i,j,k] = sum_r A[i,j] B[j,k] contributes to C[i,k].
    The decomposition says: C[i,k] = sum_c (sum_j U[i*m+j,c] A[i,j]) (sum_j V[j*p+k,c] B[j,k]) * W[k*n+i,c]
    For the standard identity, we need W[k*n+i,c] to be 1 only for the term corresponding to (i,j,k).
    Let's verify by checking the tensor directly."""
    rank = len(U[0])
    # Build tensor T from U,V,W: T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    # The correct tensor for matmul 3x4 * 4x6 is: T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
    # Actually the multiplication tensor has entries: T[i,j,k] = 1 for all valid i,j,k (it's the identity tensor)
    # No wait. The matrix multiplication tensor T_{n,m,p} has entries indexed by (i,j,k) where
    # T[i,j,k] = 1 if the (i,j) entry of first matrix and (j,k) entry of second matrix contribute to (i,k) of product.
    # The tensor entry is 1 for all valid indices (i,j,k). The decomposition reconstructs it.
    # Actually, the standard definition: the multiplication tensor has T[i,j,k] = 1 for all i,j,k in range.
    # Let me re-check: C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor T has T[i,j,k] = 1 (it's all ones for valid indices).
    # The decomposition: T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # This must equal 1 for all i,j,k.
    
    # For schoolbook, each (i,j,k) has its own column c with U=V=W=1 at that entry, so sum = 1. Good.
    # Let's verify numerically.
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_arr[i*m+j], V_arr[j*p+k]) * W_arr[k*n+i]
                # Actually it's sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
                # That's not np.dot(U, V) * W. It's sum over c of U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
                val = np.sum(U_arr[i*m+j] * V_arr[j*p+k] * W_arr[k*n+i])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _peel_rank(U, V, W, n, m, p, deadline):
    """Attempt to reduce rank by finding linearly dependent columns and removing them.
    This is a heuristic: if we can find a column that is a linear combination of others
    in a way that preserves the tensor, we can remove it. This is complex, so we try a simpler approach:
    try to find a lower rank by random linear combinations and checking if the tensor is preserved."""
    
    rank = len(U[0])
    best_rank = rank
    best_U, best_V, best_W = [row[:] for row in U], [row[:] for row in V], [row[:] for row in W]
    
    # Convert to numpy for operations
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Try to peel columns: for each column, try to see if it can be expressed as a combination of others
    # This is expensive, so we limit iterations based on deadline.
    time_start = time.time()
    
    current_rank = rank
    U_cur = U_arr.copy()
    V_cur = V_arr.copy()
    W_cur = W_arr.copy()
    
    # We'll try to remove columns one by one, checking if the tensor is still valid.
    # A column c can be removed if there exist coefficients alpha such that for all (i,j,k):
    # sum_{c' != c} U[i*m+j,c'] * V[j*p+k,c'] * W[k*n+i,c'] * alpha[c'] = U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # This is a system of linear equations in alpha. If it has a solution, we can remove column c.
    
    removed = []
    for attempt in range(current_rank):
        if time.time() > deadline - 10:  # safety margin
            break
        # Try to remove the last remaining column
        c = current_rank - 1
        # Build the tensor contribution of column c
        # We need to find alpha for all other columns such that the tensor contribution matches
        # This is a large system, so let's use a simpler heuristic: try random linear combinations
        
        # Alternative approach: try to see if the column is redundant by checking if the tensor
        # without this column can be fixed by adjusting other columns.
        # For now, let's just try a simple peeling: if two columns are identical, remove one.
        
        # Check if column c is a scalar multiple of another column
        found_duplicate = False
        for c2 in range(c):
            if c2 in removed:
                continue
            # Check if U[:,c], V[:,c], W[:,c] are proportional to U[:,c2], V[:,c2], W[:,c2]
            U_c = U_cur[:, c]
            U_c2 = U_cur[:, c2]
            V_c = V_cur[:, c]
            V_c2 = V_cur[:, c2]
            W_c = W_cur[:, c]
            W_c2 = W_cur[:, c2]
            
            # Find scalar s such that U_c = s * U_c2 (if possible)
            nonzero_mask = np.abs(U_c2) > 1e-10
            if np.sum(nonzero_mask) > 0:
                s = np.mean(U_c[nonzero_mask] / U_c2[nonzero_mask])
                if np.allclose(U_c, s * U_c2, atol=1e-10) and \
                   np.allclose(V_c, s * V_c2, atol=1e-10) and \
                   np.allclose(W_c, s * W_c2, atol=1e-10):
                    # Column c is redundant, remove it
                    removed.append(c)
                    current_rank -= 1
                    found_duplicate = True
                    break
        
        if not found_duplicate:
            # Try a more general linear dependence check
            # This is computationally expensive, so we skip for now
            pass
    
    if removed:
        # Reconstruct without removed columns
        keep = [c for c in range(rank) if c not in removed]
        best_U = U_arr[:, keep].tolist()
        best_V = V_arr[:, keep].tolist()
        best_W = W_arr[:, keep].tolist()
        best_rank = len(keep)
    
    return best_U, best_V, best_W, best_rank

# EVOLVE-BLOCK-END

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    U, V, W = _build_schoolbook(n, m, p)
    
    # Verify the schoolbook decomposition
    if not _verify(U, V, W, n, m, p):
        raise ValueError("Schoolbook decomposition failed verification")
    
    # Try to reduce rank via peeling
    U_red, V_red, W_red, red_rank = _peel_rank(U, V, W, n, m, p, deadline)
    
    # Verify the reduced decomposition
    if _verify(U_red, V_red, W_red, n, m, p):
        return {"U": U_red, "V": V_red, "W": W_red}
    else:
        # Fall back to schoolbook
        return {"U": U, "V": V, "W": W}
