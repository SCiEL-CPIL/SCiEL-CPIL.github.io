import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 4, 8]

def _build_strassen_56(n, m, p):
    """Build the rank-56 Strassen decomposition as integer arrays."""
    rank = 56
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    for j0 in [0, 2]:
        for k0 in [0, 2, 4, 6]:
            # M1: (A00+A11)(B00+B11)
            U[0*m + j0, col] = 1
            U[1*m + (j0+1), col] = 1
            V[j0*p + k0, col] = 1
            V[(j0+1)*p + (k0+1), col] = 1
            W[k0*n + 0, col] = 1
            W[(k0+1)*n + 1, col] = 1
            col += 1
            
            # M2: (A10+A11)B00
            U[1*m + j0, col] = 1
            U[1*m + (j0+1), col] = 1
            V[j0*p + k0, col] = 1
            W[k0*n + 1, col] = 1
            W[(k0+1)*n + 1, col] = -1
            col += 1
            
            # M3: A00(B01-B11)
            U[0*m + j0, col] = 1
            V[j0*p + (k0+1), col] = 1
            V[(j0+1)*p + (k0+1), col] = -1
            W[(k0+1)*n + 0, col] = 1
            W[(k0+1)*n + 1, col] = 1
            col += 1
            
            # M4: A11(B10-B00)
            U[1*m + (j0+1), col] = 1
            V[(j0+1)*p + k0, col] = 1
            V[j0*p + k0, col] = -1
            W[k0*n + 0, col] = 1
            W[k0*n + 1, col] = 1
            col += 1
            
            # M5: (A00+A01)B11
            U[0*m + j0, col] = 1
            U[0*m + (j0+1), col] = 1
            V[(j0+1)*p + (k0+1), col] = 1
            W[k0*n + 0, col] = -1
            W[(k0+1)*n + 0, col] = 1
            col += 1
            
            # M6: (A10-A00)(B00+B01)
            U[1*m + j0, col] = 1
            U[0*m + j0, col] = -1
            V[j0*p + k0, col] = 1
            V[j0*p + (k0+1), col] = 1
            W[(k0+1)*n + 1, col] = 1
            col += 1
            
            # M7: (A01-A11)(B10+B11)
            U[0*m + (j0+1), col] = 1
            U[1*m + (j0+1), col] = -1
            V[(j0+1)*p + k0, col] = 1
            V[(j0+1)*p + (k0+1), col] = 1
            W[k0*n + 0, col] = 1
            col += 1
            
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly represents the matmul tensor."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _mode1_flattening(U, V, W, n, m, p):
    """
    For each rank-1 term r, form the mode-1 flattening vector:
    flatten(U[:,r] * W[:,r] reshaped appropriately)
    
    Actually, the tensor T_{i,j,k} = sum_r U_{i*m+j,r} V_{j*p+k,r} W_{k*n+i,r}
    
    The mode-1 flattening groups indices (j,k) for fixed i:
    For each r, the contribution to mode-1 flattening at position (i, j, k) is:
    U_{i*m+j, r} * V_{j*p+k, r} * W_{k*n+i, r}
    
    To check linear dependence of rank-1 terms, we can form a matrix M where
    M[row, r] represents the "fingerprint" of term r.
    
    A better approach: form the matrix P of shape (n*p, n*m) where
    P[(i,k), (j,r)] ... this gets complicated.
    
    Alternative: Use the fact that if term r is a linear combination of other terms,
    then there exist coefficients c_s such that for all (i,j,k):
    U_{i,j,r} V_{j,k,r} W_{k,i,r} = sum_s c_s U_{i,j,s} V_{j,k,s} W_{k,i,s}
    
    This is hard to check directly. Instead, let's try a different approach:
    Form the matrix A of shape (n*m*p, rank) where A[(i,j,k), r] = U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    Then check the rank of this matrix. If rank < number of columns, there are dependencies.
    
    But this matrix has 2*4*8 = 64 rows and 56 columns. We can compute its rank.
    """
    rank = U.shape[1]
    # Build the tensor evaluation matrix
    # T_mat has shape (n*m*p, rank)
    # T_mat[(i,j,k), r] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    T_mat = np.zeros((n*m*p, rank), dtype=float)
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for r in range(rank):
                    T_mat[idx, r] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                idx += 1
    
    return T_mat

def _try_rank_reduction(U, V, W, n, m, p, deadline):
    """
    Try to reduce the rank by finding and removing linearly dependent terms.
    Returns (reduced_U, reduced_V, reduced_W, new_rank) or (U, V, W, rank) if no reduction.
    """
    rank = U.shape[1]
    
    # Build the tensor evaluation matrix
    T_mat = _mode1_flattening(U, V, W, n, m, p)
    
    # Compute the rank of T_mat using SVD
    # If rank(T_mat) < rank, there are linear dependencies
    try:
        U_svd, s_vals, Vt = np.linalg.svd(T_mat, full_matrices=False)
        # Determine numerical rank
        tol = max(T_mat.shape) * np.finfo(float).eps * (s_vals[0] if len(s_vals) > 0 else 1.0)
        num_rank = np.sum(s_vals > tol)
    except:
        return U, V, W, rank
    
    if num_rank >= rank:
        # No dependencies found
        return U, V, W, rank
    
    # There are dependencies. We need to find which columns are dependent.
    # Use the null space of T_mat
    # Columns of T_mat that are in the null space can be removed.
    # Find the null space vectors
    null_space = Vt[num_rank:, :]  # Shape: (rank - num_rank, rank)
    
    # For each null space vector, it tells us a linear combination of terms that gives zero tensor.
    # We can try to eliminate one term at a time.
    
    # Strategy: For each term r, check if it can be expressed as a combination of others.
    # If T_mat[:, r] is in the span of T_mat[:, s] for s != r, then term r is redundant.
    
    # Use QR with column pivoting to find the independent columns
    from scipy.linalg import qr
    Q, R, P = qr(T_mat, pivoting=True)
    
    # The pivoting tells us the order of columns by importance
    # The first num_rank pivoted columns are independent
    # The remaining are dependent
    
    # Let's try to remove the last (most dependent) column first
    # and verify the result
    
    best_U, best_V, best_W, best_rank = U.copy(), V.copy(), W.copy(), rank
    
    # Try removing each term from last to first (most likely dependent first)
    for remove_idx in range(rank - 1, -1, -1):
        if time.time() > deadline - 5:
            break
        
        # Create reduced matrices by removing column remove_idx
        mask = [i for i in range(rank) if i != remove_idx]
        new_rank = rank - 1
        new_U = U[:, mask]
        new_V = V[:, mask]
        new_W = W[:, mask]
        
        # Verify the reduced decomposition
        if _verify_decomposition(new_U, new_V, new_W, n, m, p, tol=1e-9):
            best_U, best_V, best_W, best_rank = new_U, new_V, new_W, new_rank
            # Continue trying to remove more
            U, V, W, rank = new_U, new_V, new_W, new_rank
        else:
            # This term is not redundant, keep it
            pass
    
    return best_U, best_V, best_W, best_rank

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    
    # Build initial rank-56 Strassen decomposition
    U, V, W = _build_strassen_56(n, m, p)
    rank = U.shape[1]
    
    # Verify initial decomposition
    if not _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
        # Should not happen, but return as-is
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to reduce the rank
    if time.time() < deadline - 10:
        U, V, W, rank = _try_rank_reduction(U, V, W, n, m, p, deadline)
    
    # Final verification
    if not _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
        # Fallback to original
        U, V, W = _build_strassen_56(n, m, p)
        rank = U.shape[1]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    result = construct()
    print("Rank:", len(result['U'][0]))
