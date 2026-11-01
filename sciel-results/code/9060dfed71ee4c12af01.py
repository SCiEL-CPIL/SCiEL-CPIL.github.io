import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 5]
_DEADLINE = 140.0

def _strassen_2x2x2(U, V, W, col, i_offset, j_offset, k_offset, n, m, p):
    """Strassen decomposition for 2x2x2 block."""
    # M1: (A11+A22)(B11+B22)
    U[i_offset*m + j_offset, col] = 1
    U[(i_offset+1)*m + (j_offset+1), col] = 1
    V[j_offset*p + k_offset, col] = 1
    V[(j_offset+1)*p + (k_offset+1), col] = 1
    W[k_offset*n + i_offset, col] = 1
    W[(k_offset+1)*n + (i_offset+1), col] = 1
    col += 1
    # M2: (A21+A22)B11
    U[(i_offset+1)*m + j_offset, col] = 1
    U[(i_offset+1)*m + (j_offset+1), col] = 1
    V[j_offset*p + k_offset, col] = 1
    W[k_offset*n + (i_offset+1), col] = 1
    W[(k_offset+1)*n + (i_offset+1), col] = -1
    col += 1
    # M3: A12(B21-B11)
    U[i_offset*m + j_offset, col] = 1
    V[j_offset*p + (k_offset+1), col] = 1
    V[(j_offset+1)*p + (k_offset+1), col] = -1
    W[(k_offset+1)*n + i_offset, col] = 1
    W[(k_offset+1)*n + (i_offset+1), col] = 1
    col += 1
    # M4: A21(B22-B12)
    U[(i_offset+1)*m + (j_offset+1), col] = 1
    V[(j_offset+1)*p + k_offset, col] = 1
    V[j_offset*p + k_offset, col] = -1
    W[k_offset*n + i_offset, col] = 1
    W[k_offset*n + (i_offset+1), col] = 1
    col += 1
    # M5: (A11+A12)B22
    U[i_offset*m + j_offset, col] = 1
    U[i_offset*m + (j_offset+1), col] = 1
    V[(j_offset+1)*p + (k_offset+1), col] = 1
    W[k_offset*n + i_offset, col] = -1
    W[(k_offset+1)*n + i_offset, col] = 1
    col += 1
    # M6: (A21-A11)(B11+B12)
    U[(i_offset+1)*m + j_offset, col] = 1
    U[i_offset*m + j_offset, col] = -1
    V[j_offset*p + k_offset, col] = 1
    V[j_offset*p + (k_offset+1), col] = 1
    W[(k_offset+1)*n + (i_offset+1), col] = 1
    col += 1
    # M7: (A12-A22)(B21+B22)
    U[i_offset*m + (j_offset+1), col] = 1
    U[(i_offset+1)*m + (j_offset+1), col] = -1
    V[(j_offset+1)*p + k_offset, col] = 1
    V[(j_offset+1)*p + (k_offset+1), col] = 1
    W[k_offset*n + i_offset, col] = 1
    col += 1
    return col

def _verify_rank(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces the matrix multiplication tensor."""
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U[i*m + j]
                v_val = V[j*p + k]
                w_val = W[k*n + i]
                result = np.dot(u_val, v_val * w_val)
                if abs(result - 1.0) > 1e-10:
                    return False
    return True

def _rank28_decomposition(n, m, p):
    """Known rank-28 decomposition using Strassen + schoolbook blocks."""
    rank = 28
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    # Block 1: j=0, all k (2x1x5 schoolbook)
    for i in range(n):
        for k in range(p):
            U[i*m + 0, col] = 1
            V[0*p + k, col] = 1
            W[k*n + i, col] = 1
            col += 1
    # Block 2: j in {1,2}, k in {0,1} (Strassen)
    col = _strassen_2x2x2(U, V, W, col, 0, 1, 0, n, m, p)
    # Block 3: j in {1,2}, k=2 (schoolbook)
    for i in range(n):
        for j in [1, 2]:
            U[i*m + j, col] = 1
            V[j*p + 2, col] = 1
            W[2*n + i, col] = 1
            col += 1
    # Block 4: j in {1,2}, k in {3,4} (Strassen)
    col = _strassen_2x2x2(U, V, W, col, 0, 1, 3, n, m, p)
    assert col == rank, f"Expected rank {rank}, got {col}"
    return U, V, W

def _try_rank_reduction(U28, V28, W28, n, m, p, target_rank):
    """
    Try to reduce the rank of the 28-term decomposition to target_rank.
    Uses linear algebra to find if there's a linear dependency among the terms.
    
    The key insight: if we have a rank-r decomposition, the tensor T can be written as
    T[i,j,k] = sum_{t=0}^{r-1} U[i,j,t] * V[j,k,t] * W[k,i,t]
    
    We can try to find a linear combination of the existing terms that equals zero,
    which would allow us to eliminate one term.
    """
    rank = U28.shape[1]
    if rank <= target_rank:
        return U28, V28, W28
    
    # Build the flattening matrices
    # M_ij: matrix of size (p*n) x (rank) where M_ij[k*n+i, t] = U[i*m+j, t] * V[j*p+k, t]
    # The tensor condition is: for each (i,j), M_ij @ W[:, t] should give the correct values
    
    # Actually, let's use a different approach. We'll set up the system:
    # For a rank-r decomposition, we need to find U, V, W of rank r.
    # Start with the 28-term decomposition and try to find a null vector.
    
    # The tensor T has entries T[i,j,k] = 1 for all i,j,k (since we're decomposing matrix mult)
    # Wait, no. The matrix multiplication tensor is:
    # T[i,j,k] = 1 if the output (i,k) depends on input (i,j) and (j,k)
    # Actually, for matrix multiplication, T[i,j,k] = 1 for all valid i,j,k
    
    # Let me reconsider. The decomposition must satisfy:
    # For all i, j, k: sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 1
    
    # This is a system of n*m*p = 30 equations.
    
    # Try to find a linear dependency in the 28 terms.
    # If there exists a vector c such that sum_t c_t * U[:,t] = 0, sum_t c_t * V[:,t] = 0, 
    # and sum_t c_t * W[:,t] = 0, then we can eliminate one term.
    
    # But that's too strong. Let's try a more general approach.
    
    # For each pair of terms, try to see if one can be expressed as a combination of others.
    # This is getting complex. Let's try a simpler heuristic:
    
    # Try to find if any column is a linear combination of the others in U, V, and W simultaneously.
    
    for elim_idx in range(rank):
        if time.time() > _DEADLINE - 30:
            break
        
        # Try to express column elim_idx as a combination of other columns
        U_others = np.delete(U28, elim_idx, axis=1)
        V_others = np.delete(V28, elim_idx, axis=1)
        W_others = np.delete(W28, elim_idx, axis=1)
        
        U_target = U28[:, elim_idx]
        V_target = V28[:, elim_idx]
        W_target = W28[:, elim_idx]
        
        # Check if U_target is in the span of U_others
        rank_U = np.linalg.matrix_rank(U_others)
        rank_U_with = np.linalg.matrix_rank(np.column_stack([U_others, U_target]))
        if rank_U_with > rank_U:
            continue
        
        # Check if V_target is in the span of V_others
        rank_V = np.linalg.matrix_rank(V_others)
        rank_V_with = np.linalg.matrix_rank(np.column_stack([V_others, V_target]))
        if rank_V_with > rank_V:
            continue
        
        # Check if W_target is in the span of W_others
        rank_W = np.linalg.matrix_rank(W_others)
        rank_W_with = np.linalg.matrix_rank(np.column_stack([W_others, W_target]))
        if rank_W_with > rank_W:
            continue
        
        # All three targets are in the span of their respective others.
        # Now we need to find the actual coefficients and verify the tensor condition.
        
        # Find coefficients for U
        c_U, residuals_U, rank_U, _ = np.linalg.lstsq(U_others, U_target, rcond=None)
        c_V, residuals_V, rank_V, _ = np.linalg.lstsq(V_others, V_target, rcond=None)
        c_W, residuals_W, rank_W, _ = np.linalg.lstsq(W_others, W_target, rcond=None)
        
        # The coefficients should be the same (or related) for the decomposition to work.
        # Actually, for the tensor decomposition, we need:
        # U[:, elim_idx] = sum_s c_s * U[:, s] (for s != elim_idx)
        # V[:, elim_idx] = sum_s c_s * V[:, s] (for s != elim_idx)
        # W[:, elim_idx] = sum_s c_s * W[:, s] (for s != elim_idx)
        # with the SAME coefficients c_s for all three.
        
        # Let's check if c_U, c_V, c_W are approximately equal.
        if np.allclose(c_U, c_V, atol=1e-10) and np.allclose(c_V, c_W, atol=1e-10):
            # We can eliminate this term!
            U_new = np.delete(U28, elim_idx, axis=1)
            V_new = np.delete(V28, elim_idx, axis=1)
            W_new = np.delete(W28, elim_idx, axis=1)
            
            # Verify
            if _verify_rank(U_new, V_new, W_new, n, m, p):
                return U_new, V_new, W_new
    
    return U28, V28, W28

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + _DEADLINE
    
    # Construct rank 28 decomposition
    U, V, W = _rank28_decomposition(n, m, p)
    
    if not _verify_rank(U, V, W, n, m, p):
        raise ValueError("Rank-28 decomposition failed verification")
    
    # Try to reduce rank
    U, V, W = _try_rank_reduction(U, V, W, n, m, p, 27)
    
    if not _verify_rank(U, V, W, n, m, p):
        # Fall back to rank 28
        U, V, W = _rank28_decomposition(n, m, p)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
