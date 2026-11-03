"""Matrix multiplication 3x3x5 with rank-optimized decomposition."""

DIMENSIONS = [3, 3, 5]  # (n, m, p): A is n x m, B is m x p, C is n x p

import time
import numpy as np
from fractions import Fraction

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T with shape (n*m, m*p, p*n).
    
    T[i,j,k] = 1 if the indices correspond to A[i%m, i//m] * B[j%m, j//m] = C[k%m, k//m]
    where i = a*m+b, j = b*p+c, k = c*n+a
    """
    I, J, K = n * m, m * p, p * n
    T = np.zeros((I, J, K))
    for a in range(n):
        for b in range(m):
            for c in range(p):
                i = a * m + b
                j = b * p + c
                k = c * n + a
                T[i, j, k] = 1
    return T

def _verify_decomposition(U, V, W, T, atol=1e-9):
    """Verify that U, V, W form a valid decomposition of tensor T."""
    rank = U.shape[1]
    recon = np.zeros(T.shape)
    for c in range(rank):
        recon += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
    return np.allclose(recon, T, atol=atol)

def _to_half_integer_matrix(M, tol=1e-8):
    """Convert a float matrix to half-integer values."""
    rows, cols = M.shape
    result = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            v = M[i, j]
            # Round to nearest half-integer
            rounded = round(2 * v) / 2.0
            if abs(v - rounded) < tol:
                result[i, j] = rounded
            else:
                # Try integer
                rounded_int = round(v)
                if abs(v - rounded_int) < tol:
                    result[i, j] = float(rounded_int)
                else:
                    result[i, j] = rounded  # Best effort
    return result

def _schoolbook_decomposition(n, m, p):
    """Return the schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    I, J, K = n * m, m * p, p * n
    U = np.zeros((I, rank))
    V = np.zeros((J, rank))
    W = np.zeros((K, rank))
    
    col = 0
    for a in range(n):
        for b in range(m):
            for c in range(p):
                i = a * m + b
                j = b * p + c
                k = c * n + a
                U[i, col] = 1
                V[j, col] = 1
                W[k, col] = 1
                col += 1
    
    return U, V, W

def _try_reduce_rank(U, V, W, T, max_rank, time_limit=30.0):
    """Try to reduce the rank of a decomposition by merging or eliminating columns."""
    start_time = time.time()
    I, J, K = T.shape
    current_rank = U.shape[1]
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = current_rank
    
    # Try to find linear dependencies or mergeable columns
    # Strategy: For each pair of columns, check if one can be expressed
    # in terms of the others with adjusted coefficients
    
    for target_rank in range(max_rank, best_rank - 1, -1):
        if time.time() - start_time > time_limit:
            break
        
        # Try random column elimination
        for attempt in range(50):
            if time.time() - start_time > time_limit:
                break
            
            # Try to eliminate one column at a time
            if best_rank <= target_rank:
                break
            
            # Randomly select a column to try eliminating
            col_to_remove = np.random.randint(0, best_rank)
            
            # Try to find coefficients to express this column as combination of others
            # This is a simplified approach - just try setting it to zero and checking
            U_test = best_U.copy()
            V_test = best_V.copy()
            W_test = best_W.copy()
            
            # Zero out the column
            U_test[:, col_to_remove] = 0
            V_test[:, col_to_remove] = 0
            W_test[:, col_to_remove] = 0
            
            # Check if still valid
            if _verify_decomposition(U_test, V_test, W_test, T, atol=1e-6):
                # Remove the column
                mask = np.ones(best_rank, dtype=bool)
                mask[col_to_remove] = False
                best_U = U_test[:, mask]
                best_V = V_test[:, mask]
                best_W = W_test[:, mask]
                best_rank = best_U.shape[1]
    
    return best_U, best_V, best_W, best_rank

def _structured_decomposition_3x3x5(n, m, p):
    """Construct a structured decomposition for 3x3x5 using block structure."""
    I, J, K = n * m, m * p, p * n
    T = _build_tensor(n, m, p)
    
    # Start with schoolbook
    U, V, W = _schoolbook_decomposition(n, m, p)
    
    # Try to find a lower rank decomposition using the structure
    # The 3x3x5 tensor can be viewed as 5 slices of 3x3 tensors
    
    # For now, return schoolbook as baseline
    return U, V, W

def construct():
    """
    Return a rank-minimal exact tensor decomposition for multiplying 
    a 3 by 3 matrix by a 3 by 5 matrix.
    
    Output factor follows the official trace convention W[k*n+i, r].
    
    Interface: construct() -> {'U', 'V', 'W'}: factor matrices of shapes 
    (n*m, rank), (m*p, rank), (p*n, rank) with integer/half-integer entries
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 3, 3, 5
    I, J, K = n * m, m * p, p * n  # 9, 15, 15
    T = _build_tensor(n, m, p)
    
    # Start with schoolbook decomposition (rank 45)
    U, V, W = _schoolbook_decomposition(n, m, p)
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = U.shape[1]
    
    # Verify schoolbook is correct
    assert _verify_decomposition(best_U, best_V, best_W, T, atol=1e-9), "Schoolbook verification failed"
    
    # Try to reduce rank
    if time.time() < deadline:
        # Attempt rank reduction
        U_red, V_red, W_red, red_rank = _try_reduce_rank(
            best_U, best_V, best_W, T, 
            max_rank=best_rank, 
            time_limit=min(60.0, deadline - time.time())
        )
        
        if red_rank < best_rank:
            # Verify the reduced decomposition
            if _verify_decomposition(U_red, V_red, W_red, T, atol=1e-8):
                best_U, best_V, best_W = U_red, V_red, W_red
                best_rank = red_rank
    
    # Convert to half-integer values
    U_final = _to_half_integer_matrix(best_U)
    V_final = _to_half_integer_matrix(best_V)
    W_final = _to_half_integer_matrix(best_W)
    
    # Final verification with half-integer values
    if not _verify_decomposition(U_final, V_final, W_final, T, atol=1e-8):
        # If half-integer conversion broke it, use original
        U_final = best_U
        V_final = best_V
        W_final = best_W
    
    # Convert to lists for the required output format
    U_list = U_final.tolist()
    V_list = V_final.tolist()
    W_list = W_final.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
