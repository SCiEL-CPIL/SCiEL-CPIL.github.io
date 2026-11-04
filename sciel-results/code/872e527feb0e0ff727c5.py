"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x4 matrix.
    
    Strategy:
    1. Start with the schoolbook decomposition (rank = 3*3*4 = 36).
    2. Use an iterative column-pruning heuristic with linear algebra:
       - The multiplication tensor T has shape (n*m, m*p, p*n) = (9, 12, 12).
       - A rank-r decomposition means T = sum_{c=1}^{r} u_c * v_c * w_c.
       - We can verify exactness by checking that the reconstruction matches T exactly.
       
    Approach:
    - Build the schoolbook decomposition with 36 rank-1 terms.
    - Try to reduce the rank by finding linear dependencies among the rank-1 tensors.
    - For each subset of columns, check if the tensor T can be expressed as a linear
      combination of the remaining rank-1 tensors.
    - Use numpy's linear algebra to solve for the coefficients.
    
    For 3x3 * 3x4, we'll try to find a decomposition with rank < 36.
    """
    n, m, p = DIMENSIONS  # 3, 3, 4
    
    # Build the multiplication tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
    # All other entries are 0.
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Schoolbook decomposition
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    # Try to reduce rank by checking if we can find a smaller set of rank-1 tensors
    # that span the same tensor.
    
    # The key insight: if we have a set of rank-1 tensors {u_c * v_c * w_c}, we can check
    # if T is in their span by solving a linear system.
    
    # Let's try a greedy pruning approach:
    # Start with all 36 columns, and try to remove one column at a time.
    # After removing a column, check if T is still in the span of the remaining columns.
    
    # To check if T is in the span of a set of rank-1 tensors:
    # Flatten T to a vector of length (n*m * m*p * p*n).
    # Each rank-1 tensor u_c * v_c * w_c also flattens to a vector.
    # Check if T is in the column space of the matrix formed by these vectors.
    
    def is_in_span(T_flat, U_cols, V_cols, W_cols):
        """Check if T is in the span of the given rank-1 tensors."""
        # Build the matrix of rank-1 tensors (flattened)
        num_cols = U_cols.shape[1]
        if num_cols == 0:
            return np.allclose(T_flat, 0)
        
        # Each column c gives a rank-1 tensor u_c * v_c * w_c
        # Flatten each to get a vector of length n*m*m*p*p*n
        flat_dim = T_flat.shape[0]
        M = np.zeros((flat_dim, num_cols))
        for c in range(num_cols):
            # Outer product of three vectors
            rank1 = np.einsum('i,j,k->ijk', U_cols[:, c], V_cols[:, c], W_cols[:, c])
            M[:, c] = rank1.flatten()
        
        # Check if T_flat is in the column space of M
        # Use least squares and check the residual
        try:
            coeffs, residuals, rank_M, s = np.linalg.lstsq(M, T_flat, rcond=None)
            reconstructed = M @ coeffs
            return np.allclose(reconstructed, T_flat, atol=1e-10)
        except np.linalg.LinAlgError:
            return False
    
    # Flatten T
    T_flat = T.flatten()
    
    # Try to prune columns greedily
    current_U = U.copy()
    current_V = V.copy()
    current_W = W.copy()
    current_rank = rank
    
    # Try removing each column one by one
    for attempt in range(100):  # Limit attempts to avoid timeout
        removed = False
        for c in range(current_rank - 1, -1, -1):
            # Try removing column c
            new_U = np.delete(current_U, c, axis=1)
            new_V = np.delete(current_V, c, axis=1)
            new_W = np.delete(current_W, c, axis=1)
            
            if is_in_span(T_flat, new_U, new_V, new_W):
                current_U = new_U
                current_V = new_V
                current_W = new_W
                current_rank -= 1
                removed = True
                break
        
        if not removed:
            break
    
    # Convert to lists
    U_list = current_U.tolist()
    V_list = current_V.tolist()
    W_list = current_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
