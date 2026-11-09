"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = 2, 5, 6
    deadline = time.time() + 130  # safety margin before 150s limit
    
    # Schoolbook rank = n*m*p = 60
    # Tensor T has shape (n*m, m*p, p*n) = (10, 30, 12)
    # T[i*m+j, j'*p+k, k'*n+i'] = 1 if i==i', j==j', k==k' else 0
    
    # Build the tensor
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Start with schoolbook decomposition (rank 60)
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1.0
                V[j*p + k, col] = 1.0
                W[k*n + i, col] = 1.0
                col += 1
    
    # Try to reduce rank by finding linear dependencies
    # Strategy: for each column, check if it can be expressed as a linear combination
    # of other columns. If so, remove it.
    
    def verify(U_mat, V_mat, W_mat, T_tensor):
        """Verify that the decomposition exactly reconstructs T."""
        # T_recon[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # Check a sample of entries
        n_rows = U_mat.shape[0]
        n_cols = V_mat.shape[0]
        n_out = W_mat.shape[0]
        r = U_mat.shape[1]
        
        # Check all entries (small enough)
        for a in range(n_rows):
            for b in range(n_cols):
                for c in range(n_out):
                    recon = 0.0
                    for rr in range(r):
                        recon += U_mat[a, rr] * V_mat[b, rr] * W_mat[c, rr]
                    if abs(recon - T_tensor[a, b, c]) > 1e-9:
                        return False
        return True
    
    # Try to reduce rank
    current_U = U.copy()
    current_V = V.copy()
    current_W = W.copy()
    current_rank = rank
    
    # Attempt rank reduction via linear algebra
    # For each column r, check if removing it still allows reconstruction
    # This is expensive, so we do it carefully
    
    # Better approach: use the fact that for matrix multiplication tensors,
    # there are known lower bounds and some known decompositions.
    
    # For 2x5x6, let's try a structured decomposition.
    # Key insight: n=2 is small. We can write C[i,k] = sum_j A[i,j] B[j,k]
    # for i=0,1 and k=0..5.
    
    # Let's try to find a rank-r decomposition by solving a system of equations.
    # For rank r, we need U (10 x r), V (30 x r), W (12 x r) such that
    # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    
    # This is a 3-tensor decomposition problem. Let's use a greedy approach:
    # Start with schoolbook, then try to merge columns.
    
    # For each pair of columns (r1, r2), check if they can be merged
    # (i.e., if there exists a linear combination that preserves the tensor).
    
    # Actually, a simpler approach: try to find if any column is redundant.
    # Column r is redundant if there exist scalars alpha_s (s != r) such that
    # U[:,r] = sum_{s!=r} alpha_s U[:,s]
    # V[:,r] = sum_{s!=r} alpha_s V[:,s]  
    # W[:,r] = sum_{s!=r} alpha_s W[:,s]
    
    # This is a strong condition. Let's check it.
    
    def try_remove_column(U_mat, V_mat, W_mat, col_idx, T_tensor):
        """Try to remove column col_idx by expressing it as a linear combination of others."""
        r = U_mat.shape[1]
        if r <= 1:
            return None, None, None, None
        
        # Get all other columns
        other_cols = [c for c in range(r) if c != col_idx]
        
        # Solve for alpha such that U[:,col_idx] = U[:,other_cols] @ alpha
        U_other = U_mat[:, other_cols]
        U_target = U_mat[:, col_idx]
        
        # Check if U_target is in the span of U_other
        # Use least squares
        try:
            alpha, residuals, rank_u, sv = np.linalg.lstsq(U_other, U_target, rcond=None)
        except:
            return None, None, None, None
        
        # Check if the reconstruction is good
        U_recon = U_other @ alpha
        if not np.allclose(U_recon, U_target, atol=1e-9):
            return None, None, None, None
        
        # Check V
        V_other = V_mat[:, other_cols]
        V_target = V_mat[:, col_idx]
        V_recon = V_other @ alpha
        if not np.allclose(V_recon, V_target, atol=1e-9):
            return None, None, None, None
        
        # Check W
        W_other = W_mat[:, other_cols]
        W_target = W_mat[:, col_idx]
        W_recon = W_other @ alpha
        if not np.allclose(W_recon, W_target, atol=1e-9):
            return None, None, None, None
        
        # All three match! Remove the column.
        new_U = np.delete(U_mat, col_idx, axis=1)
        new_V = np.delete(V_mat, col_idx, axis=1)
        new_W = np.delete(W_mat, col_idx, axis=1)
        return new_U, new_V, new_W, r - 1
    
    # Try to remove columns iteratively
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        for col_idx in range(current_U.shape[1]):
            if time.time() > deadline - 10:
                break
            new_U, new_V, new_W, new_rank = try_remove_column(
                current_U, current_V, current_W, col_idx, T
            )
            if new_U is not None:
                # Verify the new decomposition
                if verify(new_U, new_V, new_W, T):
                    current_U = new_U
                    current_V = new_V
                    current_W = new_W
                    current_rank = new_rank
                    improved = True
                    break  # Restart from beginning after a removal
    
    # Try a more aggressive approach: random linear combinations
    # Pick two random columns and try to see if a linear combination of them
    # can replace one of them.
    
    def try_merge_columns(U_mat, V_mat, W_mat, col1, col2, T_tensor):
        """Try to replace col1 with a linear combination involving col2."""
        r = U_mat.shape[1]
        if r <= 1 or col1 == col2:
            return None, None, None, None
        
        # Try alpha * col1 + beta * col2 -> new col
        # We want to eliminate one column.
        # If we can find alpha, beta such that the new column is redundant with others...
        # This is complex. Let's skip for now.
        return None, None, None, None
    
    # Final verification
    if not verify(current_U, current_V, current_W, T):
        # Fall back to schoolbook
        current_U = U.copy()
        current_V = V.copy()
        current_W = W.copy()
        current_rank = rank
    
    # Convert to lists
    return {
        "U": current_U.tolist(),
        "V": current_V.tolist(),
        "W": current_W.tolist()
    }
# EVOLVE-BLOCK-END
