"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # 1. Build the Matrix Multiplication Tensor T
    # T[i, j, k] = 1 if and only if the element (i,k) of the product
    # depends on A[i,j] * B[j,k].
    # For standard matrix multiplication, T[i,j,k] = 1 for all valid i,j,k.
    T = np.zeros((n, m, p), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
                
    # Flatten T into matrices for ALS updates
    # Mode-0 flatten: T0[a, bc] where a is index for i, bc is combined j,k
    # Shape: (n, m*p)
    T0 = T.reshape(n, m*p)
    
    # Mode-1 flatten: T1[b, ac] where b is index for j, ac is combined i,k
    # Note: Need to permute axes to (j, i, k) then reshape.
    T_perm_1 = np.transpose(T, (1, 0, 2))
    T1 = T_perm_1.reshape(m, n*p)
    
    # Mode-2 flatten: T2[c, ab] where c is index for k, ab is combined i,j
    # Note: Need to permute axes to (k, i, j) then reshape.
    T_perm_2 = np.transpose(T, (2, 0, 1))
    T2 = T_perm_2.reshape(p, n*m)

    # Schoolbook rank as fallback
    schoolbook_rank = n * m * p
    best_rank = schoolbook_rank
    best_factors = None
    
    # Initialize with schoolbook decomposition if we don't find better
    def make_schoolbook():
        U = np.zeros((n*m, schoolbook_rank))
        V = np.zeros((m*p, schoolbook_rank))
        W = np.zeros((p*n, schoolbook_rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # U index: i*m + j ? 
                    # Wait, the task says U shape is (n*m, rank).
                    # Usually U[i, j] corresponds to A[i,j].
                    # Standard convention: U row index is (i, j) combined.
                    # Let's check the seed program:
                    # U[i*m + j][column] = 1
                    # V[j*p + k][column] = 1
                    # W[k*n + i][column] = 1
                    # This matches the "trace convention" W[k*n+i, r] mentioned?
                    # The prompt says: "output factor follows the official trace convention W[k*n+i,r]"
                    # And seed uses U[i*m+j], V[j*p+k], W[k*n+i].
                    # We will stick to this indexing to be safe.
                    U[i*m + j, col] = 1.0
                    V[j*p + k, col] = 1.0
                    W[k*n + i, col] = 1.0
                    col += 1
        return U, V, W
    
    U_sb, V_sb, W_sb = make_schoolbook()
    best_factors = (U_sb, V_sb, W_sb)
    
    # Try ranks from 10 upwards. 
    # Theoretical lower bound is max(n*m, m*p, p*n) = max(9, 24, 24) = 24.
    # So we only need to try ranks >= 24.
    # However, sometimes lower ranks are claimed in literature for specific cases? 
    # No, the rank is at least the geometric rank. For 3x3x8, it's likely around 21-24.
    # Let's try ranks 15 to 24 to be safe, although 24 is the trivial upper bound from dimensions.
    # Actually, the schoolbook is 72. We want to go down.
    
    max_rank_to_try = 24
    min_rank_to_try = 15
    
    np.random.seed(42)
    
    for rank in range(min_rank_to_try, max_rank_to_try + 1):
        if time.time() > deadline:
            break
            
        found = False
        num_restarts = 5
        # Reduce restarts as rank increases or if time is tight
        if time.time() > start_time + 60:
            num_restarts = 2
            
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
                
            # Initialize U, V, W with random small values
            # Scaling is important. Let's start with N(0, 1)
            U = np.random.randn(n*m, rank)
            V = np.random.randn(m*p, rank)
            W = np.random.randn(p*n, rank)
            
            # ALS iterations
            converged = False
            max_iter = 50
            tol = 1e-6
            prev_err = float('inf')
            
            for it in range(max_iter):
                # Update U
                # T0 approx U * (V^T * W) ??? 
                # Let's derive carefully.
                # T[i, j, k] = sum_r U[i, r] * V[j, r] * W[k, r]
                # Fix V, W. For a fixed i:
                # T[i, j, k] = sum_r U[i, r] * (V[j, r] * W[k, r])
                # Let M_U[r, j, k] = V[j, r] * W[k, r].
                # We want to find U[i, :] such that T[i, :, :] approx sum_r U[i, r] M_U[r, :, :]
                # Flatten T[i, :, :] to vector t_i of length m*p.
                # Flatten M_U to matrix M of size (m*p) x rank.
                # Then t_i approx U[i, :]^T * M  (if U is column vector)
                # Or t_i approx U[i, :] * M^T (if U is row vector)
                # Let's use numpy matrix multiplication.
                # M_U shape: (rank, m, p)
                # Reshape to (rank, m*p) -> M_mat
                # We want U_row (1 x rank) * M_mat.T (rank x m*p) = t (1 x m*p)
                # So U = t * M_mat / (M_mat.T * M_mat)
                
                # Build M_mat for U update
                # M_U[r, j, k] = V[j, r] * W[k, r]
                # We can compute this efficiently using broadcasting if we transpose V and W appropriately.
                # V is (m, rank). W is (p, rank).
                # V^T is (rank, m). W^T is (rank, p).
                # Element (r, j, k) is V[r, j] * W[r, k] = V[j, r] * W[k, r].
                # So M_U = V.T * W.T ? No, that's elementwise product.
                # M_U is a tensor.
                # We need M_mat of shape (m*p, rank) such that M_mat[j*p+k, r] = V[j, r] * W[k, r].
                # Let's construct it.
                # V_T = V.T (rank, m)
                # W_T = W.T (rank, p)
                # We want to multiply V_T[r, :] and W_T[r, :] for each r? No.
                # For each r, we have vector v_r (length m) and w_r (length p).
                # We need outer product v_r * w_r.T ? No, we want to flatten it to length m*p.
                # The term is U[i, r] * sum_{j,k} (V[j, r] W[k, r] ... )
                # Wait, the tensor term is U[i,r] V[j,r] W[k,r].
                # For fixed i, the vector over (j,k) is U[i,:] dot (V[:,r] * W[:,r] for each r summed?).
                # No. T[i, j, k] = sum_r U[i, r] V[j, r] W[k, r].
                # Let's fix j, k. Then T[i, j, k] is linear in U[i, :].
                # Let's fix i. We have a system for U[i, :].
                # T[i, :, :] = sum_r U[i, r] * (V[:, r] * W[:, r])_outer_flattened?
                # The term V[j, r] W[k, r] is not an outer product of V and W.
                # It is elementwise product of V[:, r] and W[:, r]? No, indices are j and k.
                # V is m x rank. W is p x rank.
                # V[:, r] is length m. W[:, r] is length p.
                # The term in the sum for fixed r is U[i, r] * (V[:, r] * W[:, r])?
                # No, V[j, r] depends on j, W[k, r] depends on k.
                # So for fixed r, the contribution to T[i, :, :] is U[i, r] * (V[:, r] * W[:, r] as a tensor product? No.
                # It is U[i, r] * (V[:, r] * W[:, r]) where the multiplication is not defined directly because dimensions differ?
                # Ah, V[j, r] W[k, r] is just the product of two scalars.
                # So for fixed r, the matrix M_r of size m x p has entries M_r[j, k] = V[j, r] W[k, r].
                # This M_r is exactly the outer product of column r of V and column r of W.
                # So M_r = V[:, r] * W[:, r].T
                # So T[i, :, :] = sum_r U[i, r] * M_r
                # Flatten T[i, :, :] to t_i (m*p).
                # Flatten M_r to m_r (m*p).
                # t_i = sum_r U[i, r] m_r
                # Let M_mat be matrix of size (m*p, rank) where column r is m_r.
                # Then t_i = M_mat.T * U[i, :]^T  (column vector)
                # Or t_i^T = U[i, :] * M_mat
                # So U[i, :] = t_i^T * (M_mat M_mat.T)^{-1} M_mat.T ?
                # Least squares: Minimize || t_i^T - U[i, :] M_mat ||
                # Solution: U[i, :] = t_i^T M_mat^T (M_mat M_mat^T)^{-1}
                
                # Construct M_mat for U:
                # M_mat[:, r] = flatten(V[:, r] * W[:, r].T)
                # V[:, r] is m x 1. W[:, r].T is 1 x p.
                # Outer product is m x p. Flatten to m*p.
                # We can do this for all r at once.
                # V is (m, rank), W is (p, rank).
                # We want a tensor of shape (m, p, rank) where [j, k, r] = V[j, r] * W[k, r].
                # This is V[:, r] * W[k, r].
                # Let's use broadcasting.
                # V_expand: (m, 1, rank) -> V.T.T? 
                # V is (m, rank). V.T is (rank, m).
                # Let's stick to explicit loops or efficient einsum.
                # M_mat has shape (m*p, rank).
                # M_mat[j*p+k, r] = V[j, r] * W[k, r].
                
                # Efficient way:
                # V_T = V.T # (rank, m)
                # W_T = W.T # (rank, p)
                # We want to compute for each r: outer(V_T[r, :], W_T[r, :])
                # This is a batched outer product.
                # Result shape: (rank, m, p)
                # Then transpose to (m, p, rank) and reshape to (m*p, rank).
                
                if time.time() > deadline: break
                
                M_U_tensor = V.T[:, :, None] * W.T[:, None, :] # (rank, m, p)
                M_U_mat = np.transpose(M_U_tensor, (1, 2, 0)).reshape(m*p, rank)
                
                # Solve for U
                # U (n, rank)
                # T0 (n, m*p)
                # We want U @ M_U_mat ≈ T0
                # U = T0 @ M_U_mat^T @ inv(M_U_mat @ M_U_mat^T)
                try:
                    # Use least squares solver
                    # X A = B -> A = X.T (X X.T)^-1 B.T ?
                    # Here U M = T0
                    # U = T0 M^T (M M^T)^-1
                    M_MT = M_U_mat @ M_U_mat.T
                    # Add small regularization for stability
                    M_MT += 1e-6 * np.eye(rank)
                    U = T0 @ M_U_mat.T @ np.linalg.inv(M_MT)
                except:
                    # If singular, use pseudo-inverse
                    U = T0 @ np.linalg.pinv(M_U_mat)
                
                # Update V
                # T1 (m, n*p)
                # M_V_mat (n*p, rank)
                # M_V_mat[i*p+k, r] = U[i, r] * W[k, r]
                if time.time() > deadline: break
                M_V_tensor = U.T[:, :, None] * W.T[:, None, :] # (rank, n, p)
                M_V_mat = np.transpose(M_V_tensor, (1, 2, 0)).reshape(n*p, rank)
                
                try:
                    M_VT_MT = M_V_mat @ M_V_mat.T
                    M_VT_MT += 1e-6 * np.eye(rank)
                    V = T1 @ M_V_mat.T @ np.linalg.inv(M_VT_MT)
                except:
                    V = T1 @ np.linalg.pinv(M_V_mat)
                    
                # Update W
                # T2 (p, n*m)
                # M_W_mat (n*m, rank)
                # M_W_mat[i*m+j, r] = U[i, r] * V[j, r]
                if time.time() > deadline: break
                M_W_tensor = U.T[:, :, None] * V.T[:, None, :] # (rank, n, m)
                M_W_mat = np.transpose(M_W_tensor, (1, 2, 0)).reshape(n*m, rank)
                
                try:
                    M_WT_MT = M_W_mat @ M_W_mat.T
                    M_WT_MT += 1e-6 * np.eye(rank)
                    W = T2 @ M_W_mat.T @ np.linalg.inv(M_WT_MT)
                except:
                    W = T2 @ np.linalg.pinv(M_W_mat)
                    
                # Check convergence
                # Compute error || T - T_approx ||
                # T_approx[i, j, k] = sum_r U[i, r] V[j, r] W[k, r]
                # We can compute T_approx efficiently.
                # T_approx = np.einsum('ir,jr,kr->ijk', U, V, W)
                # This might be slow for large rank, but n,m,p are small.
                T_approx = np.einsum('ir,jr,kr->ijk', U, V, W)
                err = np.linalg.norm(T - T_approx)
                
                if abs(prev_err - err) < tol:
                    converged = True
                    break
                prev_err = err
                
            # If converged, check if it's an exact integer/half-integer solution
            if converged:
                # Round to nearest 0.5
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                
                # Verify exactness
                T_approx_r = np.einsum('ir,jr,kr->ijk', U_r, V_r, W_r)
                if np.allclose(T_approx_r, T, atol=1e-9):
                    # Found a valid decomposition!
                    best_rank = rank
                    best_factors = (U_r, V_r, W_r)
                    found = True
                    break # Break restart loop, try lower rank? No, we iterate rank upwards. 
                         # If we found one, we should check if lower ranks work?
                         # We iterate from min to max. So if we find rank 15, we should stop?
                         # No, we want the MINIMAL rank.
                         # So we should iterate from min to max, and stop at the first valid one.
        
        if found:
            break
        elif time.time() > start_time + 30 and rank < 24:
             # If we've spent 30s and haven't found anything low, maybe try a few more but be careful
             pass

    # If no better decomposition found, return schoolbook
    if best_factors is None:
        best_factors = (U_sb, V_sb, W_sb)
        
    U_final, V_final, W_final = best_factors
    
    # Convert to lists of lists as per seed program interface?
    # The seed returns lists. The prompt says "factor matrices of shapes ... with integer/half-integer entries".
    # The seed program returns `U`, `V`, `W` as lists of lists.
    # Let's convert numpy arrays to lists to be safe, or return numpy arrays if the evaluator accepts them.
    # The seed program:
    # return {"U": U, "V": V, "W": W}
    # where U, V, W are lists.
    # I will return lists to match the seed exactly.
    
    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}

# EVOLVE-BLOCK-END
