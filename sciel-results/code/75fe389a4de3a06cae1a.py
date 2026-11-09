"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit

    # Schoolbook decomposition as baseline
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1.0
                V_sb[j * p + k, col] = 1.0
                W_sb[k * n + i, col] = 1.0
                col += 1

    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_sb.copy(), V_sb.copy(), W_sb.copy()

    def verify(U, V, W, tol=1e-9):
        """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
        r = U.shape[1]
        # T[i,j,k] should equal sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # Reshape for efficient computation
        U_mat = U.reshape(n, m, r)
        V_mat = V.reshape(m, p, r)
        W_mat = W.reshape(p, n, r)
        # Compute T[i,j,k] = sum_c U_mat[i,j,c] * V_mat[j,k,c] * W_mat[k,i,c]
        T = np.einsum('ijc,jkc,kic->ijk', U_mat, V_mat, W_mat)
        # Expected: T[i,j,k] = 1 if i,j,k are valid indices (always true for our dims)
        # The matrix multiplication tensor has T[i,j,k] = delta_{valid} = 1 for all i,j,k
        expected = np.ones((n, m, p))
        return np.allclose(T, expected, atol=tol, rtol=0)

    def try_alternating_minimization(target_rank, n_restarts=5, max_iters=200, tol=1e-10):
        """Try to find a rank-target_rank decomposition via alternating minimization."""
        best_result = None
        for restart in range(n_restarts):
            if time.time() > deadline - 5:
                break
            # Random initialization
            U = np.random.randn(n * m, target_rank)
            V = np.random.randn(m * p, target_rank)
            W = np.random.randn(p * n, target_rank)

            for it in range(max_iters):
                if time.time() > deadline - 5:
                    break

                # Alternate: fix V, W; solve for U
                # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                # For each (i,j), U[i*m+j, :] * (V[:, j*p+...] * W[:, k*n+i])^T = T[i,j,:]
                # This is a linear system in U[i*m+j, :]
                U_mat = U.reshape(n, m, target_rank)
                V_mat = V.reshape(m, p, target_rank)
                W_mat = W.reshape(p, n, target_rank)

                # Solve for U
                for i in range(n):
                    for j in range(m):
                        # T[i,j,k] = sum_c U[i*m+j, c] * V_mat[j,k,c] * W_mat[k,i,c]
                        # Let A[k,c] = V_mat[j,k,c] * W_mat[k,i,c]
                        # Then T[i,j,:] = U[i*m+j,:] @ A.T
                        A = V_mat[j] * W_mat[:, i]  # shape (p, target_rank)
                        b = np.ones(p)  # T[i,j,k] = 1 for all k
                        # Solve U_row @ A.T = b => U_row = b @ (A.T)^{-1} if square, else least squares
                        try:
                            U_row, _, _, _ = np.linalg.lstsq(A.T, b, rcond=None)
                            U_mat[i, j] = U_row
                        except np.linalg.LinAlgError:
                            pass

                # Solve for V
                for j in range(m):
                    for k in range(p):
                        # T[i,j,k] = sum_c U_mat[i,j,c] * V_mat[j,k,c] * W_mat[k,i,c]
                        # For fixed j,k: sum_c (U_mat[i,j,c] * W_mat[k,i,c]) * V_mat[j,k,c] = 1
                        # Let B[i,c] = U_mat[i,j,c] * W_mat[k,i,c]
                        # Then sum_i sum_c B[i,c] * V_mat[j,k,c] = 1 for each... 
                        # Actually: for each i, sum_c B[i,c] * V_mat[j,k,c] = 1
                        # So: B @ V_mat[j,k,:] = ones(n)
                        B = U_mat[:, j] * W_mat[k]  # shape (n, target_rank)
                        b = np.ones(n)
                        try:
                            V_row, _, _, _ = np.linalg.lstsq(B.T, b, rcond=None)
                            V_mat[j, k] = V_row
                        except np.linalg.LinAlgError:
                            pass

                # Solve for W
                for k in range(p):
                    for i in range(n):
                        # T[i,j,k] = sum_c U_mat[i,j,c] * V_mat[j,k,c] * W_mat[k,i,c]
                        # For fixed k,i: sum_j sum_c (U_mat[i,j,c] * V_mat[j,k,c]) * W_mat[k,i,c] = 1
                        # Let C[j,c] = U_mat[i,j,c] * V_mat[j,k,c]
                        # Then C @ W_mat[k,i,:] = ones(m)
                        C = U_mat[i] * V_mat[:, k]  # shape (m, target_rank)
                        b = np.ones(m)
                        try:
                            W_row, _, _, _ = np.linalg.lstsq(C.T, b, rcond=None)
                            W_mat[k, i] = W_row
                        except np.linalg.LinAlgError:
                            pass

                # Check convergence
                T = np.einsum('ijc,jkc,kic->ijk', U_mat, V_mat, W_mat)
                err = np.max(np.abs(T - 1.0))
                if err < tol:
                    return U_mat.reshape(n*m, target_rank), V_mat.reshape(m*p, target_rank), W_mat.reshape(p*n, target_rank)

                if it % 20 == 19 and err > 1.0:
                    # Diverging, restart
                    break

            # After iterations, check if we got a solution
            T = np.einsum('ijc,jkc,kic->ijk', U_mat, V_mat, W_mat)
            err = np.max(np.abs(T - 1.0))
            if err < 1e-6:
                return U_mat.reshape(n*m, target_rank), V_mat.reshape(m*p, target_rank), W_mat.reshape(p*n, target_rank)

        return best_result

    # Try structured block decomposition first
    # For 2x4 * 4x7, we can try to use Strassen-like techniques on sub-blocks
    # The key insight: if we can decompose the 4x4 middle into smaller multiplications
    
    # Try target ranks from just below schoolbook downward
    for target_rank in range(rank_schoolbook - 1, max(rank_schoolbook - 15, 0), -1):
        if time.time() > deadline - 10:
            break
        
        result = try_alternating_minimization(target_rank, n_restarts=3, max_iters=100, tol=1e-10)
        if result is not None:
            U_try, V_try, W_try = result
            if verify(U_try, V_try, W_try, tol=1e-9):
                best_rank = target_rank
                best_U, best_V, best_W = U_try, V_try, W_try
                break  # Found the minimum rank in this search range

    # If alternating minimization didn't find anything, try a different approach
    # Use the structure: for 2x4*4x7, we can view it as two 2x4*4x? multiplications
    # Or use a Kronecker-like decomposition
    
    # Try a structured approach: decompose based on splitting the 4 into 2+2
    if best_rank == rank_schoolbook and time.time() < deadline - 15:
        # Try rank = 2 * (2*2*7) + 2 * (2*2*7) - overlaps... 
        # Actually, let's try a direct structured construction
        # Split the 4-dimension into two 2-dimensions
        # C[i, k] = sum_j A[i,j] B[j,k]
        # = sum_{j1=0}^{1} sum_{j2=2}^{3} A[i,j1] B[j1,k] + A[i,j2] B[j2,k]
        # This doesn't directly help with rank reduction.
        
        # Try: use the fact that we can share some computations
        # For rank r, we need r triples (u_c, v_c, w_c) such that
        # sum_c u_c[i,j] * v_c[j,k] * w_c[k,i] = 1 for all i,j,k
        
        # A simple structured approach: for each (i,k), we need
        # sum_c u_c[i,:] * v_c[:,k] * w_c[k,i] = e_k (the k-th standard basis vector in R^4)
        # Wait, that's not quite right. Let me reconsider.
        
        # T[i,j,k] = 1 means: for each i,j,k, sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1
        # This is a very specific tensor. The schoolbook uses one term per (i,j,k).
        # To reduce rank, we need to find linear dependencies.
        
        # Try random search with more restarts for a few target ranks
        for target_rank in range(rank_schoolbook - 1, max(rank_schoolbook - 5, 0), -1):
            if time.time() > deadline - 10:
                break
            result = try_alternating_minimization(target_rank, n_restarts=10, max_iters=300, tol=1e-10)
            if result is not None:
                U_try, V_try, W_try = result
                if verify(U_try, V_try, W_try, tol=1e-9):
                    best_rank = target_rank
                    best_U, best_V, best_W = U_try, V_try, W_try
                    break

    # Convert to list format as required
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
