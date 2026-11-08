import numpy as np
import time
import itertools

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Construct the target tensor T
    # T is a tensor of shape (dim_u, dim_v, dim_w)
    # T[I, J, K] = 1 if I, J, K correspond to a valid (i, j, k) triple
    # I = i*m + j
    # J = j*p + k
    # K = k*n + i
    T = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                I = i * m + j
                J = j * p + k
                K = k * n + i
                T[I, J, K] = 1.0

    def verify_decomposition(U, V, W, T, tol=1e-9):
        """
        Verifies if U, V, W exactly decompose T within tolerance.
        Returns True if valid.
        """
        r = U.shape[1]
        # Efficient check:
        # T_approx = sum_c U[:,c] V[:,c] W[:,c]
        # We can check mode-1 slices.
        # Or just compute the full tensor if dimensions are small.
        # dim_u=6, dim_v=15, dim_w=10. Total 900 elements.
        # r=29.
        # Computation is fast.
        
        T_approx = np.zeros_like(T)
        for c in range(r):
            T_approx += U[:, c][:, np.newaxis, np.newaxis] * V[:, c][np.newaxis, :, np.newaxis] * W[:, c][np.newaxis, np.newaxis, :]
        
        return np.allclose(T_approx, T, atol=tol)

    def try_rank(r, max_iters=200, tol=1e-8, deadline=None):
        """
        Attempts to find a decomposition of rank r using vectorized ALS.
        Returns (U, V, W) if successful, else None.
        """
        # Pre-compute transposes for speed? No, just use vectorized ops.
        # U (dim_u, r), V (dim_v, r), W (dim_w, r)
        
        # We will try a few random restarts
        for restart in range(20):
            if deadline and time.time() > deadline:
                return None
                
            np.random.seed(42 + restart)
            U = np.random.randn(dim_u, r) * 0.1
            V = np.random.randn(dim_v, r) * 0.1
            W = np.random.randn(dim_w, r) * 0.1
            
            # Initialize with some structure?
            # Often, starting with random is fine, but scaling matters.
            
            prev_error = float('inf')
            
            for it in range(max_iters):
                if deadline and time.time() > deadline:
                    return None

                # Update U
                # For each i in dim_u, solve:
                # T[i, :, :] = sum_c U[i, c] (V[:, c] * W[:, c])
                # Let M be matrix of size (dim_v * dim_w, r) where M[j*dim_w+k, c] = V[j,c]*W[k,c]
                # Then T[i, :, :].flatten() = M^T U[i, :]^T ? No.
                # T[i, j, k] = sum_c U[i, c] V[j, c] W[k, c]
                # Let b_i = T[i, :, :].flatten()
                # Let A be matrix (dim_v*dim_w, r) with A[idx, c] = V[j,c] W[k,c]
                # We want to find x = U[i, :] such that A^T x = b_i?
                # Wait.
                # sum_c x_c (V[j,c] W[k,c]) = b_{j,k}
                # This is linear in x.
                # The matrix for this system is A^T?
                # Let's check dimensions.
                # b has length dim_v * dim_w.
                # x has length r.
                # The equation is: for each (j,k), sum_c x_c A_{(j,k), c} = b_{(j,k)}.
                # This is A^T x = b.
                # So we solve A^T x = b.
                # A is (dim_v*dim_w, r). A^T is (r, dim_v*dim_w).
                # This is an underdetermined system if r < dim_v*dim_w.
                # We want the minimum norm solution? Or just any solution?
                # Actually, in ALS, we usually fix V, W and solve for U.
                # The system is linear in U.
                # We can use np.linalg.lstsq(A.T, b, rcond=None).
                
                # Vectorization:
                # We can form a big matrix for all i at once?
                # B is (dim_u, dim_v, dim_w).
                # We want U (dim_u, r).
                # For each i, A_i is the same? No, A depends on V, W, which are fixed.
                # So A is the same for all i!
                # A_{(j,k), c} = V[j, c] W[k, c].
                # This is the Khatri-Rao product of V and W (column-wise).
                # A = V \otimes W (Khatri-Rao).
                # Shape: (dim_v * dim_w, r).
                
                A_u = np.einsum('jc,kc->jkc', V, W).reshape(dim_v * dim_w, r)
                # A_u has shape (dim_v*dim_w, r)
                
                # We want to solve A_u.T @ U[i, :] = T[i, :, :].flatten()
                # This is equivalent to:
                # U.T @ A_u = T.reshape(dim_u, -1).T ?
                # Let T_flat be (dim_u, dim_v*dim_w).
                # T_flat[i, idx] = T[i, j, k].
                # Equation: T_flat[i, idx] = sum_c U[i, c] A_u[idx, c].
                # So T_flat = U @ A_u.T.
                # We want to find U such that U @ A_u.T = T_flat.
                # This is a linear least squares problem for U (row-wise).
                # We can solve: U = T_flat @ (A_u.T)^\dagger
                # Or use lstsq on the transposed system:
                # (A_u.T).T @ U.T = T_flat.T  => A_u @ U.T = T_flat.T
                # So we solve A_u X = T_flat.T for X (which is U.T).
                
                T_flat = T.reshape(dim_u, dim_v * dim_w)
                # Solve A_u @ X = T_flat.T
                # A_u is (dim_v*dim_w, r). X is (r, dim_u). T_flat.T is (dim_v*dim_w, dim_u).
                try:
                    X, residuals, rank, s = np.linalg.lstsq(A_u, T_flat.T, rcond=None)
                    U = X.T
                except np.linalg.LinAlgError:
                    pass

                # Update V
                # Similarly, T[:, j, k] = sum_c U[i, c] V[j, c] W[k, c]
                # Fix U, W.
                # For each j, solve for V[j, :].
                # A_v_{(i,k), c} = U[i, c] W[k, c].
                # A_v = U \otimes W. Shape (dim_u * dim_w, r).
                # T_v_flat is (dim_v, dim_u * dim_w).
                # T_v_flat[j, idx] = T[i, j, k].
                # T_v_flat = V @ A_v.T.
                # Solve A_v @ Y = T_v_flat.T.
                
                A_v = np.einsum('ic,kc->ikc', U, W).reshape(dim_u * dim_w, r)
                T_v_flat = np.transpose(T, (1, 0, 2)).reshape(dim_v, dim_u * dim_w)
                try:
                    Y, residuals, rank, s = np.linalg.lstsq(A_v, T_v_flat.T, rcond=None)
                    V = Y.T
                except np.linalg.LinAlgError:
                    pass

                # Update W
                # Fix U, V.
                # For each k, solve for W[k, :].
                # A_w_{(i,j), c} = U[i, c] V[j, c].
                # A_w = U \otimes V. Shape (dim_u * dim_v, r).
                # T_w_flat is (dim_w, dim_u * dim_v).
                # T_w_flat[k, idx] = T[i, j, k].
                # T_w_flat = W @ A_w.T.
                # Solve A_w @ Z = T_w_flat.T.
                
                A_w = np.einsum('ic,jc->ijc', U, V).reshape(dim_u * dim_v, r)
                T_w_flat = np.transpose(T, (2, 0, 1)).reshape(dim_w, dim_u * dim_v)
                try:
                    Z, residuals, rank, s = np.linalg.lstsq(A_w, T_w_flat.T, rcond=None)
                    W = Z.T
                except np.linalg.LinAlgError:
                    pass

                # Check convergence
                # Compute error
                T_approx = np.zeros_like(T)
                for c in range(r):
                    T_approx += U[:, c][:, np.newaxis, np.newaxis] * V[:, c][np.newaxis, :, np.newaxis] * W[:, c][np.newaxis, np.newaxis, :]
                
                error = np.linalg.norm(T_approx - T)
                if error < tol:
                    return U, V, W
                
                if abs(prev_error - error) < 1e-12:
                    break
                prev_error = error
            
            # End iterations for this restart
            
            # Post-processing: Try to round to half-integers
            # This is crucial because the task requires integer/half-integer entries.
            # If the ALS converged to a solution with small floating point errors,
            # rounding might snap it to the exact rational solution.
            
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            if verify_decomposition(U_rounded, V_rounded, W_rounded, T, tol=1e-9):
                return U_rounded, V_rounded, W_rounded
            
            # Try rounding to integers as well?
            U_int = np.round(U)
            V_int = np.round(V)
            W_int = np.round(W)
            if verify_decomposition(U_int, V_int, W_int, T, tol=1e-9):
                return U_int, V_int, W_int

            # If rounding didn't work, maybe we can refine?
            # If error is small, we can try to solve for one factor exactly given the others rounded.
            # But for now, let's just return the float solution if it's close enough?
            # No, the interface requires integer/half-integer.
            # So we must find a valid half-integer solution.
            
            # If we didn't find a valid half-integer solution, we continue to next restart.
            
        # End restarts
        return None

    # Try to find a decomposition of rank < 30
    # Start from 29 and go down?
    # Theoretical lower bound is roughly 28.
    
    deadline = time.time() + 110 # Leave 40s margin for safety
    
    # Try rank 29
    res = try_rank(29, deadline=deadline)
    if res is not None and time.time() < deadline:
        U, V, W = res
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }

    # Try rank 28 (just in case, though 29 is more likely)
    if time.time() < deadline:
        res = try_rank(28, deadline=deadline)
        if res is not None:
            U, V, W = res
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }

    # Fallback to schoolbook (rank 30)
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
