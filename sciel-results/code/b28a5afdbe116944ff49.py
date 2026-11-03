import time
import numpy as np

DIMENSIONS = [3, 3, 5]

def _schoolbook(n, m, p):
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

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 150s kill

    size1 = n * m
    size2 = m * p
    size3 = p * n

    # Build the matrix multiplication tensor T of shape (size1, size2, size3)
    # T[i*m+j, j*p+k, k*n+i] = 1 for all valid (i,j,k)
    T = np.zeros((size1, size2, size3), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    def _reconstruct(U, V, W):
        return np.einsum('ir,jr,kr->ijk', U, V, W)

    def _round_to_half(U, V, W):
        """Round to nearest half-integer and verify exactness."""
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        recon = _reconstruct(U_r, V_r, W_r)
        err = np.linalg.norm(T - recon)
        if err < 1e-8:
            return U_r, V_r, W_r, True
        # Also try integer rounding
        U_i = np.round(U)
        V_i = np.round(V)
        W_i = np.round(W)
        recon_i = _reconstruct(U_i, V_i, W_i)
        err_i = np.linalg.norm(T - recon_i)
        if err_i < 1e-8:
            return U_i, V_i, W_i, True
        return None, None, None, False

    best_result = None
    best_rank = n * m * p

    # Strategy: Try ranks from 18 upward (Strassen for 3x3x3 is 7, 
    # for 3x3x5 we expect something around 18-25)
    # Use SVD-based initialization + alternating least squares + gradient polish
    
    for rank in range(18, min(best_rank, 28)):
        if time.time() > deadline:
            break

        found = False
        for trial in range(30):
            if time.time() > deadline:
                break

            np.random.seed(42 + rank * 100 + trial)

            # Initialize using SVD of a random matricization of T
            # Fix mode 3: form matrix M of shape (size1*size2, size3)
            M = np.zeros((size1 * size2, size3), dtype=np.float64)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i * m + j
                        b = j * p + k
                        c = k * n + i
                        M[a * size2 + b, c] = 1.0

            # SVD of M to get initial structure
            U_svd, s_vals, Vt_svd = np.linalg.svd(M, full_matrices=False)
            rank_svd = np.sum(s_vals > 1e-10)

            # Use SVD components to initialize
            # M[(a,b), c] = sum_r U_svd[(a,b),r] * s[r] * Vt_svd[r,c]
            # We need T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
            # Approximate: set W = Vt_svd[:rank,:].T * s[:rank]
            # and split U_svd[:size1*size2, :rank] into U and V factors

            if rank <= rank_svd:
                # Use top-rank components
                W_init = (s_vals[:rank] * Vt_svd[:rank, :]).T  # shape (size3, rank)
                X = U_svd[:, :rank] * s_vals[:rank]  # shape (size1*size2, rank)
                # Split X into U and V via SVD of X reshaped
                # X[(a,b), r] ≈ sum_s U[a,s]*V[b,s] -- use SVD of X
                U_x, s_x, Vt_x = np.linalg.svd(X, full_matrices=False)
                min_dim = min(size1, size2)
                split_rank = min(rank, min_dim)
                U_init = U_x[:, :split_rank] * np.sqrt(s_x[:split_rank])
                V_init = (Vt_x[:split_rank, :] * np.sqrt(s_x[:split_rank])).T
                # If split_rank < rank, pad with small random
                if split_rank < rank:
                    U_init = np.hstack([U_init, np.random.randn(size1, rank - split_rank) * 0.01])
                    V_init = np.hstack([V_init, np.random.randn(size2, rank - split_rank) * 0.01])
            else:
                # Rank > SVD rank: use full SVD + random padding
                W_init = (s_vals * Vt_svd).T  # shape (size3, rank_svd)
                if rank_svd < rank:
                    W_init = np.hstack([W_init, np.random.randn(size3, rank - rank_svd) * 0.01])
                X = U_svd * s_vals
                U_x, s_x, Vt_x = np.linalg.svd(X, full_matrices=False)
                split_rank = min(rank, min(size1, size2))
                U_init = U_x[:, :split_rank] * np.sqrt(s_x[:split_rank])
                V_init = (Vt_x[:split_rank, :] * np.sqrt(s_x[:split_rank])).T
                if split_rank < rank:
                    U_init = np.hstack([U_init, np.random.randn(size1, rank - split_rank) * 0.01])
                    V_init = np.hstack([V_init, np.random.randn(size2, rank - split_rank) * 0.01])

            # Also try pure random init in some trials
            if trial % 3 == 2:
                U_init = np.random.randn(size1, rank) * 0.5
                V_init = np.random.randn(size2, rank) * 0.5
                W_init = np.random.randn(size3, rank) * 0.5

            U = U_init.astype(np.float64).copy()
            V = V_init.astype(np.float64).copy()
            W = W_init.astype(np.float64).copy()

            # Alternating least squares + gradient descent hybrid
            for iteration in range(300):
                if time.time() > deadline:
                    break

                recon = _reconstruct(U, V, W)
                residual = T - recon
                err = np.linalg.norm(residual)

                if err < 1e-10:
                    break

                # Alternating least squares: fix two, solve for third
                # Update U: for each r, U[:,r] minimizes ||T - sum_r' U_r' V_r' W_r'||^2
                # Fix V, W: the problem is linear in U
                # T - sum_{r'!=r} U[:,r'] V[:,r'] W[:,r'] = U[:,r] V[:,r] W[:,r] (outer product)
                # This is still nonlinear per r. Use gradient step.

                # Adaptive learning rate
                lr = 0.05 / (1 + iteration * 0.01)

                grad_U = -2 * np.einsum('ijk,jr,kr->ir', residual, V, W)
                grad_V = -2 * np.einsum('ijk,ir,kr->jr', residual, U, W)
                grad_W = -2 * np.einsum('ijk,ir,jr->kr', residual, U, V)

                # Gradient norm for adaptive step
                gnorm = np.sqrt(np.sum(grad_U**2) + np.sum(grad_V**2) + np.sum(grad_W**2))
                if gnorm < 1e-15:
                    break

                # Scale step to control norm
                step_scale = min(1.0, 0.1 / (gnorm + 1e-10))
                lr_eff = lr * step_scale

                U -= lr_eff * grad_U
                V -= lr_eff * grad_V
                W -= lr_eff * grad_W

                # Periodic ALS step for better convergence
                if iteration % 10 == 5:
                    # ALS: fix V and W, solve for U via linear system
                    # For each r: minimize ||residual + U[:,r] V[:,r] W[:,r] - sum_{r'!=r}...||^2
                    # Simplified: use normal equations on the full tensor
                    # T_flat = sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r]
                    # For fixed V, W: T = U @ (V ⊙ W) where ⊙ is Khatri-Rao product
                    # This is a linear problem: T_reshaped = U @ KR(V,W)
                    # Let's do a simple ALS update
                    KR_VW = np.einsum('jr,kr->rjk', V, W)  # (rank, size2, size3)
                    # T[i,j,k] = sum_r U[i,r] * KR_VW[r,j,k]
                    # Reshape: T_mat[i, r*j+k] = U[i,r] * ... -- not quite linear
                    # Skip ALS for now, gradient is sufficient

            # Final check
            recon = _reconstruct(U, V, W)
            err = np.linalg.norm(T - recon)

            if err < 1e-6:
                U_r, V_r, W_r, ok = _round_to_half(U, V, W)
                if ok:
                    best_result = {
                        'U': U_r.tolist(),
                        'V': V_r.tolist(),
                        'W': W_r.tolist()
                    }
                    best_rank = rank
                    found = True
                    break

        if found:
            break

    if best_result is None:
        # Return schoolbook
        best_result = _schoolbook(n, m, p)

    return best_result
