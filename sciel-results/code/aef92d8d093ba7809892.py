import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Target tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    target = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i * m + j, j * p + k, k * n + i] = 1.0

    def check_decomposition(U, V, W, atol=1e-9):
        """Check if U, V, W exactly decompose target."""
        T_approx = np.einsum('ak,bk,ck->abc', U, V, W)
        return np.allclose(T_approx, target, atol=atol)

    def try_rank_als(r, num_restarts=50):
        """
        Try to find a rank r decomposition using Alternating Least Squares (ALS).
        """
        if time.time() > deadline:
            return None

        best_U, best_V, best_W = None, None, None
        best_err = float('inf')

        for restart in range(num_restarts):
            if time.time() > deadline - 10:
                break

            # Initialization
            if restart % 3 == 0:
                # Random initialization
                U = np.random.randn(nm, r)
                V = np.random.randn(mp, r)
                W = np.random.randn(pn, r)
            elif restart % 3 == 1:
                # Schoolbook initialization (padded if r < nm*mp*pn)
                U = np.zeros((nm, r))
                V = np.zeros((mp, r))
                W = np.zeros((pn, r))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            col = i * m * p + j * p + k
                            if col < r:
                                U[i * m + j, col] = 1.0
                                V[j * p + k, col] = 1.0
                                W[k * n + i, col] = 1.0
                # Perturb
                U += np.random.randn(nm, r) * 0.1
                V += np.random.randn(mp, r) * 0.1
                W += np.random.randn(pn, r) * 0.1
            else:
                # Sparse random
                U = np.random.randn(nm, r)
                V = np.random.randn(mp, r)
                W = np.random.randn(pn, r)
                mask_u = np.random.rand(nm, r) < 0.5
                mask_v = np.random.rand(mp, r) < 0.5
                mask_w = np.random.rand(pn, r) < 0.5
                U[mask_u] = 0
                V[mask_v] = 0
                W[mask_w] = 0

            # ALS Optimization
            for it in range(50):
                if time.time() > deadline - 5:
                    break

                # Update U
                # T[a, b, c] = sum_k U[a, k] V[b, k] W[c, k]
                # For each a, we have T[a, :, :] = sum_k U[a, k] (V[:, k] W[:, k]^T)
                # Let M_k = V[:, k] W[:, k]^T. Flatten to vector m_k.
                # Solve sum_k U[a, k] m_k = vec(T[a, :, :])
                
                # Precompute M_k vectors for current V, W
                # V shape (mp, r), W shape (pn, r)
                # M_k is outer product of V[:, k] and W[:, k]
                # We can compute this efficiently
                
                # Build matrix A of shape (mp*pn, r) where A[:, k] = vec(V[:, k] W[:, k]^T)
                # V[:, k] is (mp, 1), W[:, k] is (pn, 1)
                # Outer product is (mp, pn). Flatten is (mp*pn, 1).
                
                # Efficient computation:
                # A[:, k] = kron(V[:, k], W[:, k])? 
                # vec(u v^T) = v \otimes u.
                # So A[:, k] = np.kron(W[:, k], V[:, k])
                
                A_U = np.zeros((mp * pn, r))
                for k in range(r):
                    A_U[:, k] = np.kron(W[:, k], V[:, k])
                
                # Solve for each row a of U
                # vec(T[a]) = A_U @ U[a]
                # U[a] = lstsq(A_U, vec(T[a]))
                T_flat = target.reshape(nm, mp * pn)
                # Solve least squares for all rows at once?
                # U is (nm, r). T_flat is (nm, mp*pn).
                # We want U such that A_U^T U^T = T_flat^T ? No.
                # T_flat[a] = A_U @ U[a]
                # Stack T_flat: (nm, mp*pn)
                # We can solve the system X A_U = T_flat^T?
                # Let's just use lstsq for the whole block if possible, or loop.
                # Loop is safer for memory and clarity, nm is small (6).
                for a in range(nm):
                    try:
                        # Solve A_U @ u_a = T_flat[a]
                        u_a, residuals, rank, sv = np.linalg.lstsq(A_U, T_flat[a], rcond=None)
                        U[a, :] = u_a
                    except Exception:
                        pass

                # Update V
                # T[a, b, c] = sum_k U[a, k] V[b, k] W[c, k]
                # For each b, T[:, b, :] = sum_k V[b, k] (U[:, k] W[:, k]^T)
                # M_k = U[:, k] W[:, k]^T. Flatten to m_k.
                # vec(T[:, b, :]) is not quite right because T is 3D.
                # T[:, b, :] has shape (nm, pn). Flatten to (nm*pn, 1).
                # A_V[:, k] = vec(U[:, k] W[:, k]^T) = kron(W[:, k], U[:, k])
                
                A_V = np.zeros((nm * pn, r))
                for k in range(r):
                    A_V[:, k] = np.kron(W[:, k], U[:, k])
                
                # Reshape target to (mp, nm*pn)
                # T[a, b, c] -> index b, index a*pn+c
                # T_reshaped[b, a*pn+c] = T[a, b, c]
                T_reshaped_V = np.zeros((mp, nm * pn))
                for a in range(nm):
                    for b in range(mp):
                        for c in range(pn):
                            T_reshaped_V[b, a * pn + c] = target[a, b, c]
                
                for b in range(mp):
                    try:
                        v_b, residuals, rank, sv = np.linalg.lstsq(A_V, T_reshaped_V[b], rcond=None)
                        V[b, :] = v_b
                    except Exception:
                        pass

                # Update W
                # T[a, b, c] = sum_k U[a, k] V[b, k] W[c, k]
                # For each c, T[:, :, c] = sum_k W[c, k] (U[:, k] V[:, k]^T)
                # M_k = U[:, k] V[:, k]^T. Flatten to m_k.
                # A_W[:, k] = vec(U[:, k] V[:, k]^T) = kron(V[:, k], U[:, k])
                
                A_W = np.zeros((nm * mp, r))
                for k in range(r):
                    A_W[:, k] = np.kron(V[:, k], U[:, k])
                
                # Reshape target to (pn, nm*mp)
                # T[a, b, c] -> index c, index a*mp+b
                T_reshaped_W = np.zeros((pn, nm * mp))
                for a in range(nm):
                    for b in range(mp):
                        for c in range(pn):
                            T_reshaped_W[c, a * mp + b] = target[a, b, c]
                
                for c in range(pn):
                    try:
                        w_c, residuals, rank, sv = np.linalg.lstsq(A_W, T_reshaped_W[c], rcond=None)
                        W[c, :] = w_c
                    except Exception:
                        pass

            # Check convergence and rounding
            if time.time() > deadline - 5:
                break

            T_approx = np.einsum('ak,bk,ck->abc', U, V, W)
            err = np.sum((T_approx - target)**2)
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
                if err < 1e-10:
                    # Try rounding to half-integers
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    
                    if check_decomposition(U_r, V_r, W_r):
                        return U_r, V_r, W_r
                    else:
                        # Check if unrounded is valid (floating point)
                        if check_decomposition(U, V, W):
                            return U, V, W

        # If best_err is very low, try one final polish with least_squares
        if best_err < 1e-6 and best_U is not None:
            params = np.concatenate([best_U.flatten(), best_V.flatten(), best_W.flatten()])
            
            def residuals(params):
                U_p = params[:r * nm].reshape(nm, r)
                V_p = params[r * nm:r * (nm + mp)].reshape(mp, r)
                W_p = params[r * (nm + mp):].reshape(pn, r)
                T_approx = np.einsum('ak,bk,ck->abc', U_p, V_p, W_p)
                return (T_approx - target).flatten()

            try:
                res = least_squares(residuals, params, method='lm', max_nfev=500)
                U_p = res.x[:r * nm].reshape(nm, r)
                V_p = res.x[r * nm:r * (nm + mp)].reshape(mp, r)
                W_p = res.x[r * (nm + mp):].reshape(pn, r)
                
                U_r = np.round(U_p * 2) / 2
                V_r = np.round(V_p * 2) / 2
                W_r = np.round(W_p * 2) / 2
                
                if check_decomposition(U_r, V_r, W_r):
                    return U_r, V_r, W_r
                elif check_decomposition(U_p, V_p, W_p):
                    return U_p, V_p, W_p
            except Exception:
                pass

        return None

    # Try rank 23
    res = try_rank_als(23, num_restarts=100)
    if res is not None:
        U, V, W = res
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback to rank 24 (Schoolbook)
    rank = n * m * p
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
