"""Rewritten solver: ALS-based rank search for 2x5x5 multiplication tensor."""
import time
import numpy as np

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _build_target_tensor():
    """Build the multiplication tensor T with shape (n*m, m*p, p*n).
    T[(i,j),(j',k'),(k,i')] = 1 if i==i' and j==j' and j'==k, else 0.
    """
    n, m, p = DIMENSIONS
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # T[(i,j),(j',k'),(k,i')] with j'=j, k'=k, i'=i
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _matmul_check(U, V, W, A, B):
    """Compute C = A @ B using the decomposition."""
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    # X[r] = sum_{i,j} U[(i,j),r] A[i,j]
    X = A.reshape(1, -1) @ U  # (1, rank)
    Y = B.reshape(1, -1) @ V  # (1, rank)
    # C[k,i] = sum_r W[(k,i),r] X[r] Y[r]
    C = (W * (X * Y)[None, :]).T  # (p*n,) -> reshape
    return C.reshape(p, n).T  # (n, p)

def _validate(U, V, W, tol=1e-8, num_trials=30):
    """Validate exact decomposition with random trials."""
    n, m, p = DIMENSIONS
    rng = np.random.default_rng(12345)
    for _ in range(num_trials):
        A = rng.standard_normal((n, m))
        B = rng.standard_normal((m, p))
        C_expected = A @ B
        C_computed = _matmul_check(U, V, W, A, B)
        if not np.allclose(C_computed, C_expected, atol=tol, rtol=0):
            return False
    return True

def _loss(U, V, W, T, num_trials=20):
    """Compute MSE loss over random A,B."""
    n, m, p = DIMENSIONS
    rng = np.random.default_rng(777)
    total = 0.0
    for _ in range(num_trials):
        A = rng.standard_normal((n, m))
        B = rng.standard_normal((m, p))
        C_expected = A @ B
        C_computed = _matmul_check(U, V, W, A, B)
        total += np.sum((C_computed - C_expected) ** 2)
    return total / num_trials

def _snap_to_half_int(M, tol=1e-6):
    M2 = M.copy()
    for i in range(M2.shape[0]):
        for j in range(M2.shape[1]):
            nearest = np.round(M2[i, j] * 2) / 2
            if abs(M2[i, j] - nearest) < tol:
                M2[i, j] = nearest
    return M2

def _als_step_U(V, W, T, rank):
    """Solve for U given V, W by least squares on the tensor."""
    n, m, p = DIMENSIONS
    # T = sum_r U[:,r] outer V[:,r] outer W[:,r]
    # For fixed V,W: T[a, :, :] = sum_r U[a,r] * (V[:,r] outer W[:,r])
    # So T_reshaped (n*m, m*p*p*n) = U @ (V outer W)_reshaped
    # Build matrix M where M[a, (r, b, c)] = V[b,r] * W[c,r]
    # Then T[a, b, c] = sum_r U[a,r] V[b,r] W[c,r]
    # Vectorize: T2 = T.reshape(n*m, m*p*p*n)
    # M2 = (V[:, :, None, None] * W[None, None, :, :]).transpose(2, 0, 3, 1).reshape(m*p*p*n, rank)
    # Actually: M2[(b, c), r] = V[b, r] * W[c, r]
    Vb = V[:, :, None]  # (m*p, rank, 1)
    Wc = W[None, :, :]  # (1, p*n, rank)
    M2 = (Vb * Wc).reshape(m * p * p * n, rank)
    T2 = T.reshape(n * m, m * p * p * n)
    # Solve T2 = U @ M2
    U_sol, _, _, _ = np.linalg.lstsq(M2.T, T2.T, rcond=None)
    return U_sol.T

def _als_step_V(U, W, T, rank):
    n, m, p = DIMENSIONS
    # T[a, b, c] = sum_r U[a,r] V[b,r] W[c,r]
    # For fixed U,W: T[:, b, :] = sum_r V[b,r] * (U[:,r] outer W[:,r])
    # M2[(a, c), r] = U[a,r] * W[c,r]
    Ua = U[:, :, None]  # (n*m, rank, 1)
    Wc = W[None, :, :]  # (1, p*n, rank)
    M2 = (Ua * Wc).reshape(n * m * p * n, rank)
    T2 = T.reshape(m * p, n * m * p * n)
    V_sol, _, _, _ = np.linalg.lstsq(M2.T, T2.T, rcond=None)
    return V_sol.T

def _als_step_W(U, V, T, rank):
    n, m, p = DIMENSIONS
    # T[a, b, c] = sum_r U[a,r] V[b,r] W[c,r]
    # For fixed U,V: T[:, :, c] = sum_r W[c,r] * (U[:,r] outer V[:,r])
    # M2[(a, b), r] = U[a,r] * V[b,r]
    Ua = U[:, :, None]  # (n*m, rank, 1)
    Vb = V[None, :, :]  # (1, m*p, rank)
    M2 = (Ua * Vb).reshape(n * m * m * p, rank)
    T2 = T.reshape(p * n, n * m * m * p)
    W_sol, _, _, _ = np.linalg.lstsq(M2.T, T2.T, rcond=None)
    return W_sol.T

def _als_decompose(T, rank, init_seed, max_iter=300, tol=1e-12, time_budget=20):
    """Run ALS for a given rank and seed."""
    n, m, p = DIMENSIONS
    start = time.time()
    rng = np.random.default_rng(init_seed)
    U = rng.standard_normal((n * m, rank))
    V = rng.standard_normal((m * p, rank))
    W = rng.standard_normal((p * n, rank))
    prev_loss = np.inf
    for it in range(max_iter):
        if time.time() > start + time_budget:
            break
        U = _als_step_U(V, W, T, rank)
        V = _als_step_V(U, W, T, rank)
        W = _als_step_W(U, V, T, rank)
        # Compute exact tensor residual
        # T_recon[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        # Use einsum
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        loss = np.sum((T_recon - T) ** 2)
        if abs(prev_loss - loss) < tol:
            break
        prev_loss = loss
    # Final residual check
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    final_loss = np.sum((T_recon - T) ** 2)
    return U, V, W, final_loss

def _try_rank(rank, T, time_budget=30, max_restarts=8):
    """Try to find a rank-`rank` decomposition via ALS with restarts."""
    start = time.time()
    best = None
    best_loss = np.inf
    for seed in range(max_restarts):
        if time.time() > start + time_budget:
            break
        U, V, W, loss = _als_decompose(T, rank, init_seed=seed * 1000 + 42, time_budget=time_budget / max_restarts)
        if loss < best_loss:
            best_loss = loss
            best = (U, V, W)
    if best is not None and best_loss < 1e-10:
        U, V, W = best
        # Snap to half-integers
        U = _snap_to_half_int(U)
        V = _snap_to_half_int(V)
        W = _snap_to_half_int(W)
        if _validate(U, V, W, tol=1e-7, num_trials=20):
            return U, V, W
        # Try without snapping
        if _validate(best[0], best[1], best[2], tol=1e-7, num_trials=20):
            return best
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 10s safety margin

    # Schoolbook as fallback
    U, V, W = _schoolbook()
    best_rank = U.shape[1]
    best_result = (U, V, W)

    T = _build_target_tensor()

    # Try ranks in increasing order
    for target_rank in [10, 12, 14, 16, 18, 20, 25, 30, 35, 40, 45]:
        if time.time() > deadline - 15:
            break
        remaining = deadline - time.time() - 5
        if remaining < 5:
            break
        result = _try_rank(target_rank, T, time_budget=min(remaining, 30))
        if result is not None:
            U_new, V_new, W_new = result
            if _validate(U_new, V_new, W_new, tol=1e-7, num_trials=30):
                best_rank = target_rank
                best_result = (U_new, V_new, W_new)
                break

    U, V, W = best_result
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
