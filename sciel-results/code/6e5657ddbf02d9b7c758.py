"""Tensor decomposition for 2x3 by 3x7 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def _build_schoolbook():
    """Return the rank-42 schoolbook decomposition."""
    n, m, p = 2, 3, 7
    rank = n * m * p  # 42
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def _verify(U, V, W, tol=1e-9):
    """Verify decomposition using einsum."""
    n, m, p = 2, 3, 7
    Ur = U.reshape(n, m, -1)
    Vr = V.reshape(m, p, -1)
    Wr = W.reshape(p, n, -1)
    T = np.einsum('ijr,jkr,kir->ijk', Ur, Vr, Wr)
    return np.allclose(T, 1.0, atol=tol)


def _als_decompose(rank, n=2, m=3, p=7, max_iter=200, tol=1e-10, seed=42):
    """Alternating least squares for tensor decomposition."""
    rng = np.random.RandomState(seed)
    U = rng.randn(n * m, rank) * 0.1
    V = rng.randn(m * p, rank) * 0.1
    W = np.ones((p * n, rank)) * 0.1
    
    for it in range(max_iter):
        # Update U
        for i in range(n):
            for j in range(m):
                A_mat = np.zeros((p, rank))
                for k in range(p):
                    A_mat[k, :] = V[j * p + k, :] * W[k * n + i, :]
                try:
                    u_sol, _, _, _ = np.linalg.lstsq(A_mat, np.ones(p), rcond=None)
                    U[i * m + j, :] = u_sol
                except:
                    pass
        
        # Update V
        for j in range(m):
            for k in range(p):
                A_mat = np.zeros((n, rank))
                for i in range(n):
                    A_mat[i, :] = U[i * m + j, :] * W[k * n + i, :]
                try:
                    v_sol, _, _, _ = np.linalg.lstsq(A_mat, np.ones(n), rcond=None)
                    V[j * p + k, :] = v_sol
                except:
                    pass
        
        # Update W
        for k in range(p):
            for i in range(n):
                A_mat = np.zeros((m, rank))
                for j in range(m):
                    A_mat[j, :] = U[i * m + j, :] * V[j * p + k, :]
                try:
                    w_sol, _, _, _ = np.linalg.lstsq(A_mat, np.ones(m), rcond=None)
                    W[k * n + i, :] = w_sol
                except:
                    pass
        
        # Check convergence
        Ur = U.reshape(n, m, rank)
        Vr = V.reshape(m, p, rank)
        Wr = W.reshape(p, n, rank)
        T = np.einsum('ijr,jkr,kir->ijk', Ur, Vr, Wr)
        err = np.max(np.abs(T - 1.0))
        if err < tol:
            if _verify(U, V, W, tol=1e-8):
                return U, V, W
    
    return None


def construct():
    """Construct rank-minimal decomposition for 2x3 x 3x7."""
    deadline = time.time() + 110.0
    
    U, V, W = _build_schoolbook()
    best_rank = 42
    best_U, best_V, best_W = U, V, W
    
    # Try ALS for decreasing ranks
    for rank in range(35, 15, -1):
        if time.time() > deadline - 20:
            break
        for seed in [42, 123, 777]:
            if time.time() > deadline - 20:
                break
            result = _als_decompose(rank, seed=seed, max_iter=100)
            if result is not None:
                U_r, V_r, W_r = result
                if U_r.shape[1] < best_rank:
                    best_rank = U_r.shape[1]
                    best_U, best_V, best_W = U_r, V_r, W_r
                    break
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
