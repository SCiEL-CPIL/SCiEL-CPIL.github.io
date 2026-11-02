"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i*m+j, j*p+k, k*n+i] = 1."""
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _verify_exact(U, V, W, n, m, p, tol=1e-9):
    """Verify decomposition matches the tensor exactly."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    T = _build_tensor(n, m, p)
    S = np.zeros_like(T)
    for c in range(U.shape[1]):
        S += np.einsum('a,b,c->abc', U[:, c], V[:, c], W[:, c])
    return np.allclose(S, T, atol=tol)

def _als_decompose(T, target_rank, max_iter=100, tol=1e-10, seed=0):
    """Alternating least squares for tensor decomposition."""
    np.random.seed(seed)
    nm, mp, pn = T.shape
    
    # Initialize
    U = np.random.randn(nm, target_rank) * 0.1
    V = np.random.randn(mp, target_rank) * 0.1
    W = np.random.randn(pn, target_rank) * 0.1
    
    for it in range(max_iter):
        # Update U
        for c in range(target_rank):
            # Fix V[:, c] and W[:, c], solve for U[:, c]
            # T = sum_c U_c ⊗ V_c ⊗ W_c
            # For fixed c, U_c = (V_c ⊗ W_c)^† T_residual
            residual = T.copy()
            for c2 in range(target_rank):
                if c2 != c:
                    residual -= np.einsum('a,b,d->abd', U[:, c2], V[:, c2], W[:, c2])
            # Now find U[:, c] such that sum over a,b,d of U[a,c]*V[b,c]*W[d,c]*X[a,b,d] = residual
            # This is: (V_c ⊗ W_c) · vec(U_c) = vec(residual)
            VW = np.outer(V[:, c], W[:, c])
            # vec(residual) = (W_c ⊗ V_c) @ vec(U_c)
            WCVC = np.kron(W[:, c], V[:, c])
            vec_res = residual.flatten()
            # Solve least squares
            U[:, c] = np.linalg.lstsq(WCVC, vec_res, rcond=None)[0].reshape(nm)
        
        # Update V
        for c in range(target_rank):
            residual = T.copy()
            for c2 in range(target_rank):
                if c2 != c:
                    residual -= np.einsum('a,b,d->abd', U[:, c2], V[:, c2], W[:, c2])
            UW = np.outer(U[:, c], W[:, c])
            UCWC = np.kron(W[:, c], U[:, c])
            vec_res = residual.flatten()
            V[:, c] = np.linalg.lstsq(UCWC, vec_res, rcond=None)[0].reshape(mp)
        
        # Update W
        for c in range(target_rank):
            residual = T.copy()
            for c2 in range(target_rank):
                if c2 != c:
                    residual -= np.einsum('a,b,d->abd', U[:, c2], V[:, c2], W[:, c2])
            UV = np.outer(U[:, c], V[:, c])
            UCVC = np.kron(V[:, c], U[:, c])
            vec_res = residual.flatten()
            W[:, c] = np.linalg.lstsq(UCVC, vec_res, rcond=None)[0].reshape(pn)
        
        # Check convergence
        S = np.zeros_like(T)
        for c in range(target_rank):
            S += np.einsum('a,b,d->abd', U[:, c], V[:, c], W[:, c])
        err = np.linalg.norm(S - T)
        if err < tol:
            break
    
    return U, V, W, err

def _round_to_rational(U, V, W, denom=2):
    """Round to nearest multiple of 1/denom."""
    U_r = np.round(U * denom) / denom
    V_r = np.round(V * denom) / denom
    W_r = np.round(W * denom) / denom
    return U_r, V_r, W_r

def _try_rank(T, target_rank, n, m, p, deadline, n_restarts=5):
    """Try to find a decomposition of given rank."""
    for seed in range(n_restarts):
        if time.time() > deadline:
            break
        U, V, W, err = _als_decompose(T, target_rank, max_iter=50, tol=1e-10, seed=seed)
        if err < 1e-8:
            # Try rounding
            U_r, V_r, W_r = _round_to_rational(U, V, W, denom=2)
            if _verify_exact(U_r, V_r, W_r, n, m, p):
                return U_r, V_r, W_r
            # Try without rounding
            if _verify_exact(U, V, W, n, m, p):
                return U, V, W
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    T = _build_tensor(n, m, p)
    
    # Schoolbook decomposition
    rank_sb = n * m * p
    U = np.zeros((nm, rank_sb))
    V = np.zeros((mp, rank_sb))
    W = np.zeros((pn, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank_sb
    
    # Try to reduce rank
    lower_bound = max(nm, mp, pn)
    for target_rank in range(lower_bound, best_rank):
        if time.time() > deadline:
            break
        result = _try_rank(T, target_rank, n, m, p, deadline, n_restarts=3)
        if result is not None:
            U, V, W = result
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            best_rank = target_rank
            break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END

