import time
import numpy as np
import scipy.optimize

_START_TIME = time.time()
_DEADLINE = 140  # 10s safety margin

def _build_target_tensor(n, m, p):
    """Build the target matrix multiplication tensor."""
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    T = np.zeros((dim_U, dim_V, dim_W))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T


def _verify(U, V, W, n, m, p, tol=1e-10, T_target=None):
    """Check if U,V,W form a valid decomposition."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    if T_target is None:
        T_target = _build_target_tensor(n, m, p)
    
    # Compute reconstructed tensor
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    
    # Check difference
    if np.max(np.abs(T_recon - T_target)) > tol:
        return False
    return True

def _als_decompose(T, R, n_iter=200, seed=42, init_U=None, init_V=None, init_W=None):
    """
    Alternating Least Squares for tensor decomposition.
    T: shape (A, B, C)
    R: rank
    Returns U, V, W or None if it fails to converge.
    """
    A, B, C = T.shape
    rng = np.random.RandomState(seed)
    
    # Initialize
    if init_U is not None:
        U = init_U.copy()
        V = init_V.copy()
        W = init_W.copy()
    else:
        U = rng.randn(A, R) * 0.1
        V = rng.randn(B, R) * 0.1
        W = rng.randn(C, R) * 0.1
    
    best_result = None
    best_err = float('inf')
    
    for it in range(n_iter):
        # Solve for U
        A_U = np.einsum('br,cr->bcr', V, W).reshape(B * C, R)
        T_a = T.reshape(A, B * C)
        U_new = np.linalg.lstsq(A_U, T_a.T, rcond=None)[0].T
        
        # Solve for V
        A_V = np.einsum('ar,cr->acr', U_new, W).reshape(A * C, R)
        T_b = T.transpose(1, 0, 2).reshape(B, A * C)
        V_new = np.linalg.lstsq(A_V, T_b.T, rcond=None)[0].T
        
        # Solve for W
        A_W = np.einsum('ar,br->abr', U_new, V_new).reshape(A * B, R)
        T_c = T.transpose(2, 0, 1).reshape(C, A * B)
        W_new = np.linalg.lstsq(A_W, T_c.T, rcond=None)[0].T
        
        U, V, W = U_new, V_new, W_new
        
        # Check convergence every 5 iterations or at the end
        if it % 5 == 0 or it == n_iter - 1:
            T_recon = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.max(np.abs(T_recon - T))
            if err < best_err:
                best_err = err
                best_result = (U.copy(), V.copy(), W.copy())
            if err < 1e-10:
                return U, V, W
                
    # Final check
    if best_result is not None and best_err < 1e-6:
        return best_result
    
    return None

def _structured_init(rank, n, m, p, seed=42):
    """
    Create a structured initialization that respects the matrix multiplication structure.
    For 2x3*3x4, we know the tensor has 24 non-zero entries.
    We try to create rank-rank inits that are close to valid decompositions.
    """
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    rng = np.random.RandomState(seed)
    
    U = np.zeros((dim_U, rank))
    V = np.zeros((dim_V, rank))
    W = np.zeros((dim_W, rank))
    
    # Start with the schoolbook pattern for the first min(24, rank) columns
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if col >= rank:
                    break
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U[a, col] = 1.0
                V[b, col] = 1.0
                W[c, col] = 1.0
                col += 1
    
    # For remaining columns, add structured perturbations
    for r in range(col, rank):
        # Use small random values but structured
        U[:, r] = rng.randn(dim_U) * 0.1
        V[:, r] = rng.randn(dim_V) * 0.1
        W[:, r] = rng.randn(dim_W) * 0.1
    
    return U, V, W

def _try_rank_als(rank, n, m, p, max_time=5, seed=42, T_target=None):
    """Try to find a rank-rank decomposition via ALS."""
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    
    if T_target is None:
        T_target = _build_target_tensor(n, m, p)

    start_time = time.time()
    
    # Try multiple initialization strategies
    for init_seed in [seed, seed + 100, seed + 200, seed + 300, seed + 400, seed + 500]:
        if time.time() - start_time > max_time - 1:
            break
        
        # Strategy 1: Random init
        res = _als_decompose(T_target, rank, n_iter=200, seed=init_seed)
        
        if res is not None:
            U, V, W = res
            if _try_round_and_verify(U, V, W, n, m, p, T_target):
                return (U.tolist(), V.tolist(), W.tolist())
        
        # Strategy 2: Structured init
        if time.time() - start_time > max_time - 1:
            break
        U_init, V_init, W_init = _structured_init(rank, n, m, p, seed=init_seed + 1000)
        
        res = _als_decompose(T_target, rank, n_iter=200, seed=init_seed + 2000, 
                            init_U=U_init, init_V=V_init, init_W=W_init)
        
        if res is not None:
            U, V, W = res
            if _try_round_and_verify(U, V, W, n, m, p, T_target):
                return (U.tolist(), V.tolist(), W.tolist())
            
    return None

def _try_round_and_verify(U, V, W, n, m, p, T_target, tol=1e-10):
    """Try various rounding strategies and verify."""
    # First check if already close enough
    if _verify(U, V, W, n, m, p, tol=1e-8, T_target=T_target):
        return True
    
    # Try half-integers
    U_r = np.round(U * 2) / 2
    V_r = np.round(V * 2) / 2
    W_r = np.round(W * 2) / 2
    if _verify(U_r, V_r, W_r, n, m, p, tol=tol, T_target=T_target):
        U, V, W = U_r, V_r, W_r
        return True
    
    # Try integers
    U_r = np.round(U)
    V_r = np.round(V)
    W_r = np.round(W)
    if _verify(U_r, V_r, W_r, n, m, p, tol=tol, T_target=T_target):
        U, V, W = U_r, V_r, W_r
        return True
    
    # Try with factor 4
    U_r = np.round(U * 4) / 4
    V_r = np.round(V * 4) / 4
    W_r = np.round(W * 4) / 4
    if _verify(U_r, V_r, W_r, n, m, p, tol=tol, T_target=T_target):
        U, V, W = U_r, V_r, W_r
        return True
    
    # Try with factor 8
    U_r = np.round(U * 8) / 8
    V_r = np.round(V * 8) / 8
    W_r = np.round(W * 8) / 8
    if _verify(U_r, V_r, W_r, n, m, p, tol=tol, T_target=T_target):
        U, V, W = U_r, V_r, W_r
        return True
    
    # Try with factor 16
    U_r = np.round(U * 16) / 16
    V_r = np.round(V * 16) / 16
    W_r = np.round(W * 16) / 16
    if _verify(U_r, V_r, W_r, n, m, p, tol=tol, T_target=T_target):
        U, V, W = U_r, V_r, W_r
        return True
    
    # Final polish: if error is small, do one more ALS iteration with tighter tolerance
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    err = np.max(np.abs(T_recon - T_target))
    if err < 1e-6:
        # Do a few more iterations to tighten
        for _ in range(15):
            A_U = np.einsum('br,cr->bcr', V, W).reshape(V.shape[0] * W.shape[0], V.shape[1])
            T_a = T_target.reshape(U.shape[0], -1)
            U = np.linalg.lstsq(A_U, T_a.T, rcond=None)[0].T
            A_V = np.einsum('ar,cr->acr', U, W).reshape(U.shape[0] * W.shape[0], U.shape[1])
            T_b = T_target.transpose(1, 0, 2).reshape(V.shape[0], -1)
            V = np.linalg.lstsq(A_V, T_b.T, rcond=None)[0].T
            A_W = np.einsum('ar,br->abr', U, V).reshape(U.shape[0] * V.shape[0], U.shape[1])
            T_c = T_target.transpose(2, 0, 1).reshape(W.shape[0], -1)
            W = np.linalg.lstsq(A_W, T_c.T, rcond=None)[0].T
        
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        err = np.max(np.abs(T_recon - T_target))
        if err < 1e-8:
            return True
    
    return False

def construct():
    n, m, p = 2, 3, 4
    T_target = _build_target_tensor(n, m, p)
    
    # Try ranks from 22 to 24
    # 22 is the conjectured minimum, 24 is schoolbook
    for rank in [22, 23, 24]:
        if time.time() > _DEADLINE - 15:
            break
            
        # Try multiple seeds
        # Increase seeds for lower ranks to improve chance of finding the global min
        num_seeds = 60 if rank == 22 else (30 if rank == 23 else 10)
        for seed in range(num_seeds):
            if time.time() > _DEADLINE - 3:
                break
            res = _try_rank_als(rank, n, m, p, seed=seed, T_target=T_target)
            if res is not None:
                U, V, W = res
                return {"U": U, "V": V, "W": W}
                
    # Fallback: Schoolbook
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
