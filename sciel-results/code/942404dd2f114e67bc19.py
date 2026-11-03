import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = 145  # 5s safety margin before 150s kill

def _build_target_tensor(n, m, p):
    """Build the target tensor T[a,b,c] for matrix multiplication."""
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

def _verify(U, V, W, T_target, tol=1e-8):
    """Check if U,V,W form a valid decomposition with integer/half-integer entries."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Check if entries are close to integers or half-integers
    # The problem statement says "integer/half-integer entries"
    # We assume the solver finds values close to these, and we round them.
    
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    
    if np.max(np.abs(T_recon - T_target)) > tol:
        return False
    return True

def _round_and_verify(U, V, W, T_target):
    """Try to round to half-integers or integers and verify."""
    # Try half-integers
    U_h = np.round(U * 2) / 2
    V_h = np.round(V * 2) / 2
    W_h = np.round(W * 2) / 2
    if _verify(U_h, V_h, W_h, T_target, tol=1e-6):
        return U_h.tolist(), V_h.tolist(), W_h.tolist()
    
    # Try integers
    U_i = np.round(U)
    V_i = np.round(V)
    W_i = np.round(W)
    if _verify(U_i, V_i, W_i, T_target, tol=1e-6):
        return U_i.tolist(), V_i.tolist(), W_i.tolist()
    
    return None

def _als_decompose(T, R, n_iter=1000, seed=42, tol=1e-9):
    """
    Alternating Least Squares for tensor decomposition.
    T: shape (A, B, C)
    R: rank
    Returns U, V, W or None if it fails to converge.
    """
    A, B, C = T.shape
    rng = np.random.RandomState(seed)
    
    # Initialize with sparse random values to encourage sparsity and exactness
    U = rng.rand(A, R)
    V = rng.rand(B, R)
    W = rng.rand(C, R)
    # Threshold to create sparsity
    U[U < 0.7] = 0
    V[V < 0.7] = 0
    W[W < 0.7] = 0
    
    # Normalize to prevent overflow/underflow
    for r in range(R):
        nu = np.linalg.norm(U[:, r])
        nv = np.linalg.norm(V[:, r])
        nw = np.linalg.norm(W[:, r])
        if nu > 1e-9 and nv > 1e-9 and nw > 1e-9:
            U[:, r] /= nu
            V[:, r] /= nv
            W[:, r] /= nw

    # Precompute target slices
    T_a = [T[a, :, :].flatten() for a in range(A)]
    T_b = [T[:, b, :].flatten() for b in range(B)]
    T_c = [T[:, :, c].flatten() for c in range(C)]
    
    for it in range(n_iter):
        if time.time() > _DEADLINE - 2:
            break
        
        # Solve for U
        A_U = np.outer(V.flatten(), W.flatten()).reshape(B * C, R)
        for a in range(A):
            U[a, :], _, _, _ = np.linalg.lstsq(A_U, T_a[a], rcond=None)
        
        # Solve for V
        A_V = np.outer(U.flatten(), W.flatten()).reshape(A * C, R)
        for b in range(B):
            V[b, :], _, _, _ = np.linalg.lstsq(A_V, T_b[b], rcond=None)
        
        # Solve for W
        A_W = np.outer(U.flatten(), V.flatten()).reshape(A * B, R)
        for c in range(C):
            W[c, :], _, _, _ = np.linalg.lstsq(A_W, T_c[c], rcond=None)
        
        # Check convergence every 5 iterations
        if it % 5 == 0:
            T_recon = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.max(np.abs(T_recon - T))
            if err < tol:
                return U, V, W
    
    # Final check
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    if np.max(np.abs(T_recon - T)) < 1e-7:
        return U, V, W
    return None

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
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
    return U.tolist(), V.tolist(), W.tolist()

def construct():
    n, m, p = 2, 3, 4
    T = _build_target_tensor(n, m, p)
    
    # Try ranks from 20 up to 23
    # Lower is better, so try 20 first, then 21, etc.
    for rank in [20, 21, 22, 23]:
        if time.time() > _DEADLINE - 10:
            break
        
        # Try multiple seeds
        # Allocate time based on remaining time
        time_per_seed = 1.0
        num_seeds = max(1, int((_DEADLINE - 10 - time.time()) / time_per_seed))
        
        for seed in range(num_seeds):
            if time.time() > _DEADLINE - 5:
                break
            
            res = _als_decompose(T, rank, n_iter=500, seed=seed, tol=1e-9)
            if res is not None:
                U, V, W = res
                result = _round_and_verify(U, V, W, T)
                if result is not None:
                    return {"U": result[0], "V": result[1], "W": result[2]}
        
        # If ALS didn't work, try a different initialization strategy for the next rank?
        # Or just move to next rank.
        
    # Fallback: Schoolbook
    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
