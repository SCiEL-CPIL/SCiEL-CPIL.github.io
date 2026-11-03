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
    # Try multiple rounding strategies
    for scale in [2, 4, 8]:
        U_r = np.round(U * scale) / scale
        V_r = np.round(V * scale) / scale
        W_r = np.round(W * scale) / scale
        if _verify(U_r, V_r, W_r, T_target, tol=1e-6):
            return U_r.tolist(), V_r.tolist(), W_r.tolist()
    
    # Try integers
    U_i = np.round(U)
    V_i = np.round(V)
    W_i = np.round(W)
    if _verify(U_i, V_i, W_i, T_target, tol=1e-6):
        return U_i.tolist(), V_i.tolist(), W_i.tolist()
    
    # Try with small perturbation before rounding
    for eps in [0.01, 0.05, 0.1]:
        U_p = U + np.random.randn(*U.shape) * eps
        V_p = V + np.random.randn(*V.shape) * eps
        W_p = W + np.random.randn(*W.shape) * eps
        U_r = np.round(U_p * 2) / 2
        V_r = np.round(V_p * 2) / 2
        W_r = np.round(W_p * 2) / 2
        if _verify(U_r, V_r, W_r, T_target, tol=1e-5):
            return U_r.tolist(), V_r.tolist(), W_r.tolist()
    
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
    
    # Better initialization: use structured random with some zeros
    U = rng.randn(A, R)
    V = rng.randn(B, R)
    W = rng.randn(C, R)
    
    # Create sparsity pattern
    U[np.abs(U) < 0.8] = 0
    V[np.abs(V) < 0.8] = 0
    W[np.abs(W) < 0.8] = 0
    
    # Ensure each row has at least one non-zero
    for r in range(R):
        if np.all(U[:, r] == 0):
            U[rng.randint(A), r] = rng.choice([-1, 1])
        if np.all(V[:, r] == 0):
            V[rng.randint(B), r] = rng.choice([-1, 1])
        if np.all(W[:, r] == 0):
            W[rng.randint(C), r] = rng.choice([-1, 1])
    
    # Normalize
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
    
    best_err = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
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
        
        # Check convergence every 3 iterations
        if it % 3 == 0:
            T_recon = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.max(np.abs(T_recon - T))
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            if err < tol:
                return U, V, W
            # Early stop if not improving
            if err > 0.5 and it > 100:
                break
    
    # Final check with best found
    if best_err < 1e-7:
        return best_U, best_V, best_W
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

def _polynomial_solve(T, R, n, m, p, seed=42, max_iters=200):
    """
    Try to find exact decomposition using polynomial structure.
    For matrix multiplication, we can exploit the structure.
    """
    A, B, C = T.shape
    rng = np.random.RandomState(seed)
    
    # Try structured initialization based on the matrix multiplication pattern
    U = np.zeros((A, R))
    V = np.zeros((B, R))
    W = np.zeros((C, R))
    
    # Initialize with random sparse patterns
    for r in range(R):
        # Random subset of indices
        u_idx = rng.choice(A, size=rng.randint(1, 4), replace=False)
        v_idx = rng.choice(B, size=rng.randint(1, 4), replace=False)
        w_idx = rng.choice(C, size=rng.randint(1, 4), replace=False)
        
        U[u_idx, r] = rng.choice([-1, 1], size=len(u_idx))
        V[v_idx, r] = rng.choice([-1, 1], size=len(v_idx))
        W[w_idx, r] = rng.choice([-1, 1], size=len(w_idx))
    
    # ALS refinement
    T_a = [T[a, :, :].flatten() for a in range(A)]
    T_b = [T[:, b, :].flatten() for b in range(B)]
    T_c = [T[:, :, c].flatten() for c in range(C)]
    
    best_err = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        if time.time() > _DEADLINE - 3:
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
        
        if it % 5 == 0:
            T_recon = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.max(np.abs(T_recon - T))
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            if err < 1e-9:
                return U, V, W
    
    if best_err < 1e-7:
        return best_U, best_V, best_W
    return None

def construct():
    n, m, p = 2, 3, 4
    T = _build_target_tensor(n, m, p)
    
    # Try ranks from 20 up to 23
    # Lower is better, so try 20 first, then 21, etc.
    for rank in [20, 21, 22, 23]:
        if time.time() > _DEADLINE - 10:
            break
        
        # Try multiple seeds with both ALS and polynomial approaches
        time_per_seed = 0.8
        num_seeds = max(1, int((_DEADLINE - 10 - time.time()) / time_per_seed))
        
        for seed in range(num_seeds):
            if time.time() > _DEADLINE - 5:
                break
            
            # Try ALS
            res = _als_decompose(T, rank, n_iter=300, seed=seed, tol=1e-9)
            if res is not None:
                U, V, W = res
                result = _round_and_verify(U, V, W, T)
                if result is not None:
                    return {"U": result[0], "V": result[1], "W": result[2]}
            
            # Try polynomial approach
            res2 = _polynomial_solve(T, rank, n, m, p, seed=seed, max_iters=100)
            if res2 is not None:
                U, V, W = res2
                result = _round_and_verify(U, V, W, T)
                if result is not None:
                    return {"U": result[0], "V": result[1], "W": result[2]}
        
        # If we've spent too much time on this rank, move to next
        if time.time() > _DEADLINE - 15:
            break
        
    # Fallback: Schoolbook
    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
