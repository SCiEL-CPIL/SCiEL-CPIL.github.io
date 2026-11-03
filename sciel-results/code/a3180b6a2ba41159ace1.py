import time
import numpy as np
import scipy.optimize

_START_TIME = time.time()
_DEADLINE = 140  # 10s safety margin before 150s kill

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

def _verify(U, V, W, n, m, p, tol=1e-8):
    """Check if U,V,W form a valid decomposition with integer/half-integer entries."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    T_target = _build_target_tensor(n, m, p)
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    
    if np.max(np.abs(T_recon - T_target)) > tol:
        return False
    return True

def _round_half(x):
    """Round to nearest half-integer."""
    return np.round(x * 2) / 2

def _coordinate_descent(T, U, V, W, n_iter=100, tol=1e-10):
    """
    Perform coordinate descent with exact linear solves and rounding.
    T: target tensor
    U, V, W: initial continuous solutions
    """
    A, B, C = T.shape
    R = U.shape[1]
    
    U_c = U.copy()
    V_c = V.copy()
    W_c = W.copy()
    
    # Precompute target slices for linear solves
    # For W_c (c-th slice of W):
    # We want to find W_c such that sum_{a,b} (sum_r U_{a,r} V_{b,r} W_{c,r} - T_{a,b,c})^2 is minimized.
    # This is a linear least squares problem for W_c.
    # Variables: W_{c,1}, ..., W_{c,R}
    # Equations: For each (a,b), sum_r (U_{a,r} V_{b,r}) W_{c,r} = T_{a,b,c}
    # There are A*B equations and R unknowns.
    
    # Construct matrix M_W where row (a,b) is [U_{a,1}V_{b,1}, ..., U_{a,R}V_{b,R}]
    # This matrix is the same for all c? No, U and V change.
    
    for it in range(n_iter):
        if time.time() > _DEADLINE - 3:
            break
            
        # 1. Update W
        # Build M_W of shape (A*B, R)
        M_W = np.outer(U_c.flatten(), V_c.flatten()).reshape(A, B, R).reshape(A*B, R)
        # Target vector for each c is T[:, :, c].flatten()
        for c in range(C):
            target = T[:, :, c].flatten()
            # Solve M_W @ w_c = target
            w_c, _, _, _ = np.linalg.lstsq(M_W, target, rcond=None)
            W_c[c, :] = w_c
        
        # Round W
        W_rounded = _round_half(W_c)
        
        # 2. Update V
        # Build M_V of shape (A*C, R)
        # Equation: sum_r U_{a,r} W_{c,r} V_{b,r} = T_{a,b,c}
        # For fixed b, unknowns are V_{b,1}, ..., V_{b,R}
        # Coefficients: U_{a,r} W_{c,r} for each (a,c)
        M_V = np.outer(U_c.flatten(), W_rounded.flatten()).reshape(A, C, R).reshape(A*C, R)
        for b in range(B):
            target = T[:, b, :].flatten()
            v_b, _, _, _ = np.linalg.lstsq(M_V, target, rcond=None)
            V_c[b, :] = v_b
            
        V_rounded = _round_half(V_c)
        
        # 3. Update U
        # Build M_U of shape (B*C, R)
        # Equation: sum_r V_{b,r} W_{c,r} U_{a,r} = T_{a,b,c}
        # For fixed a, unknowns are U_{a,1}, ..., U_{a,R}
        # Coefficients: V_{b,r} W_{c,r} for each (b,c)
        M_U = np.outer(V_rounded.flatten(), W_rounded.flatten()).reshape(B, C, R).reshape(B*C, R)
        for a in range(A):
            target = T[a, :, :].flatten()
            u_a, _, _, _ = np.linalg.lstsq(M_U, target, rcond=None)
            U_c[a, :] = u_a
            
        U_rounded = _round_half(U_c)
        
        # Check convergence / exactness
        T_recon = np.einsum('ar,br,cr->abc', U_rounded, V_rounded, W_rounded)
        err = np.max(np.abs(T_recon - T))
        
        if err < tol:
            return U_rounded, V_rounded, W_rounded, True
        
        # Optional: If error is not decreasing, break?
        # For now, just keep going.
        
    return U_rounded, V_rounded, W_rounded, False

def _als_decompose(T, R, n_iter=1000, seed=42, init_type='random', tol=1e-7):
    """
    Alternating Least Squares for tensor decomposition.
    """
    A, B, C = T.shape
    rng = np.random.RandomState(seed)
    
    if init_type == 'random':
        U = rng.randn(A, R) * 0.1
        V = rng.randn(B, R) * 0.1
        W = rng.randn(C, R) * 0.1
    elif init_type == 'sparse_int':
        U = rng.randint(-1, 2, size=(A, R)).astype(float)
        V = rng.randint(-1, 2, size=(B, R)).astype(float)
        W = rng.randint(-1, 2, size=(C, R)).astype(float)
        for r in range(R):
            nu = np.linalg.norm(U[:, r])
            nv = np.linalg.norm(V[:, r])
            nw = np.linalg.norm(W[:, r])
            if nu > 1e-9 and nv > 1e-9 and nw > 1e-9:
                U[:, r] /= nu
                V[:, r] /= nv
                W[:, r] /= nw
    elif init_type == 'schoolbook_trunc':
        n, m, p = 2, 3, 4
        U = np.zeros((A, R))
        V = np.zeros((B, R))
        W = np.zeros((C, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if col < R:
                        U[i*m+j, col] = 1
                        V[j*p+k, col] = 1
                        W[k*n+i, col] = 1
                        col += 1
        for r in range(col, R):
            U[:, r] = rng.randn(A) * 0.5
            V[:, r] = rng.randn(B) * 0.5
            W[:, r] = rng.randn(C) * 0.5
            
    T_a = [T[a, :, :].flatten() for a in range(A)]
    T_b = [T[:, b, :].flatten() for b in range(B)]
    T_c = [T[:, :, c].flatten() for c in range(C)]
    
    for it in range(n_iter):
        if time.time() > _DEADLINE - 3:
            break
        
        A_U = np.outer(V.flatten(), W.flatten()).reshape(B * C, R)
        for a in range(A):
            U[a, :], _, _, _ = np.linalg.lstsq(A_U, T_a[a], rcond=None)
        
        A_V = np.outer(U.flatten(), W.flatten()).reshape(A * C, R)
        for b in range(B):
            V[b, :], _, _, _ = np.linalg.lstsq(A_V, T_b[b], rcond=None)
        
        A_W = np.outer(U.flatten(), V.flatten()).reshape(A * B, R)
        for c in range(C):
            W[c, :], _, _, _ = np.linalg.lstsq(A_W, T_c[c], rcond=None)
        
        if it % 10 == 0:
            T_recon = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.max(np.abs(T_recon - T))
            if err < tol:
                break
    
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    if np.max(np.abs(T_recon - T)) < 1e-6:
        return U, V, W
    return None

def construct():
    n, m, p = 2, 3, 4
    T = _build_target_tensor(n, m, p)
    
    # Try ranks from 22 up to 23
    for rank in [22, 23]:
        if time.time() > _DEADLINE - 10:
            break
        
        # Try multiple seeds and initializations
        num_seeds = 50 if rank == 22 else 30
        init_types = ['random', 'sparse_int', 'schoolbook_trunc']
        
        for seed in range(num_seeds):
            if time.time() > _DEADLINE - 5:
                break
            
            for init_type in init_types:
                if time.time() > _DEADLINE - 3:
                    break
                
                # Get continuous solution
                res = _als_decompose(T, rank, n_iter=200, seed=seed, init_type=init_type, tol=1e-8)
                if res is not None:
                    U, V, W = res
                    
                    # Try coordinate descent with rounding
                    U_r, V_r, W_r, success = _coordinate_descent(T, U, V, W, n_iter=50)
                    if success:
                        return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                    
                    # If CD failed, try direct rounding of ALS result
                    U_h = _round_half(U)
                    V_h = _round_half(V)
                    W_h = _round_half(W)
                    if _verify(U_h, V_h, W_h, n, m, p):
                        return {"U": U_h.tolist(), "V": V_h.tolist(), "W": W_h.tolist()}
                    
                    # Try integer rounding
                    U_i = np.round(U)
                    V_i = np.round(V)
                    W_i = np.round(W)
                    if _verify(U_i, V_i, W_i, n, m, p):
                        return {"U": U_i.tolist(), "V": V_i.tolist(), "W": W_i.tolist()}
        
    # Fallback: Schoolbook
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
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
