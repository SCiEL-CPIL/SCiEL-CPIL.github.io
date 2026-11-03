"""Rank-minimal exact tensor decomposition for 3x4x4 matrix multiplication."""
import numpy as np
import time
import random

DIMENSIONS = [3, 4, 4]

def _build_schoolbook(n, m, p):
    """Build the schoolbook decomposition of rank n*m*p."""
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

def _check_exact(U, V, W, n, m, p, tol=1e-9):
    """Check if the decomposition is exact for all (i,j,k)."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _max_error(U, V, W, n, m, p):
    """Compute the maximum absolute error over all (i,j,k)."""
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                max_err = max(max_err, abs(val - 1.0))
    return max_err

def _snap_to_half_integer(arr, tol=1e-6):
    """Snap array entries to nearest half-integer if close enough."""
    snapped = np.round(arr * 2) / 2
    if np.max(np.abs(arr - snapped)) < tol:
        return snapped, True
    return arr, False

def _try_snap_and_check(U, V, W, n, m, p):
    """Try to snap to half-integers and check exactness."""
    for tol in [1e-5, 1e-4, 1e-3, 1e-2]:
        U_snapped, u_ok = _snap_to_half_integer(U, tol=tol)
        V_snapped, v_ok = _snap_to_half_integer(V, tol=tol)
        W_snapped, w_ok = _snap_to_half_integer(W, tol=tol)
        if u_ok and v_ok and w_ok:
            if _check_exact(U_snapped, V_snapped, W_snapped, n, m, p, tol=1e-9):
                return U_snapped, V_snapped, W_snapped
    return None

def _als_iterate(U, V, W, n, m, p, max_iter=500, tol=1e-14):
    """Alternating least squares for tensor decomposition."""
    rank = U.shape[1]
    for it in range(max_iter):
        U_old = U.copy()
        V_old = V.copy()
        W_old = W.copy()
        
        # Update U
        for idx in range(n * m):
            i, j = divmod(idx, m)
            A = np.zeros((p, rank))
            for k in range(p):
                A[k, :] = V[j * p + k, :] * W[k * n + i, :]
            b = np.ones(p)
            try:
                sol = np.linalg.lstsq(A, b, rcond=None)[0]
                U[idx, :] = sol
            except:
                pass
        
        # Update V
        for idx in range(m * p):
            j, k = divmod(idx, p)
            A = np.zeros((n, rank))
            for i in range(n):
                A[i, :] = U[i * m + j, :] * W[k * n + i, :]
            b = np.ones(n)
            try:
                sol = np.linalg.lstsq(A, b, rcond=None)[0]
                V[idx, :] = sol
            except:
                pass
        
        # Update W
        for idx in range(p * n):
            k, i = divmod(idx, n)
            A = np.zeros((m, rank))
            for j in range(m):
                A[j, :] = U[i * m + j, :] * V[j * p + k, :]
            b = np.ones(m)
            try:
                sol = np.linalg.lstsq(A, b, rcond=None)[0]
                W[idx, :] = sol
            except:
                pass
        
        # Check convergence
        diff = (np.max(np.abs(U - U_old)) + 
                np.max(np.abs(V - V_old)) + 
                np.max(np.abs(W - W_old)))
        if diff < tol:
            break
    
    return U, V, W

def _structured_init_strassen_style(n, m, p, rank, seed=0):
    """
    Strassen-inspired structured initialization.
    """
    rng = np.random.RandomState(seed)
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # For each (i,k), use terms that combine j values
    for i in range(n):
        for k in range(p):
            # Term 1: sum over first pair
            if col < rank:
                for j in [0, 1]:
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
            
            # Term 2: sum over second pair
            if col < rank:
                for j in [2, 3]:
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
            
            # Term 3: difference pattern
            if col < rank:
                U[i*m+0, col] = 1.0
                U[i*m+1, col] = -1.0
                V[0*p+k, col] = 1.0
                V[1*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
            
            # Term 4: another difference
            if col < rank:
                U[i*m+2, col] = 1.0
                U[i*m+3, col] = -1.0
                V[2*p+k, col] = 1.0
                V[3*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    
    # Fill remaining with random
    while col < rank:
        U[:, col] = rng.randn(n*m) * 0.3
        V[:, col] = rng.randn(m*p) * 0.3
        W[:, col] = rng.randn(p*n) * 0.3
        col += 1
    
    return U, V, W

def _structured_init_kronecker(n, m, p, rank, seed=0):
    """
    Kronecker-product inspired initialization.
    """
    rng = np.random.RandomState(seed)
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    for i in range(n):
        for k in range(p):
            for j in range(m):
                if col < rank:
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
    
    # Fill remaining
    while col < rank:
        U[:, col] = rng.randn(n*m) * 0.3
        V[:, col] = rng.randn(m*p) * 0.3
        W[:, col] = rng.randn(p*n) * 0.3
        col += 1
    
    return U, V, W

def _structured_init_mixed(n, m, p, rank, seed=0):
    """
    Mixed structured initialization.
    """
    rng = np.random.RandomState(seed)
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    n_struct = min(rank * 2 // 3, 48)
    
    for s in range(n_struct):
        i = rng.randint(0, n-1)
        k = rng.randint(0, p-1)
        
        n_j = rng.randint(1, 4)
        j_vals = rng.choice(4, size=n_j, replace=False)
        
        U[i*m + j_vals[0], col] = 1.0
        for j in j_vals[1:]:
            U[i*m + j, col] = rng.choice([1.0, -1.0])
        
        for j in j_vals:
            V[j*p + k, col] = rng.choice([1.0, -1.0])
        
        W[k*n + i, col] = rng.choice([1.0, -1.0])
        col += 1
    
    while col < rank:
        U[:, col] = rng.randn(n*m) * 0.3
        V[:, col] = rng.randn(m*p) * 0.3
        W[:, col] = rng.randn(p*n) * 0.3
        col += 1
    
    return U, V, W

def _try_rank_als(n, m, p, rank, max_restarts=20, seed=None, deadline=None):
    """Try to find a rank-r exact decomposition using ALS."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    best_err = float('inf')
    best_U, best_V, best_W = None, None, None
    
    for restart in range(max_restarts):
        if deadline is not None and time.time() > deadline - 5:
            break
        
        strat = restart % 4
        if strat == 0:
            scale = 0.3 + random.random() * 0.4
            U = np.random.randn(n * m, rank) * scale
            V = np.random.randn(m * p, rank) * scale
            W = np.random.randn(p * n, rank) * scale
        elif strat == 1:
            U, V, W = _structured_init_strassen_style(n, m, p, rank, seed=seed + restart * 7)
        elif strat == 2:
            U, V, W = _structured_init_kronecker(n, m, p, rank, seed=seed + restart * 11)
        else:
            U, V, W = _structured_init_mixed(n, m, p, rank, seed=seed + restart * 13)
        
        U, V, W = _als_iterate(U, V, W, n, m, p, max_iter=300)
        
        max_err = _max_error(U, V, W, n, m, p)
        
        if max_err < best_err:
            best_err = max_err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if max_err < 1e-7:
            result = _try_snap_and_check(U, V, W, n, m, p)
            if result is not None:
                return result
    
    return None

def _try_rank_gradient(n, m, p, rank, max_restarts=10, seed=None, deadline=None):
    """Try to find a rank-r exact decomposition using gradient descent."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    dim_u, dim_v, dim_w = n*m, m*p, p*n
    
    T = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    for restart in range(max_restarts):
        if deadline is not None and time.time() > deadline - 5:
            break
        
        strat = restart % 3
        if strat == 0:
            U, V, W = _structured_init_strassen_style(n, m, p, rank, seed=seed + restart * 17)
        elif strat == 1:
            U, V, W = _structured_init_kronecker(n, m, p, rank, seed=seed + restart * 19)
        else:
            U, V, W = _structured_init_mixed(n, m, p, rank, seed=seed + restart * 23)
        
        lr = 0.01
        prev_loss = float('inf')
        
        for step in range(1500):
            T_rec = np.zeros((dim_u, dim_v, dim_w))
            for r in range(rank):
                T_rec += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
            
            err = T_rec - T
            loss = np.sum(err**2)
            
            if loss < 1e-12:
                break
            if loss > 1e8:
                break
            
            dU = 2 * np.einsum('abc,br,cr->ar', err, V, W)
            dV = 2 * np.einsum('abc,ar,cr->br', err, U, W)
            dW = 2 * np.einsum('abc,ar,br->cr', err, U, V)
            
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            if loss < prev_loss * 0.98:
                lr *= 1.05
            elif loss > prev_loss * 1.02:
                lr *= 0.95
            prev_loss = loss
        
        max_err = _max_error(U, V, W, n, m, p)
        
        if max_err < 1e-7:
            result = _try_snap_and_check(U, V, W, n, m, p)
            if result is not None:
                return result
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    U_sb, V_sb, W_sb = _build_schoolbook(n, m, p)
    
    best_result = None
    best_rank = schoolbook_rank
    
    # Phase 1: ALS with structured initializations
    for rank in range(42, 34, -1):
        if time.time() > deadline - 20:
            break
        
        result = _try_rank_als(n, m, p, rank, max_restarts=15, seed=rank * 1000, deadline=deadline)
        if result is not None:
            best_result = result
            best_rank = rank
            break
    
    # Phase 2: Gradient descent
    if best_result is None:
        for rank in range(40, 34, -1):
            if time.time() > deadline - 15:
                break
            
            result = _try_rank_gradient(n, m, p, rank, max_restarts=8, seed=rank * 2000, deadline=deadline)
            if result is not None:
                best_result = result
                best_rank = rank
                break
    
    if best_result is None:
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    U, V, W = best_result
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
