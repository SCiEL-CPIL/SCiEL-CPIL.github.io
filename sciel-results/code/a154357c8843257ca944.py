"""Matrix multiplication 3,3,7 tensor decomposition."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = 145.0

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor for n x m times m x p."""
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return T

def _als_decompose(T, rank, max_iters=500, tol=1e-10, seed=42, time_limit=20.0):
    """Alternating least squares for tensor decomposition."""
    np.random.seed(seed)
    n, m, p = T.shape
    U = np.random.randn(n, rank) * 0.5
    V = np.random.randn(m, rank) * 0.5
    W = np.random.randn(p, rank) * 0.5
    best_err = float('inf')
    best_UVW = None
    start = time.time()
    
    for it in range(max_iters):
        if time.time() - start > time_limit:
            break
        if _time_left() < 15.0:
            break
            
        # Update U
        for r in range(rank):
            Vr = V[:, r]
            Wr = W[:, r]
            denom = np.sum(Vr**2) * np.sum(Wr**2)
            if denom < 1e-12:
                U[:, r] = np.random.randn(n) * 0.1
                continue
            R = T - np.einsum('ir,jr,kr->ijk', 
                             np.delete(U, r, axis=1), 
                             np.delete(V, r, axis=1), 
                             np.delete(W, r, axis=1)) if rank > 1 else T.copy()
            U[:, r] = np.einsum('ijk,j,k->i', R, Vr, Wr) / denom
        
        # Update V
        for r in range(rank):
            Ur = U[:, r]
            Wr = W[:, r]
            denom = np.sum(Ur**2) * np.sum(Wr**2)
            if denom < 1e-12:
                V[:, r] = np.random.randn(m) * 0.1
                continue
            R = T - np.einsum('ir,jr,kr->ijk', 
                             np.delete(U, r, axis=1), 
                             np.delete(V, r, axis=1), 
                             np.delete(W, r, axis=1)) if rank > 1 else T.copy()
            V[:, r] = np.einsum('ijk,i,k->j', R, Ur, Wr) / denom
        
        # Update W
        for r in range(rank):
            Ur = U[:, r]
            Vr = V[:, r]
            denom = np.sum(Ur**2) * np.sum(Vr**2)
            if denom < 1e-12:
                W[:, r] = np.random.randn(p) * 0.1
                continue
            R = T - np.einsum('ir,jr,kr->ijk', 
                             np.delete(U, r, axis=1), 
                             np.delete(V, r, axis=1), 
                             np.delete(W, r, axis=1)) if rank > 1 else T.copy()
            W[:, r] = np.einsum('ijk,i,j->k', R, Ur, Vr) / denom
        
        if it % 5 == 0:
            recon = np.einsum('ir,jr,kr->ijk', U, V, W)
            err = np.linalg.norm(T - recon) / (np.linalg.norm(T) + 1e-15)
            if err < best_err:
                best_err = err
                best_UVW = (U.copy(), V.copy(), W.copy())
            if err < tol:
                break
    
    if best_UVW is not None:
        return best_UVW
    return U, V, W

def _snap_to_half_integers(U, V, W, T, tol=1e-6):
    """Try to snap to half-integers and verify."""
    U_h = np.round(U * 2) / 2.0
    V_h = np.round(V * 2) / 2.0
    W_h = np.round(W * 2) / 2.0
    recon_h = np.einsum('ir,jr,kr->ijk', U_h, V_h, W_h)
    err_h = np.linalg.norm(T - recon_h) / (np.linalg.norm(T) + 1e-15)
    if err_h < tol:
        return U_h, V_h, W_h, err_h
    return None

def _find_decomposition(n, m, p, ranks_to_try, seeds, time_per_attempt=20.0):
    """Find a low-rank decomposition for the given dimensions."""
    T = _build_tensor(n, m, p)
    for rank in ranks_to_try:
        if _time_left() < 30.0:
            break
        for seed in seeds:
            if _time_left() < 25.0:
                break
            U, V, W = _als_decompose(T, rank, max_iters=500, seed=seed, time_limit=time_per_attempt)
            recon = np.einsum('ir,jr,kr->ijk', U, V, W)
            err = np.linalg.norm(T - recon) / (np.linalg.norm(T) + 1e-15)
            if err < 1e-6:
                snapped = _snap_to_half_integers(U, V, W, T)
                if snapped is not None:
                    U_h, V_h, W_h, _ = snapped
                    return U_h, V_h, W_h, rank
                return U, V, W, rank
    return None

def _schoolbook(n, m, p):
    """Schoolbook decomposition."""
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W, rank

def _build_full_3x3x7(U33, V33, W33, rank33):
    """Build the full 3x3 x 3x7 decomposition from 3x3 x 3x3 sub-decompositions."""
    U31 = np.zeros((9, 9))
    V31 = np.zeros((3, 9))
    W31 = np.zeros((3, 9))
    col = 0
    for i in range(3):
        for j in range(3):
            U31[i*3+j, col] = 1
            V31[j, col] = 1
            W31[i, col] = 1
            col += 1
    
    total_rank = rank33 + rank33 + 9
    U_full = np.zeros((9, total_rank))
    V_full = np.zeros((21, total_rank))
    W_full = np.zeros((21, total_rank))
    
    # Group 1: k=0,1,2
    U_full[:, 0:rank33] = U33
    for j in range(3):
        for k in range(3):
            V_full[j*7+k, 0:rank33] = V33[j*3+k, :]
    for i in range(3):
        for k in range(3):
            W_full[k*3+i, 0:rank33] = W33[k*3+i, :]
    
    # Group 2: k=3,4,5
    U_full[:, rank33:2*rank33] = U33
    for j in range(3):
        for k in range(3):
            V_full[j*7+(k+3), rank33:2*rank33] = V33[j*3+k, :]
    for i in range(3):
        for k in range(3):
            W_full[(k+3)*3+i, rank33:2*rank33] = W33[k*3+i, :]
    
    # Group 3: k=6
    U_full[:, 2*rank33:2*rank33+9] = U31
    for j in range(3):
        V_full[j*7+6, 2*rank33:2*rank33+9] = V31[j, :]
    for i in range(3):
        W_full[6*3+i, 2*rank33:2*rank33+9] = W31[i, :]
    
    return U_full, V_full, W_full

def _verify_full(U, V, W, n, m, p):
    """Verify the full decomposition."""
    T = _build_tensor(n, m, p)
    recon = np.einsum('ir,jr,kr->ijk', U, V, W)
    return np.linalg.norm(T - recon) / (np.linalg.norm(T) + 1e-15)

def construct():
    n, m, p = DIMENSIONS
    
    # Try to find a good 3x3 x 3x3 decomposition
    # Ranks to try: 19, 20, 21, 22, 23
    # Laderman is 23, known to be integer.
    U33, V33, W33, rank33 = None, None, None, None
    
    if _time_left() > 60.0:
        result = _find_decomposition(3, 3, 3, [19, 20, 21, 22, 23], [42, 123, 999, 7, 13, 55, 77], time_per_attempt=15.0)
        if result is not None:
            U33, V33, W33, rank33 = result
    
    if U33 is None:
        # Fall back to schoolbook for 3x3x3
        U33, V33, W33, rank33 = _schoolbook(3, 3, 3)
    
    # Build the full 3x3 x 3x7 decomposition
    if _time_left() > 20.0:
        U_full, V_full, W_full = _build_full_3x3x7(U33, V33, W33, rank33)
        err = _verify_full(U_full, V_full, W_full, n, m, p)
        if err < 1e-6:
            return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
    
    # Final fallback: schoolbook for the full problem
    U, V, W, rank = _schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
