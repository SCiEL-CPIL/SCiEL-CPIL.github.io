"""Matrix multiplication 3x4x5 - rank-minimal tensor decomposition search."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 5]
n, m, p = DIMENSIONS
N = n * m  # 12
M = m * p  # 20
P = p * n  # 15

def build_tensor():
    """Build the matrix multiplication tensor as a 3D array."""
    T = np.zeros((N, M, P), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def verify_decomposition(U, V, W, T=None):
    """Verify that U, V, W exactly decompose the tensor."""
    if T is None:
        T = build_tensor()
    rank = U.shape[1]
    result = np.einsum('ar,br,cr->abc', U, V, W)
    if not np.allclose(result, T, atol=1e-10):
        return False
    for mat in [U, V, W]:
        for val in mat:
            if not (abs(val * 2 - round(val * 2)) < 1e-10):
                return False
    return True

def schoolbook_decomposition():
    """Standard rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((N, rank), dtype=np.float64)
    V = np.zeros((M, rank), dtype=np.float64)
    W = np.zeros((P, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def snap_to_halfint(X, tol=1e-8):
    """Snap values to nearest half-integer if within tolerance."""
    Y = np.round(X * 2) / 2
    if np.allclose(X, Y, atol=tol):
        return Y
    return None

def try_rank_als(r, T, max_iters=50, timeout=140.0, seed=42):
    """
    Try to find a rank-r decomposition using ALS with full-system solves.
    """
    start_time = time.time()
    rng = np.random.RandomState(seed)
    
    best_error = np.inf
    best_U, best_V, best_W = None, None, None
    
    for init_trial in range(3):
        if time.time() - start_time > timeout - 5:
            break
        
        # Initialize
        if init_trial == 0:
            U = rng.randn(N, r)
            V = rng.randn(M, r)
            W = rng.randn(P, r)
        elif init_trial == 1:
            # Sparse random
            U = np.zeros((N, r))
            V = np.zeros((M, r))
            W = np.zeros((P, r))
            for col in range(r):
                nnz_u = max(1, N // 4)
                nnz_v = max(1, M // 4)
                nnz_w = max(1, P // 4)
                rows_u = rng.choice(N, nnz_u, replace=False)
                rows_v = rng.choice(M, nnz_v, replace=False)
                rows_w = rng.choice(P, nnz_w, replace=False)
                U[rows_u, col] = rng.choice([-1, 1], nnz_u)
                V[rows_v, col] = rng.choice([-1, 1], nnz_v)
                W[rows_w, col] = rng.choice([-1, 1], nnz_w)
        else:
            U = rng.randn(N, r) * 0.3
            V = rng.randn(M, r) * 0.3
            W = rng.randn(P, r) * 0.3
        
        for it in range(max_iters):
            if time.time() - start_time > timeout - 5:
                break
            
            # Update U: for each a, T[a,:,:] = V @ diag(U[a,:]) @ W.T
            # vec(T[a]) = (W ⊗ V) @ U[a,:]
            WV = np.kron(W, V)  # (M*P, r)
            for a in range(N):
                target = T[a, :, :].flatten()
                try:
                    U[a, :] = np.linalg.lstsq(WV, target, rcond=None)[0]
                except:
                    pass
            
            # Update V: for each b, T[:,b,:] = U @ diag(V[b,:]) @ W.T
            # vec(T[:,b,:]) = (W ⊗ U) @ V[b,:]
            WU = np.kron(W, U)  # (N*P, r)
            for b in range(M):
                target = T[:, b, :].flatten()
                try:
                    V[b, :] = np.linalg.lstsq(WU, target, rcond=None)[0]
                except:
                    pass
            
            # Update W: for each c, T[:,:,c] = U @ diag(V[:,r]) ... 
            # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
            # For fixed c: vec(T[:,:,c]) = (V ⊗ U) @ W[c,:]
            VU = np.kron(V, U)  # (N*M, r)
            for c in range(P):
                target = T[:, :, c].flatten()
                try:
                    W[c, :] = np.linalg.lstsq(VU, target, rcond=None)[0]
                except:
                    pass
            
            # Compute error
            result = np.einsum('ar,br,cr->abc', U, V, W)
            error = np.linalg.norm(result - T)
            
            if error < best_error:
                best_error = error
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
                if error < 1e-10:
                    # Snap to half-integers
                    sU = snap_to_halfint(best_U)
                    sV = snap_to_halfint(best_V)
                    sW = snap_to_halfint(best_W)
                    if sU is not None and sV is not None and sW is not None:
                        if verify_decomposition(sU, sV, sW, T):
                            return sU, sV, sW
    
    if best_error < 1e-10:
        sU = snap_to_halfint(best_U)
        sV = snap_to_halfint(best_V)
        sW = snap_to_halfint(best_W)
        if sU is not None and sV is not None and sW is not None:
            if verify_decomposition(sU, sV, sW, T):
                return sU, sV, sW
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    T = build_tensor()
    
    # Schoolbook fallback
    U, V, W = schoolbook_decomposition()
    best_rank = n * m * p
    best_result = (U, V, W)
    
    random.seed(42)
    np.random.seed(42)
    
    # Try ranks from lower bound upward
    lower_bound = max(N, M, P)  # 20
    
    for r in range(lower_bound, best_rank + 1):
        if time.time() > deadline - 15:
            break
        
        found = None
        # Use more trials for smaller ranks
        max_trials = 5 if r <= 25 else 3
        
        for trial in range(max_trials):
            if time.time() > deadline - 15:
                break
            
            result = try_rank_als(r, T, max_iters=40, timeout=deadline - time.time() - 15, seed=42 + trial)
            if result is not None:
                found = result
                break
        
        if found is not None:
            best_rank = r
            best_result = found
            break
    
    U, V, W = best_result
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
