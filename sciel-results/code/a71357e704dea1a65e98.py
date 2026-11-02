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

def polish_decomposition(U, V, W, T, max_iters=100, tol=1e-14):
    """
    High-precision polish pass: alternate least squares with tighter convergence.
    """
    r = U.shape[1]
    best_error = np.linalg.norm(np.einsum('ar,br,cr->abc', U, V, W) - T)
    
    for it in range(max_iters):
        # Update U
        WV = np.kron(W, V)
        for a in range(N):
            target = T[a, :, :].flatten()
            U[a, :] = np.linalg.lstsq(WV, target, rcond=None)[0]
        
        # Update V
        WU = np.kron(W, U)
        for b in range(M):
            target = T[:, b, :].flatten()
            V[b, :] = np.linalg.lstsq(WU, target, rcond=None)[0]
        
        # Update W
        VU = np.kron(V, U)
        for c in range(P):
            target = T[:, :, c].flatten()
            W[c, :] = np.linalg.lstsq(VU, target, rcond=None)[0]
        
        result = np.einsum('ar,br,cr->abc', U, V, W)
        error = np.linalg.norm(result - T)
        
        if error < best_error:
            best_error = error
            if error < tol:
                break
    
    if best_error < 1e-10:
        sU = snap_to_halfint(U, tol=1e-8)
        sV = snap_to_halfint(V, tol=1e-8)
        sW = snap_to_halfint(W, tol=1e-8)
        if sU is not None and sV is not None and sW is not None:
            if verify_decomposition(sU, sV, sW, T):
                return sU, sV, sW
    
    return None

def try_rank_als(r, T, max_iters=50, timeout=140.0, seed=42):
    """
    Try to find a rank-r decomposition using ALS with full-system solves.
    """
    start_time = time.time()
    rng = np.random.RandomState(seed)
    
    best_error = np.inf
    best_U, best_V, best_W = None, None, None
    
    # Generate multiple diverse initializations
    init_candidates = []
    
    # Init 0: Gaussian
    init_candidates.append(('gauss', rng.randn(N, r), rng.randn(M, r), rng.randn(P, r)))
    
    # Init 1: Sparse random
    U_s = np.zeros((N, r))
    V_s = np.zeros((M, r))
    W_s = np.zeros((P, r))
    for col in range(r):
        nnz_u = max(1, N // 4)
        nnz_v = max(1, M // 4)
        nnz_w = max(1, P // 4)
        rows_u = rng.choice(N, nnz_u, replace=False)
        rows_v = rng.choice(M, nnz_v, replace=False)
        rows_w = rng.choice(P, nnz_w, replace=False)
        U_s[rows_u, col] = rng.choice([-1, 1], nnz_u)
        V_s[rows_v, col] = rng.choice([-1, 1], nnz_v)
        W_s[rows_w, col] = rng.choice([-1, 1], nnz_w)
    init_candidates.append(('sparse', U_s, V_s, W_s))
    
    # Init 2: Structured based on tensor support
    # Each rank-1 term should have support matching a (i,j,k) triple
    # Try to assign random triples to each rank component
    U_t = np.zeros((N, r))
    V_t = np.zeros((M, r))
    W_t = np.zeros((P, r))
    triples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    for col in range(r):
        idx = rng.randint(0, len(triples) - 1)
        i, j, k = triples[idx]
        sign = rng.choice([-1, 1])
        U_t[i * m + j, col] = sign
        V_t[j * p + k, col] = sign
        W_t[k * n + i, col] = sign
    init_candidates.append(('structured', U_t, V_t, W_t))
    
    # Init 3: Small Gaussian
    init_candidates.append(('small_gauss', rng.randn(N, r) * 0.3, 
                            rng.randn(M, r) * 0.3, 
                            rng.randn(P, r) * 0.3))
    
    # Init 4: Block-structured (groups of rows)
    U_b = np.zeros((N, r))
    V_b = np.zeros((M, r))
    W_b = np.zeros((P, r))
    for col in range(r):
        # Assign to random row blocks
        i_idx = rng.randint(0, n - 1)
        j_idx = rng.randint(0, m - 1)
        k_idx = rng.randint(0, p - 1)
        sign = rng.choice([-1, 1])
        U_b[i_idx * m + j_idx, col] = sign
        V_b[j_idx * p + k_idx, col] = sign
        W_b[k_idx * n + i_idx, col] = sign
    init_candidates.append(('block', U_b, V_b, W_b))
    
    for init_name, U_init, V_init, W_init in init_candidates:
        if time.time() - start_time > timeout - 5:
            break
        
        U = U_init.copy()
        V = V_init.copy()
        W = W_init.copy()
        
        for it in range(max_iters):
            if time.time() - start_time > timeout - 5:
                break
            
            # Update U
            WV = np.kron(W, V)
            for a in range(N):
                target = T[a, :, :].flatten()
                try:
                    U[a, :] = np.linalg.lstsq(WV, target, rcond=None)[0]
                except:
                    pass
            
            # Update V
            WU = np.kron(W, U)
            for b in range(M):
                target = T[:, b, :].flatten()
                try:
                    V[b, :] = np.linalg.lstsq(WU, target, rcond=None)[0]
                except:
                    pass
            
            # Update W
            VU = np.kron(V, U)
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
                    # Try polish
                    polished = polish_decomposition(best_U, best_V, best_W, T, max_iters=50)
                    if polished is not None:
                        return polished
    
    # Final check
    if best_error < 1e-10 and best_U is not None:
        polished = polish_decomposition(best_U, best_V, best_W, T, max_iters=100)
        if polished is not None:
            return polished
        
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
        # Use more trials for smaller ranks, but limit total time per rank
        remaining = deadline - time.time() - 15
        if remaining < 10:
            break
        
        # Allocate time: smaller ranks get more time
        if r <= 25:
            max_trials = 3
            per_trial_time = min(remaining / max_trials, 30)
        else:
            max_trials = 2
            per_trial_time = min(remaining / max_trials, 20)
        
        for trial in range(max_trials):
            if time.time() > deadline - 15:
                break
            
            result = try_rank_als(r, T, max_iters=30, timeout=per_trial_time, seed=42 + trial * 7)
            if result is not None:
                found = result
                break
        
        if found is not None:
            best_rank = r
            best_result = found
            break
    
    U, V, W = best_result
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
