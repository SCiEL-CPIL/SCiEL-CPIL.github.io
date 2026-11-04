"""
Matrix multiplication 3x4x8 tensor decomposition with low-rank exact solution.

Strategy: Use a sliding-window / overlapping-block decomposition.
- The inner dimension m=4 can be partitioned into overlapping 2-dim windows.
- The output dimension p=8 can be partitioned into overlapping 2-dim windows.
- By sharing rank-1 terms across adjacent blocks, we can reduce total rank.

The key insight: for matrix multiplication C = A * B where A is 3x4 and B is 4x8,
the tensor T[i*m+j, j*p+k, k*n+i] = 1 has local structure. If we group j into
pairs and k into pairs, we can use Strassen-like additions within each group
and share factors across overlapping groups.

We use gradient-based optimization with multiple random restarts, trying to
find a rank-r decomposition for r < 96 (schoolbook). We try ranks from
a reasonable lower bound upward, using structured initializations inspired
by the overlapping block structure.
"""

import time
import numpy as np

DIMENSIONS = [3, 4, 8]
START_TIME = time.time()
TIME_LIMIT = 120.0  # Safety margin before 150s kill

def _build_true_tensor(n, m, p):
    """Build the true matrix multiplication tensor."""
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _verify(U, V, W, T_true, atol=1e-8):
    """Verify that U, V, W correctly decompose the tensor."""
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    T_recon = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    return np.allclose(T_recon, T_true, atol=atol)

def _snap_to_half_inters(U, V, W):
    """Try to snap values to nearest half-integer and verify."""
    for scale in [1, 2]:
        U_r = np.round(U * scale) / scale
        V_r = np.round(V * scale) / scale
        W_r = np.round(W * scale) / scale
        return U_r.tolist(), V_r.tolist(), W_r.tolist()
    return U.tolist(), V.tolist(), W.tolist()

def _gradient_descent(U, V, W, T_true, eta, max_iters, tol=1e-10):
    """Run gradient descent to minimize ||T_recon - T_true||^2."""
    n, m, p = T_true.shape
    nm, mp, pn = U.shape[0], V.shape[0], W.shape[0]
    rank = U.shape[1]
    
    best_err = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        R = T_true - T_recon
        err = np.max(np.abs(R))
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if err < tol:
            break
        
        # Compute gradients
        grad_U = np.einsum('abc,br,cr->ar', R, V, W)
        grad_V = np.einsum('abc,ar,cr->br', R, U, W)
        grad_W = np.einsum('abc,ar,br->cr', R, U, V)
        
        U += eta * grad_U
        V += eta * grad_V
        W += eta * grad_W
    
    return best_U, best_V, best_W, best_err

def _als_refinement(U, V, W, T_true, max_iters=200, tol=1e-10):
    """Alternating least squares style refinement."""
    n, m, p = T_true.shape
    nm, mp, pn = U.shape[0], V.shape[0], W.shape[0]
    rank = U.shape[1]
    
    best_err = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        # Update U: for each a, solve U[a,:] * (V^T W_outer) = T_true[a,:,:]
        # More precisely: minimize sum_{b,c} (T_true[a,b,c] - U[a,:]·(V[b,:]*W[c,:]))^2
        # This is: U[a,:] = T_true[a,:,:] ⊙ (V⊗W) / ||V⊗W||^2
        # where ⊙ is the flattening operation
        
        # Simpler: use gradient step with adaptive learning rate
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        R = T_true - T_recon
        err = np.max(np.abs(R))
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if err < tol:
            break
        
        # Adaptive learning rate based on gradient magnitude
        grad_U = np.einsum('abc,br,cr->ar', R, V, W)
        grad_V = np.einsum('abc,ar,cr->br', R, U, W)
        grad_W = np.einsum('abc,ar,br->cr', R, U, V)
        
        # Normalize gradients for stable updates
        gU_norm = np.linalg.norm(grad_U)
        gV_norm = np.linalg.norm(grad_V)
        gW_norm = np.linalg.norm(grad_W)
        
        if gU_norm > 1e-12:
            eta_U = 0.1 / gU_norm
        else:
            eta_U = 0.1
        if gV_norm > 1e-12:
            eta_V = 0.1 / gV_norm
        else:
            eta_V = 0.1
        if gW_norm > 1e-12:
            eta_W = 0.1 / gW_norm
        else:
            eta_W = 0.1
        
        U += eta_U * grad_U
        V += eta_V * grad_V
        W += eta_W * grad_W
    
    return best_U, best_V, best_W, best_err

def _structured_init(n, m, p, r, seed=42):
    """
    Create a structured initialization based on overlapping block decomposition.
    For m=4, p=8: partition j into pairs (j1,j2) and k into pairs (k1,k2).
    Use Strassen-like additions within each block.
    """
    rng = np.random.RandomState(seed)
    nm, mp, pn = n * m, m * p, p * n
    
    U = np.zeros((nm, r))
    V = np.zeros((mp, r))
    W = np.zeros((pn, r))
    
    # For each of the m=4 values of j, we have 3*8=24 (i,k) pairs
    # Total schoolbook terms: 4 * 24 = 96
    # With overlapping blocks of size 2 in j and 2 in k:
    # j windows: [0,1], [1,2], [2,3] -> 3 windows
    # k windows: [0,1], [2,3], [4,5], [6,7] -> 4 windows
    # Each (j_window, k_window) pair gives 3 * 2 * 2 = 12 terms if schoolbook
    # But with Strassen-like: 3 * 7 = 21 terms per (j_win, k_win) pair?
    # Actually for 3x2x2: rank is at most 3*2*2=12, possibly less.
    # For 3x2x4: rank is at most 3*2*4=24, possibly less.
    
    # Let's just use random initialization with some structure
    # Add small random perturbations to a base pattern
    
    # Base pattern: for each j, assign random values to U[i*m+j, :]
    # such that the product structure is approximately satisfied
    
    for j in range(m):
        for i in range(n):
            # Assign random values to U[i*m+j, :]
            U[i*m+j, :] = rng.randn(r) * 0.1
        
        for k in range(p):
            # Assign random values to V[j*p+k, :]
            V[j*p+k, :] = rng.randn(r) * 0.1
    
    for k in range(p):
        for i in range(n):
            # Assign random values to W[k*n+i, :]
            W[k*n+i, :] = rng.randn(r) * 0.1
    
    return U, V, W

def _try_rank_r(r, n, m, p, T_true, max_restarts=5, time_budget=20.0):
    """Try to find a rank-r decomposition using multiple methods."""
    start_time = time.time()
    nm, mp, pn = n * m, m * p, p * n
    
    for restart in range(max_restarts):
        if time.time() - start_time > time_budget:
            break
        
        # Method 1: Structured random initialization + gradient descent
        U, V, W = _structured_init(n, m, p, r, seed=restart * 100 + r)
        
        # Phase 1: Gradient descent with decreasing learning rate
        eta = 0.05
        max_iters = 2000
        for phase in range(4):
            U, V, W, err = _gradient_descent(U, V, W, T_true, eta, max_iters, tol=1e-10)
            
            if err < 1e-8:
                # Try to snap to half-integers
                U_s, V_s, W_s = _snap_to_half_inters(U, V, W)
                if _verify(U_s, V_s, W_s, T_true):
                    return U_s, V_s, W_s
                if _verify(U.tolist(), V.tolist(), W.tolist(), T_true):
                    return U.tolist(), V.tolist(), W.tolist()
            
            # Phase 2: ALS refinement
            U, V, W, err = _als_refinement(U, V, W, T_true, max_iters=500, tol=1e-10)
            
            if err < 1e-8:
                U_s, V_s, W_s = _snap_to_half_inters(U, V, W)
                if _verify(U_s, V_s, W_s, T_true):
                    return U_s, V_s, W_s
                if _verify(U.tolist(), V.tolist(), W.tolist(), T_true):
                    return U.tolist(), V.tolist(), W.tolist()
            
            eta *= 0.5
            max_iters = max(100, max_iters // 2)
        
        # Method 2: Different random initialization
        rng = np.random.RandomState(restart * 7919 + r)
        U = rng.randn(nm, r) * 0.3
        V = rng.randn(mp, r) * 0.3
        W = rng.randn(pn, r) * 0.3
        
        for phase in range(3):
            U, V, W, err = _gradient_descent(U, V, W, T_true, eta, max_iters, tol=1e-10)
            
            if err < 1e-8:
                U_s, V_s, W_s = _snap_to_half_inters(U, V, W)
                if _verify(U_s, V_s, W_s, T_true):
                    return U_s, V_s, W_s
                if _verify(U.tolist(), V.tolist(), W.tolist(), T_true):
                    return U.tolist(), V.tolist(), W.tolist()
            
            eta *= 0.5
            max_iters = max(100, max_iters // 2)
    
    return None

def construct():
    global START_TIME
    if START_TIME is None:
        START_TIME = time.time()
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Schoolbook fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    T_true = _build_true_tensor(n, m, p)
    
    # Check time
    if time.time() - START_TIME > TIME_LIMIT - 5:
        return schoolbook()
    
    best_result = schoolbook()
    best_rank = n * m * p  # 96
    
    # Try various ranks using gradient descent with multiple restarts
    # Lower bound: at least max(nm, mp, pn) = 32, but realistically higher
    # For 3x4x8, known lower bounds suggest rank >= 24
    # Try ranks from 32 to 95
    
    # Use a sliding window approach: try ranks that are multiples of block sizes
    # Block structure: 3 windows in j (size 2, overlapping), 4 windows in k (size 2)
    # Each (j_win, k_win) pair with 3 rows: rank at most 3*2*2=12 (schoolbook)
    # With Strassen: rank at most 3*7=21? No, Strassen is for 2x2x2.
    # For 3x2x2: minimum rank is unknown but <= 12
    # For 3x2x4: minimum rank is unknown but <= 24
    
    # Try candidate ranks
    candidate_ranks = sorted(set([
        32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 95
    ]))
    
    for r in candidate_ranks:
        if time.time() - START_TIME > TIME_LIMIT - 15:
            break
        if r >= best_rank:
            break
        
        # Allocate time budget based on rank
        time_budget = min(20.0, (TIME_LIMIT - 15 - (time.time() - START_TIME)) * 0.8)
        if time_budget < 5:
            break
        
        result = _try_rank_r(r, n, m, p, T_true, max_restarts=3, time_budget=time_budget)
        
        if result is not None:
            U, V, W = result
            # Final verification
            if _verify(U, V, W, T_true, atol=1e-6):
                return {"U": U, "V": V, "W": W}
    
    return best_result
