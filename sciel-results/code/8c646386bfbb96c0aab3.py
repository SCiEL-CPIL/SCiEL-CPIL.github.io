"""Matrix multiplication 3x5x7: search for low-rank exact tensor decomposition."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 5, 7]
n, m, p = DIMENSIONS
N = n * m  # 15
M = m * p  # 35
P = p * n  # 21
RANK_SCHOOLBOOK = n * m * p  # 105

# Deadline guard
T_START = time.time()
DEADLINE = 120.0  # seconds total budget; leave margin


def _elapsed():
    return time.time() - T_START


def _time_left():
    return DEADLINE - _elapsed() - 5.0  # 5s safety margin


def _build_matrices_from_factors(U, V, W):
    """
    U: (N, r), V: (M, r), W: (P, r)
    Returns the 3-tensor T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
    where i in [0,n), j in [0,m), k in [0,p)
    But our indexing: U row index = i*m+j, V row = j*p+k, W row = k*n+i
    So T[i*m+j, j*p+k, k*n+i] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    The matrix multiplication tensor is:
    T[i*m+j, j*p+k, k*n+i] = 1 if (i,j,k) valid else 0
    Actually the tensor for matmul C = A*B where A is n x m, B is m x p, C is n x p:
    C[i,k] = sum_j A[i,j] * B[j,k]
    The tensor T has indices (i, j, k) with T[i,j,k] = 1 for all valid (i,j,k).
    In the flattened format:
    U index: i*m + j  (0..n*m-1)
    V index: j*p + k  (0..m*p-1)
    W index: k*n + i  (0..p*n-1)
    T[U_idx, V_idx, W_idx] = 1 if the triple (i,j,k) is consistent.
    """
    r = U.shape[1]
    # Build the target tensor in a sparse way
    # T is a 3D array of shape (N, M, P)
    # T[i*m+j, j*p+k, k*n+i] = 1
    T_target = np.zeros((N, M, P), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i * m + j, j * p + k, k * n + i] = 1
    
    # Compute T_decomposed = sum_r U[:,r] * V[:,r] * W[:,r]
    # This is an outer product sum
    # T_decomposed[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # Efficient: T_decomposed = np.einsum('ar,br,cr->abc', U, V, W)
    # But this might be large. N=15, M=35, P=21, r up to ~105
    # 15*35*21 = 11025 entries, r up to 105 -> 1.1M ops per einsum, fine.
    T_decomposed = np.einsum('ar,br,cr->abc', U, V, W)
    return T_target, T_decomposed


def _check_exact(U, V, W, tol=1e-9):
    """Check if the decomposition exactly reproduces the matmul tensor."""
    T_target, T_decomp = _build_matrices_from_factors(U, V, W)
    # The matmul tensor has 1s at specific positions and 0 elsewhere
    # Check that T_decomp matches T_target
    diff = np.abs(T_decomp - T_target)
    return np.max(diff) < tol


def _snap_to_half_ints(A, tol=1e-6):
    """Snap values close to half-integers to exact half-integers."""
    B = np.array(A, dtype=float)
    B2 = 2 * B
    B2_rounded = np.round(B2)
    if np.max(np.abs(B2 - B2_rounded)) < tol:
        return B2_rounded / 2.0
    return B


def _verify_half_int(U, V, W, tol=1e-6):
    """Verify that all entries are half-integers and the decomposition is exact."""
    for A in [U, V, W]:
        A2 = 2 * A
        if np.max(np.abs(A2 - np.round(A2))) > tol:
            return False
    return _check_exact(U, V, W)


def _random_init(r, scale=1.0):
    """Random initialization of factors."""
    U = np.random.randn(N, r) * scale
    V = np.random.randn(M, r) * scale
    W = np.random.randn(P, r) * scale
    return U, V, W


def _gradient_step(U, V, W, T_target, lr=0.01):
    """
    One gradient descent step on the loss:
    L = sum_{a,b,c} (sum_r U[a,r]*V[b,r]*W[c,r] - T_target[a,b,c])^2
    Gradient wrt U[a,r] = 2 * sum_{b,c} (T_decomp[a,b,c] - T_target[a,b,c]) * V[b,r] * W[c,r]
    Similarly for V and W.
    """
    r = U.shape[1]
    # Compute T_decomp
    T_decomp = np.einsum('ar,br,cr->abc', U, V, W)
    E = T_decomp - T_target  # error tensor
    
    # grad_U[a,r] = 2 * sum_{b,c} E[a,b,c] * V[b,r] * W[c,r]
    # = 2 * einsum('abc,br,cr->ar', E, V, W)
    grad_U = 2 * np.einsum('abc,br,cr->ar', E, V, W)
    grad_V = 2 * np.einsum('abc,ar,cr->br', E, U, W)
    grad_W = 2 * np.einsum('abc,ar,br->cr', E, U, V)
    
    U_new = U - lr * grad_U
    V_new = V - lr * grad_V
    W_new = W - lr * grad_W
    return U_new, V_new, W_new, np.sum(E**2)


def _try_rank(r, max_iters=5000, lr_init=0.001, restarts=3, seed=42):
    """Try to find a decomposition of rank r."""
    T_target, _ = _build_matrices_from_factors(
        np.ones((N, 1)), np.ones((M, 1)), np.ones((P, 1))
    )
    # Actually build T_target properly
    T_target = np.zeros((N, M, P), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i * m + j, j * p + k, k * n + i] = 1.0
    
    best_loss = float('inf')
    best_U, best_V, best_W = None, None, None
    
    for restart in range(restarts):
        if _time_left() < 10:
            break
        rng = np.random.RandomState(seed + restart * 1000)
        U = rng.randn(N, r) * 0.1
        V = rng.randn(M, r) * 0.1
        W = rng.randn(P, r) * 0.1
        
        lr = lr_init
        for it in range(max_iters):
            if _time_left() < 10:
                break
            U, V, W, loss = _gradient_step(U, V, W, T_target, lr)
            
            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            # Adaptive learning rate
            if it % 100 == 0 and it > 0:
                # Reduce lr if loss is decreasing slowly
                lr *= 0.95
            
            if loss < 1e-10:
                break
        
        if best_loss < 1e-8:
            break
    
    if best_U is not None and best_loss < 1e-6:
        # Try to snap to half-integers
        U_s = _snap_to_half_ints(best_U)
        V_s = _snap_to_half_ints(best_V)
        W_s = _snap_to_half_ints(best_W)
        if _verify_half_int(U_s, V_s, W_s):
            return U_s, V_s, W_s, r
        # Check if original is close enough
        if _check_exact(best_U, best_V, best_W):
            # Try harder to snap
            U_s2 = _snap_to_half_ints(best_U, tol=1e-4)
            V_s2 = _snap_to_half_ints(best_V, tol=1e-4)
            W_s2 = _snap_to_half_ints(best_W, tol=1e-4)
            if _verify_half_int(U_s2, V_s2, W_s2, tol=1e-4):
                return U_s2, V_s2, W_s2, r
    return None, None, None, None


def _schoolbook():
    """Return the schoolbook decomposition of rank 105."""
    rank = RANK_SCHOOLBOOK
    U = [[0] * rank for _ in range(N)]
    V = [[0] * rank for _ in range(M)]
    W = [[0] * rank for _ in range(P)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x5x7 matrix multiplication.
    Returns {'U', 'V', 'W'} with shapes (n*m, rank), (m*p, rank), (p*n, rank).
    """
    # Start with schoolbook as fallback
    result = _schoolbook()
    
    # Try to find lower rank decompositions
    # Known lower bounds: for n x m x p matmul, rank >= max(n*m, m*p, p*n) / something
    # Actually the trivial lower bound is max(n, m, p) but that's not tight.
    # For square n x n x n, Strassen gives 7 for n=2.
    # For 3x3x3, best known is 23 (or 20 with some schemes).
    # For 3x5x7, let's try ranks from a reasonable lower bound up.
    
    # Lower bound: rank >= ceil(n*m*p / max possible sharing)
    # A simple lower bound: rank >= max(n, m, p) = 7 (not useful)
    # Better: the tensor has support size n*m*p = 105, but rank can be less.
    # Let's try from 40 up to 104.
    
    best_rank = RANK_SCHOOLBOOK
    best_result = result
    
    # Try a few ranks in decreasing order of likelihood
    ranks_to_try = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 104]
    
    for r in ranks_to_try:
        if _time_left() < 20:
            break
        if r >= best_rank:
            continue
        
        U, V, W, found_rank = _try_rank(r, max_iters=3000, lr_init=0.0005, restarts=2)
        if U is not None:
            # Convert to list of lists with exact values
            U_list = [[float(x) for x in row] for row in U]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W]
            
            # Verify once more
            U_np = np.array(U_list)
            V_np = np.array(V_list)
            W_np = np.array(W_list)
            if _verify_half_int(U_np, V_np, W_np):
                best_rank = found_rank
                best_result = {"U": U_list, "V": V_list, "W": W_list}
                # Don't break, try even lower
                # But adjust the search range
                ranks_to_try = [r2 for r2 in ranks_to_try if r2 < found_rank]
                continue
    
    # If we found something better, use it; otherwise schoolbook
    if best_rank < RANK_SCHOOLBOOK:
        return best_result
    return result
