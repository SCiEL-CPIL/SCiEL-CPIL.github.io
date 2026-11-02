"""Rewritten program: search for a lower-rank exact decomposition of the 2x3 * 3x10 matmul tensor."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 10]
N, M, P = DIMENSIONS
RANK_SCHOOLBOOK = N * M * P  # 60

# Deadline guard: keep a safety margin before the 150s wall-clock limit.
DEADLINE = 135.0  # seconds
_start_time = time.time()

def _elapsed():
    return time.time() - _start_time

def _time_left():
    return DEADLINE - _elapsed()

def _verify(U, V, W, tol=1e-6):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor.
    
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Convention: W[k*n + i, r]
    Tensor T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    must equal 1 for all i, j, k (since (A*B)[i,k] = sum_j A[i,j]*B[j,k]).
    """
    rank = U.shape[1]
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U3 = U.reshape(N, M, rank)
    V3 = V.reshape(M, P, rank)
    W3 = W.reshape(P, N, rank)
    
    # Use einsum: T[i,j,k] = sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    
    # The matmul tensor is all ones
    expected = np.ones((N, M, P))
    return np.allclose(T, expected, atol=tol)

def _schoolbook():
    """Return the trivial rank-60 schoolbook decomposition."""
    U = np.zeros((N * M, RANK_SCHOOLBOOK))
    V = np.zeros((M * P, RANK_SCHOOLBOOK))
    W = np.zeros((P * N, RANK_SCHOOLBOOK))
    col = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i * M + j, col] = 1.0
                V[j * P + k, col] = 1.0
                W[k * N + i, col] = 1.0
                col += 1
    return U, V, W

def _try_random_search(max_rank, n_iter=5000, seed=42):
    """Random search for a decomposition of given rank using integer/half-integer entries."""
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)
    
    for it in range(n_iter):
        if _time_left() < 5.0:
            break
        
        # Random integer/half-integer entries in range [-2, 2]
        # Values: -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2
        vals = np.array([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2])
        
        U = np_rng.choice(vals, size=(N * M, max_rank))
        V = np_rng.choice(vals, size=(M * P, max_rank))
        W = np_rng.choice(vals, size=(P * N, max_rank))
        
        if _verify(U, V, W, tol=1e-8):
            return U, V, W
    return None

def _try_structured_search(max_rank):
    """Try structured low-rank decompositions based on known matmul tensor structure."""
    for rank in range(15, max_rank + 1):
        if _time_left() < 10.0:
            break
        result = _try_random_search(rank, n_iter=2000, seed=rank * 1000 + 42)
        if result is not None:
            return result
    return None

def _try_alternating_least_squares(rank, n_iter=200, tol=1e-10, seed=0):
    """
    Optimized Alternating Least Squares (ALS) using vectorized Khatri-Rao products.
    """
    np_rng = np.random.RandomState(seed)
    
    # Initialize randomly with small values
    U = np_rng.randn(N * M, rank) * 0.1
    V = np_rng.randn(M * P, rank) * 0.1
    W = np_rng.randn(P * N, rank) * 0.1
    
    # Precompute T as a 3D array
    T = np.zeros((N * M, M * P, P * N))
    for i in range(N):
        for j in range(M):
            for k in range(P):
                a = i * M + j
                b = j * P + k
                c = k * N + i
                T[a, b, c] = 1.0
    
    # Precompute flattened T views
    T_flat_U = T.reshape(N * M, M * P * P * N)
    T_flat_V = T.reshape(M * P, N * M * P * N)
    T_flat_W = T.reshape(P * N, N * M * M * P)
    
    for it in range(n_iter):
        if _time_left() < 10.0:
            break
        
        # Update U: Solve U B^T = T_flat_U
        # B is Khatri-Rao of V and W
        # V: (M*P, rank), W: (P*N, rank)
        # B: (M*P * P*N, rank)
        # B[(b,c), r] = V[b, r] * W[c, r]
        # Vectorized: np.einsum('br,cr->bcr', V, W).reshape(-1, rank)
        try:
            B = np.einsum('br,cr->bcr', V, W).reshape(-1, rank)
            U_new, _, _, _ = np.linalg.lstsq(B.T, T_flat_U.T, rcond=None)
            U_new = U_new.T
        except:
            break
        
        # Update V: Solve V C^T = T_flat_V
        # C is Khatri-Rao of U and W
        # C: (N*M * P*N, rank)
        # C[(a,c), r] = U[a, r] * W[c, r]
        try:
            C = np.einsum('ar,cr->acr', U_new, W).reshape(-1, rank)
            V_new, _, _, _ = np.linalg.lstsq(C.T, T_flat_V.T, rcond=None)
            V_new = V_new.T
        except:
            break
        
        # Update W: Solve W D^T = T_flat_W
        # D is Khatri-Rao of U and V
        # D: (N*M * M*P, rank)
        # D[(a,b), r] = U[a, r] * V[b, r]
        try:
            D = np.einsum('ar,br->abr', U_new, V_new).reshape(-1, rank)
            W_new, _, _, _ = np.linalg.lstsq(D.T, T_flat_W.T, rcond=None)
            W_new = W_new.T
        except:
            break
        
        U, V, W = U_new, V_new, W_new
        
        # Check convergence
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        if np.allclose(T_approx, T, atol=tol):
            break
    
    # Snap to nearest half-integers
    U_snap = np.round(U * 2) / 2
    V_snap = np.round(V * 2) / 2
    W_snap = np.round(W * 2) / 2
    
    if _verify(U_snap, V_snap, W_snap, tol=1e-6):
        return U_snap, V_snap, W_snap
    
    return None

def construct():
    """Find a rank-minimal exact tensor decomposition for 2x3 * 3x10 matrix multiplication."""
    global _start_time
    _start_time = time.time()
    
    # First, get the schoolbook solution as a fallback
    U_fb, V_fb, W_fb = _schoolbook()
    
    best = (U_fb, V_fb, W_fb)
    best_rank = RANK_SCHOOLBOOK
    
    # Try ALS for lower ranks
    # Start from a reasonable lower bound
    # Theoretical lower bound is max(n*m, m*p, p*n) = 30? No, that's not a lower bound for tensor rank.
    # But let's try ranks from 30 up to 59.
    # We'll try a few ranks and a few random initializations for each.
    
    # Targeted search for lower ranks based on known patterns (e.g., 58, 56, 54...)
    # We prioritize higher ranks first as they are more likely to exist.
    candidate_ranks = [58, 56, 54, 52, 50, 48, 45, 40, 35, 30]
    candidate_ranks = [r for r in candidate_ranks if r < best_rank]
    
    for rank in candidate_ranks:
        if _time_left() < 20.0:
            break
        
        # Try multiple seeds for this rank
        found = False
        for seed in range(10):
            if _time_left() < 15.0:
                break
            
            result = _try_alternating_least_squares(rank, n_iter=100, tol=1e-10, seed=seed)
            if result is not None:
                U, V, W = result
                if _verify(U, V, W, tol=1e-8):
                    best = (U, V, W)
                    best_rank = rank
                    found = True
                    break
        
        if found:
            # If we found a lower rank, we can continue to search for even lower ones
            # if time permits, but we should be cautious.
            # For now, we break out of the candidate loop to save time for verification/return
            # or we can continue. Let's continue to try to go lower.
            pass
    
    # Fallback to random search for very low ranks if time remains
    if _time_left() > 30.0:
        for rank in range(20, best_rank):
            if _time_left() < 15.0:
                break
            result = _try_random_search(rank, n_iter=1000, seed=rank * 7919 + 13)
            if result is not None:
                U, V, W = result
                if _verify(U, V, W, tol=1e-8):
                    best = (U, V, W)
                    best_rank = rank
                    break
    
    U, V, W = best
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
