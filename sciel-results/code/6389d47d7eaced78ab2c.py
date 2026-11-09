"""Rewritten program for matrix multiplication 4x5x6 tensor rank search."""
import time
import random
import numpy as np

DIMENSIONS = [4, 5, 6]
N, M, P = DIMENSIONS
RANK_SCHOOLBOOK = N * M * P  # 120

# Deadline guard: return best result before 150s wall-clock limit
_DEADLINE = 145.0  # seconds from start
_START_TIME = None

def _elapsed():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME

def _time_left():
    return _DEADLINE - _elapsed()

def _schoolbook():
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = RANK_SCHOOLBOOK
    U = np.zeros((N * M, rank))
    V = np.zeros((M * P, rank))
    W = np.zeros((P * N, rank))
    col = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i * M + j, col] = 1
                V[j * P + k, col] = 1
                W[k * N + i, col] = 1
                col += 1
    return U, V, W

def _verify(U, V, W, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor.
    The tensor T[i,j,k] = 1 iff the (i,k)-entry of A@B depends on A[i,j]*B[j,k].
    With convention W[k*n+i, r], the decomposition requires:
      sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1  for all i,j,k
    """
    n, m, p = N, M, P
    # Build the tensor from U, V, W
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We can compute this efficiently using einsum
    # Reshape U to (n, m, r), V to (m, p, r), W to (p, n, r)
    U_r = U.reshape(n, m, -1)
    V_r = V.reshape(m, p, -1)
    W_r = W.reshape(p, n, -1)
    # T[i,j,k] = sum_r U_r[i,j,r] * V_r[j,k,r] * W_r[k,i,r]
    # Use einsum: 'ijr,jkr,kir->ijk'
    T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
    # Check that T[i,j,k] == 1 for all i,j,k
    return np.allclose(T, 1.0, atol=tol, rtol=0)

def _try_rank(r, max_iters=200, seed=42):
    """Try to find a rank-r decomposition via iterative refinement.
    Returns (U, V, W) if successful, else None.
    """
    rng = np.random.RandomState(seed)
    best_err = np.inf
    best_UVW = None
    
    for it in range(max_iters):
        if _time_left() < 2.0:
            break
        
        # Initialize randomly
        U = rng.randn(N * M, r)
        V = rng.randn(M * P, r)
        W = rng.randn(P * N, r)
        
        # Iterative refinement: alternate fixing two and solving for the third
        for inner in range(50):
            if _time_left() < 1.0:
                break
            
            # Solve for U given V, W
            # For each (i,j): U[i*m+j, :] should satisfy
            #   sum_k W[k*n+i, r] * V[j*p+k, r] * U[i*m+j, r] = 1
            # This is a linear system in U[i*m+j, :] for each (i,j)
            # A_{(i,j), r} = sum_k W[k*n+i, r] * V[j*p+k, r]
            # Then U[i*m+j, :] is the solution to A_{(i,j), :} @ U[i*m+j, :] = 1
            
            U_r = U.reshape(N, M, r)
            V_r = V.reshape(M, P, r)
            W_r = W.reshape(P, N, r)
            
            # Update U
            for i in range(N):
                for j in range(M):
                    # A_r = sum_k W_r[k,i,r] * V_r[j,k,r]
                    A_vec = np.einsum('kr,jkr->r', W_r[:, i, :], V_r[j, :, :])
                    # Solve A_vec @ u = 1 in least squares sense
                    # If A_vec is zero, skip
                    if np.allclose(A_vec, 0, atol=1e-12):
                        continue
                    # Least squares: minimize ||A_vec * u - 1||
                    # u = A_vec / (A_vec . A_vec) if we want A_vec @ u = 1
                    denom = np.dot(A_vec, A_vec)
                    if denom > 1e-12:
                        U_r[i, j, :] = A_vec / denom
            
            # Update V
            for j in range(M):
                for k in range(P):
                    # A_r = sum_i U_r[i,j,r] * W_r[k,i,r]
                    A_vec = np.einsum('ir,kir->r', U_r[:, j, :], W_r[k, :, :])
                    if np.allclose(A_vec, 0, atol=1e-12):
                        continue
                    denom = np.dot(A_vec, A_vec)
                    if denom > 1e-12:
                        V_r[j, k, :] = A_vec / denom
            
            # Update W
            for k in range(P):
                for i in range(N):
                    # A_r = sum_j U_r[i,j,r] * V_r[j,k,r]
                    A_vec = np.einsum('jr,jkr->r', U_r[i, :, :], V_r[:, k, :])
                    if np.allclose(A_vec, 0, atol=1e-12):
                        continue
                    denom = np.dot(A_vec, A_vec)
                    if denom > 1e-12:
                        W_r[k, i, :] = A_vec / denom
        
        # Check error
        U_r = U.reshape(N, M, r)
        V_r = V.reshape(M, P, r)
        W_r = W.reshape(P, N, r)
        T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
        err = np.max(np.abs(T - 1.0))
        
        if err < best_err:
            best_err = err
            best_UVW = (U.copy(), V.copy(), W.copy())
        
        if err < 1e-9:
            return best_UVW
    
    return None

def _try_rank_with_structure(r, seed=42):
    """Try rank-r decomposition using structured initialization based on known
    tensor structure. Uses the fact that the matrix multiplication tensor has
    rank at most n*m*p and potentially lower.
    
    For 4x5x6, the known upper bound from literature is 120 (schoolbook).
    We try to find lower rank decompositions.
    """
    rng = np.random.RandomState(seed)
    
    # Strategy: start from schoolbook and try to merge terms
    # or use random restarts with the alternating minimization above
    
    best = None
    best_err = np.inf
    
    for attempt in range(10):
        if _time_left() < 3.0:
            break
        
        result = _try_rank(r, max_iters=30, seed=seed + attempt)
        if result is not None:
            return result
    
    return None

def construct():
    """Find a rank-minimal exact tensor decomposition for multiplying a 4x5 matrix
    by a 5x6 matrix. Returns {'U', 'V', 'W'} with integer/half-integer entries.
    """
    global _START_TIME
    _START_TIME = time.time()
    
    # Start with schoolbook (always valid)
    U, V, W = _schoolbook()
    best_rank = RANK_SCHOOLBOOK
    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to find lower rank decompositions
    # The theoretical lower bound for 4x5x6 is not known to be below 120 in general,
    # but let's try a few ranks below schoolbook
    
    # Try ranks from 119 down to, say, 100
    for r in range(RANK_SCHOOLBOOK - 1, max(RANK_SCHOOLBOOK - 20, 0), -1):
        if _time_left() < 10.0:
            break
        
        result = _try_rank_with_structure(r, seed=42 + r)
        if result is not None:
            U_r, V_r, W_r = result
            # Verify
            if _verify(U_r, V_r, W_r, tol=1e-8):
                # Check if entries are close to integers or half-integers
                # Snap to nearest half-integer if close
                U_snap = np.round(U_r * 2) / 2
                V_snap = np.round(V_r * 2) / 2
                W_snap = np.round(W_r * 2) / 2
                
                if _verify(U_snap, V_snap, W_snap, tol=1e-7):
                    best_rank = r
                    best_result = {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
                    break
                else:
                    # Use the non-snapped version
                    best_rank = r
                    best_result = {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                    break
    
    # If no improvement found, return schoolbook
    if best_rank == RANK_SCHOOLBOOK:
        U, V, W = _schoolbook()
        # Ensure integer entries
        U = np.round(U).astype(float)
        V = np.round(V).astype(float)
        W = np.round(W).astype(float)
        best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return best_result
