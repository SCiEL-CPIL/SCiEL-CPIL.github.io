"""Tensor decomposition search for 4x4 * 4x7 matrix multiplication."""
import time
import random
import itertools
import numpy as np

DIMENSIONS = [4, 4, 7]
N, M, P = DIMENSIONS
DEADLINE = 150.0  # seconds, with safety margin


def _build_tensor():
    """Build the matrix multiplication tensor T[i,j,k] indexed as (i*n+m*p) etc.
    Actually we need the bilinear form: C[i,k] = sum_j A[i,j]*B[j,k]
    Tensor T[a,b,c] where a=(i,j) in [0,n*m), b=(j,k) in [0,m*p), c=(k,i) in [0,p*n)
    T[a,b,c] = 1 if b's j-index == a's j-index, else 0.
    """
    T = np.zeros((N * M, M * P, P * N), dtype=np.int64)
    for i in range(N):
        for j in range(M):
            for k in range(P):
                a = i * M + j
                b = j * P + k
                c = k * N + i
                T[a, b, c] = 1
    return T


def _verify(U, V, W, T, tol=1e-6):
    """Check that sum_r U[:,r]*V[:,r]*W[:,r] == T."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    # Compute tensor reconstruction
    # T_recon[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # Efficient: einsum('ar,br,cr->abc', U, V, W)
    T_recon = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
    return np.allclose(T_recon, T, atol=tol)


def _schoolbook():
    """Return the schoolbook decomposition of rank n*m*p=112."""
    n, m, p = N, M, P
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def _try_rank_decomposition(T, rank, time_budget, rng):
    """Try to find a rank-`rank` decomposition via random initialization + gradient-like refinement.
    Returns (U, V, W) or None."""
    n, m, p = N, M, P
    dim_a, dim_b, dim_c = n * m, m * p, p * n
    
    # Use random initialization with small integers/half-integers
    # Strategy: random sign patterns + scaling, then refine
    best_err = np.inf
    best_UVW = None
    
    start_time = time.time()
    max_iters = 5000
    
    for attempt in range(20):
        if time.time() - start_time > time_budget:
            break
        # Random initialization
        U = rng.standard_normal((dim_a, rank)) * 0.5
        V = rng.standard_normal((dim_b, rank)) * 0.5
        W = rng.standard_normal((dim_c, rank)) * 0.5
        
        # Simple iterative refinement: alternating least squares style
        # For each factor, fix the other two and solve linear system
        err = np.inf
        for it in range(max_iters):
            if time.time() - start_time > time_budget:
                break
            
            # Update U: for each a, U[a,:] should satisfy sum_b,c T[a,b,c]*V[b,:]*W[c,:] = U[a,:] * (V*W outer sum)
            # Actually the tensor equation is: sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
            # Fix V,W: U[a,r] * sum_b,c T[a,b,c] * V[b,r]*W[c,r]... no, this is nonlinear in U too.
            # 
            # Better: use the fact that for fixed V,W, the equation becomes:
            # For each (a,b,c): sum_r U[a,r] * (V[b,r]*W[c,r]) = T[a,b,c]
            # This is a linear system in U[a,:] for each a, but the RHS depends on b,c.
            # 
            # Actually let's use a simpler approach: gradient descent on the loss
            # L = 0.5 * sum_{a,b,c} (sum_r U[a,r]V[b,r]W[c,r] - T[a,b,c])^2
            # 
            # Gradient w.r.t. U[a,r] = sum_{b,c} (sum_s U[a,s]V[b,s]W[c,s] - T[a,b,c]) * V[b,r] * W[c,r]
            
            # Compute current reconstruction
            recon = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
            diff = recon - T  # shape (dim_a, dim_b, dim_c)
            err = np.sum(diff * diff)
            
            if err < 1e-10:
                break
            
            if err < best_err:
                best_err = err
                best_UVW = (U.copy(), V.copy(), W.copy())
            
            # Gradient step
            lr = 0.01
            # grad_U[a,r] = einsum('abc,br,cr->ar', diff, V, W)
            grad_U = np.einsum('abc,br,cr->ar', diff, V, W, optimize=True)
            grad_V = np.einsum('abc,ar,cr->br', diff, U, W, optimize=True)
            grad_W = np.einsum('abc,ar,br->cr', diff, U, V, optimize=True)
            
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
        
        if best_err < 1e-8:
            break
    
    if best_err < 1e-8:
        return best_UVW
    return None


def _try_small_ranks(T, time_budget, rng):
    """Try ranks from a small set downward."""
    # Known lower bound: rank >= max(n*m, m*p, p*n) = max(16,28,28) = 28
    # Upper bound: 112 (schoolbook)
    # Try some candidate ranks
    candidates = [28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 72, 80, 88, 96, 104, 112]
    
    for rank in candidates:
        if time.time() > time_budget:
            break
        time_per_rank = min(5.0, (time_budget - time.time()) / max(1, len(candidates) - candidates.index(rank)))
        result = _try_rank_decomposition(T, rank, time_per_rank, rng)
        if result is not None:
            return result, rank
    return None, None


def construct():
    start_time = time.time()
    rng = random.Random(42)
    np_rng = np.random.default_rng(42)
    
    # Get the tensor
    T = _build_tensor()
    
    # Try to find a lower-rank decomposition
    time_budget = DEADLINE - 5  # safety margin
    result, rank = _try_small_ranks(T, time_budget, np_rng)
    
    if result is not None:
        U, V, W = result
        # Convert to lists for the required interface
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    
    # Fallback: schoolbook
    U, V, W = _schoolbook()
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
