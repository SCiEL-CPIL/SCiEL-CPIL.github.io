import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 4]
_START_TIME = time.time()
_TIME_LIMIT = 115.0  # safety margin before 120s

def _schoolbook():
    n, m, p = DIMENSIONS
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

def _is_half_integer(x, tol=1e-9):
    return abs(x * 2 - round(x * 2)) < tol

def _snap_to_half_integer(x, tol=1e-9):
    if abs(x * 2 - round(x * 2)) < tol:
        return round(x * 2) / 2.0
    return None

def _verify(U, V, W, tol=1e-9):
    n, m, p = DIMENSIONS
    # Build the tensor T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # and check it equals the identity-like tensor for matmul
    # The matmul tensor: T[i,j,k] = 1 if the (i,j) element of A times (j,k) of B
    # gives (i,k) of C. So T[i,j,k] should be 1 for all valid i,j,k.
    # Actually the tensor decomposition means:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # = sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] A[i,j] B[j,k] ... no
    # The tensor T[i,j,k] for matmul (n x m) * (m x p) -> (n x p) is:
    # T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # because C[i,k] = sum_j A[i,j]*B[j,k]
    # So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # Efficient check: for each r, the term is outer product
    # Let's compute the full tensor
    T = np.zeros((n, m, p))
    for r in range(U.shape[1]):
        Ur = U[:, r].reshape(n, m)
        Vr = V[:, r].reshape(m, p)
        Wr = W[:, r].reshape(p, n)
        # The contribution to T[i,j,k] from rank r is U[i,j,r]*V[j,k,r]*W[k,i,r]
        # This is not a simple outer product; it's a tensor product
        # T[i,j,k] += U[i,j,r] * V[j,k,r] * W[k,i,r]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i,j,k] += Ur[i,j] * Vr[j,k] * Wr[k,i]
    # Check T equals ones
    return np.allclose(T, 1.0, atol=tol)

def _als_decompose(n, m, p, rank, max_iter=200, tol=1e-10, seed=0):
    """ALS decomposition for matrix multiplication tensor."""
    rng = np.random.RandomState(seed)
    # Initialize
    U = rng.randn(n * m, rank) * 0.1
    V = rng.randn(m * p, rank) * 0.1
    W = rng.randn(p * n, rank) * 0.1
    
    for it in range(max_iter):
        if time.time() - _START_TIME > _TIME_LIMIT - 5:
            break
        # Update U: for each (i,j), solve least squares
        for i in range(n):
            for j in range(m):
                # T[i,j,k] = 1 for all k
                # sum_r U[i,j,r] * V[j*p+k, r] * W[k*n+i, r] = 1
                # For fixed i,j: sum_r U[i,j,r] * (V[j*p+0:r] * W[0*n+i:r] ... ) 
                # Let's build the system: for each k, sum_r U[i,j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
                # This is: U[i,j,:] dot (V[j*p+k,:] * W[k*n+i,:]) = 1 for each k
                # So we have p equations in rank unknowns
                A = np.zeros((p, rank))
                b = np.ones(p)
                for k in range(p):
                    A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                # Solve A @ U[i,j,:] = b
                try:
                    U[i*m+j, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                except:
                    pass
        # Update V
        for j in range(m):
            for k in range(p):
                # sum_r U[i*m+j,r] * V[j,k,r] * W[k*n+i,r] = 1 for all i
                A = np.zeros((n, rank))
                b = np.ones(n)
                for i in range(n):
                    A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                try:
                    V[j*p+k, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                except:
                    pass
        # Update W
        for k in range(p):
            for i in range(n):
                # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k,i,r] = 1 for all j
                A = np.zeros((m, rank))
                b = np.ones(m)
                for j in range(m):
                    A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                try:
                    W[k*n+i, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                except:
                    pass
        # Check convergence
        if it % 10 == 0:
            if _verify(U, V, W, tol=1e-6):
                break
    return U, V, W

def _snap_and_verify(U, V, W):
    """Try to snap to half-integers and verify."""
    n, m, p = DIMENSIONS
    snapped_U = np.zeros_like(U)
    snapped_V = np.zeros_like(V)
    snapped_W = np.zeros_like(W)
    
    for idx in range(U.size):
        val = _snap_to_half_integer(U.flat[idx])
        if val is None:
            return None
        snapped_U.flat[idx] = val
    for idx in range(V.size):
        val = _snap_to_half_integer(V.flat[idx])
        if val is None:
            return None
        snapped_V.flat[idx] = val
    for idx in range(W.size):
        val = _snap_to_half_integer(W.flat[idx])
        if val is None:
            return None
        snapped_W.flat[idx] = val
    
    if _verify(snapped_U, snapped_V, snapped_W, tol=1e-9):
        return snapped_U, snapped_V, snapped_W
    return None

def construct():
    n, m, p = DIMENSIONS
    best_rank = n * m * p
    best_result = None
    
    # Start with schoolbook
    U0, V0, W0 = _schoolbook()
    best_result = (U0, V0, W0)
    
    # Try lower ranks with ALS
    for rank in range(n * m + p - 1, best_rank):  # Strassen-like lower bound
        if time.time() - _START_TIME > _TIME_LIMIT - 10:
            break
        for seed in range(5):
            if time.time() - _START_TIME > _TIME_LIMIT - 10:
                break
            U, V, W = _als_decompose(n, m, p, rank, max_iter=100, seed=seed)
            snapped = _snap_and_verify(U, V, W)
            if snapped is not None:
                best_rank = rank
                best_result = snapped
                break
        if best_rank < n * m * p:
            break
    
    U, V, W = best_result
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
