"""Rewritten program for 3x3 by 3x8 matrix multiplication tensor decomposition."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 8]
DEADLINE = 140.0  # safety margin before 150s wall-clock kill


def _build_schoolbook(n, m, p):
    """Return (U, V, W) for the trivial schoolbook decomposition, rank = n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def _tensor_from_factors(U, V, W):
    """
    Reconstruct the multiplication tensor T[i,j,k] from factors.
    T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    """
    n, m, p = DIMENSIONS
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
    return T


def _is_valid(U, V, W, tol=1e-6):
    """Check if the factors exactly reproduce the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    T = _tensor_from_factors(U, V, W)
    # The multiplication tensor has T[i,j,k] = 1 for all i,j,k
    return np.allclose(T, 1.0, atol=tol, rtol=tol)


def _try_rank(n, m, p, rank, time_budget=30.0, seed=0):
    """
    Try to find a rank-`rank` decomposition using iterative refinement.
    Returns (U, V, W) or None.
    """
    start = time.time()
    rng = np.random.default_rng(seed)

    # Initialize with random factors
    U = rng.standard_normal((n * m, rank))
    V = rng.standard_normal((m * p, rank))
    W = rng.standard_normal((p * n, rank))

    for iteration in range(2000):
        if time.time() - start > time_budget:
            break

        # Update U: for each (i,j), U[i*m+j, :] should satisfy
        # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all k
        # This is a linear system in U[i*m+j, :]
        # Let A[i,j,k,c] = V[j*p+k,c] * W[k*n+i,c]
        # Then sum_c U[i*m+j,c] * A[i,j,k,c] = 1 for all k
        # So U[i*m+j, :] is a solution to A @ U[i*m+j,:] = 1 (vector of length p)
        for i in range(n):
            for j in range(m):
                A = np.zeros((p, rank))
                for k in range(p):
                    A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                # Solve A @ u = ones(p)
                try:
                    U[i * m + j, :], residuals, rank_a, sv = np.linalg.lstsq(A, np.ones(p), rcond=None)
                except np.linalg.LinAlgError:
                    pass

        # Update V: for each (j,k), V[j*p+k, :] should satisfy
        # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i
        for j in range(m):
            for k in range(p):
                A = np.zeros((n, rank))
                for i in range(n):
                    A[i, :] = U[i * m + j, :] * W[k * n + i, :]
                try:
                    V[j * p + k, :], residuals, rank_a, sv = np.linalg.lstsq(A, np.ones(n), rcond=None)
                except np.linalg.LinAlgError:
                    pass

        # Update W: for each (k,i), W[k*n+i, :] should satisfy
        # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all j
        for k in range(p):
            for i in range(n):
                A = np.zeros((m, rank))
                for j in range(m):
                    A[j, :] = U[i * m + j, :] * V[j * p + k, :]
                try:
                    W[k * n + i, :], residuals, rank_a, sv = np.linalg.lstsq(A, np.ones(m), rcond=None)
                except np.linalg.LinAlgError:
                    pass

        # Check convergence
        if _is_valid(U, V, W, tol=1e-8):
            return U, V, W

    # Final check
    if _is_valid(U, V, W, tol=1e-6):
        return U, V, W
    return None


def _try_rank_with_structure(n, m, p, rank, time_budget=30.0, seed=0):
    """
    Try rank decomposition with some structural priors.
    For 3x3 x 3x8, we know that rank of 3x3x3 is at most 23 (actually 19 for 3x3x3? No, 3x3x3 is 27 schoolbook, but can be done in less).
    Actually, the bilinear complexity of 3x3x3 is 23 (Laderman's algorithm).
    For 3x3 x 3x8, we can think of it as 8 copies of 3x3x1? No.
    
    Let's try a different approach: use the fact that multiplying 3x3 by 3x8 is equivalent to 
    8 independent 3x3x1 multiplications? No, it's 3x3 times 3x8.
    
    Actually, A (3x3) times B (3x8) = C (3x8).
    C[i,k] = sum_j A[i,j] * B[j,k] for i in 0..2, k in 0..7.
    
    The tensor is T[i,j,k] = 1 for all i,j,k.
    
    We can try to exploit the structure: since m=3 and p=8, we can group the k dimension.
    """
    start = time.time()
    rng = np.random.default_rng(seed)

    # Try using Laderman-like structure for the 3x3 part
    # Laderman's algorithm for 3x3x3 has rank 23.
    # For 3x3 x 3x8, we could potentially use rank 23 * something?
    # Actually, C = A * B where A is 3x3, B is 3x8.
    # We can write B as [B1, B2, B3, B4, B5, B6, B7, B8] where each Bj is 3x1.
    # Then C = [A*B1, A*B2, ..., A*B8].
    # Each A*Bj is a 3x1 vector, which requires rank 3 (schoolbook) or can be done with shared structure.
    
    # Actually, the tensor for 3x3 x 3x8 has dimensions (3*3, 3*8, 8*3) = (9, 24, 24).
    # The schoolbook rank is 9*3*8/3... no, it's n*m*p = 3*3*8 = 72.
    
    # Let's try to find a lower rank by using the iterative method with better initialization.
    
    # Initialize U with random values, but scale to avoid numerical issues
    U = rng.standard_normal((n * m, rank))
    V = rng.standard_normal((m * p, rank))
    W = rng.standard_normal((p * n, rank))
    
    # Normalize to unit norm per row
    for i in range(n * m):
        norm = np.linalg.norm(U[i, :])
        if norm > 1e-10:
            U[i, :] /= norm
    for j in range(m * p):
        norm = np.linalg.norm(V[j, :])
        if norm > 1e-10:
            V[j, :] /= norm
    for k in range(p * n):
        norm = np.linalg.norm(W[k, :])
        if norm > 1e-10:
            W[k, :] /= norm

    for iteration in range(3000):
        if time.time() - start > time_budget:
            break

        # Update U
        for i in range(n):
            for j in range(m):
                A = np.zeros((p, rank))
                for k in range(p):
                    A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                try:
                    U[i * m + j, :], _, _, _ = np.linalg.lstsq(A, np.ones(p), rcond=None)
                except np.linalg.LinAlgError:
                    pass

        # Update V
        for j in range(m):
            for k in range(p):
                A = np.zeros((n, rank))
                for i in range(n):
                    A[i, :] = U[i * m + j, :] * W[k * n + i, :]
                try:
                    V[j * p + k, :], _, _, _ = np.linalg.lstsq(A, np.ones(n), rcond=None)
                except np.linalg.LinAlgError:
                    pass

        # Update W
        for k in range(p):
            for i in range(n):
                A = np.zeros((m, rank))
                for j in range(m):
                    A[j, :] = U[i * m + j, :] * V[j * p + k, :]
                try:
                    W[k * n + i, :], _, _, _ = np.linalg.lstsq(A, np.ones(m), rcond=None)
                except np.linalg.LinAlgError:
                    pass

        if _is_valid(U, V, W, tol=1e-8):
            return U, V, W

    if _is_valid(U, V, W, tol=1e-6):
        return U, V, W
    return None


def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 72
    
    # Start with schoolbook as fallback
    U_sb, V_sb, W_sb = _build_schoolbook(n, m, p)
    
    # Try to find lower rank decompositions
    # Known lower bounds: for 3x3x3, rank >= 19 (actually 23 is known to be minimal for 3x3x3)
    # For 3x3 x 3x8, the rank is at least... let's try various ranks
    
    # Try ranks from a reasonable lower bound up to schoolbook
    # Lower bound: max(n*m, m*p, p*n) = max(9, 24, 24) = 24
    # But actual lower bound is higher. Let's try from 25 up to 72.
    
    best_result = None
    best_rank = schoolbook_rank
    
    # Try a few promising ranks
    ranks_to_try = []
    for r in range(25, schoolbook_rank + 1):
        ranks_to_try.append(r)
    
    # Sort by rank (try lower ranks first)
    ranks_to_try.sort()
    
    for rank in ranks_to_try:
        if time.time() > DEADLINE - 10:
            break
        
        # Try a few seeds for each rank
        found = None
        for seed in range(5):
            if time.time() > DEADLINE - 10:
                break
            result = _try_rank(n, m, p, rank, time_budget=15.0, seed=seed)
            if result is not None:
                found = result
                break
        
        if found is not None:
            U, V, W = found
            best_result = (U, V, W)
            best_rank = rank
            # If we found a valid decomposition, we could stop, but let's try to find even lower
            # Actually, let's continue to see if we can find an even lower rank
            # But to save time, let's stop if we find one below 40
            if rank < 40:
                break
    
    if best_result is None:
        # Fall back to schoolbook
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    U, V, W = best_result
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
