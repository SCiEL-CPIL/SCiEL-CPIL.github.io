"""Matrix multiplication 3x4x4 rank-minimal decomposition search."""
import time
import random
import itertools
from fractions import Fraction

import numpy as np

DIMENSIONS = [3, 4, 4]
N, M, P = DIMENSIONS
SCHOOLBOOK_RANK = N * M * P

# Deadline guard: evaluator kills after 150s, use 120s budget with 30s safety margin
_SEARCH_DEADLINE = 120.0
_start_time = None


def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i,j,k] = 1."""
    return np.ones((n, m, p), dtype=np.float64)


def _verify(U, V, W, tol=1e-9):
    """Verify that U, V, W decompose the matrix multiplication tensor."""
    n, m, p = N, M, P
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    if U.ndim == 1:
        U = U.reshape(-1, 1)
    if V.ndim == 1:
        V = V.reshape(-1, 1)
    if W.ndim == 1:
        W = W.reshape(-1, 1)
    rank = U.shape[1]
    # Reconstruct tensor
    T = np.zeros((n, m, p), dtype=np.float64)
    for r in range(rank):
        T += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
    target = np.ones((n, m, p), dtype=np.float64)
    return np.allclose(T, target, atol=tol)


def _to_fraction_list(mat):
    """Convert a numpy array to a list of lists of Fractions for exact output."""
    result = []
    for row in mat:
        r = []
        for val in row:
            f = Fraction(val).limit_denominator(1000000)
            r.append(f.numerator / f.denominator if f.denominator != 1 else f.numerator)
        result.append(r)
    return result


def _alternating_minimize(target_rank, seed, max_iters=500, tol=1e-10):
    """
    Alternating minimization to find rank-target_rank decomposition.
    Returns (U, V, W) as numpy arrays of shape (n*m, rank), (m*p, rank), (p*n, rank) or None.
    """
    global _start_time
    n, m, p = N, M, P
    T = np.ones((n, m, p), dtype=np.float64)
    
    rng = np.random.default_rng(seed)
    
    # Initialize with random values
    U = rng.normal(size=(n * m, target_rank)) * 0.3
    V = rng.normal(size=(m * p, target_rank)) * 0.3
    W = rng.normal(size=(p * n, target_rank)) * 0.3
    
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        if time.time() - _start_time > _SEARCH_DEADLINE - 5:
            break
        
        # Update U
        for i in range(n):
            for j in range(m):
                W_i = W[:, i]  # (p, target_rank)
                V_j = V[j * p : (j + 1) * p, :]  # (p, target_rank)
                A = V_j * W_i  # (p, target_rank)
                try:
                    u, _, _, _ = np.linalg.lstsq(A, np.ones(p), rcond=None)
                    U[i * m + j, :] = u
                except:
                    pass
        
        # Update V
        for j in range(m):
            for k in range(p):
                B = np.zeros((n, target_rank))
                for i in range(n):
                    B[i, :] = U[i * m + j, :] * W[k * n + i, :]
                try:
                    v, _, _, _ = np.linalg.lstsq(B, np.ones(n), rcond=None)
                    V[j * p + k, :] = v
                except:
                    pass
        
        # Update W
        for k in range(p):
            for i in range(n):
                C = np.zeros((m, target_rank))
                for j in range(m):
                    C[j, :] = U[i * m + j, :] * V[j * p + k, :]
                try:
                    w, _, _, _ = np.linalg.lstsq(C, np.ones(m), rcond=None)
                    W[k * n + i, :] = w
                except:
                    pass
        
        # Check convergence
        T_recon = np.zeros((n, m, p), dtype=np.float64)
        for r in range(target_rank):
            T_recon += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
        err = np.max(np.abs(T_recon - T))
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if err < tol:
            break
    
    if best_err < 1e-6:
        return best_U, best_V, best_W
    return None


def _try_rank_with_restarts(target_rank, num_restarts=50, max_iters=300):
    """Try multiple random restarts for a given target rank."""
    global _start_time
    for seed in range(num_restarts):
        if time.time() - _start_time > _SEARCH_DEADLINE - 10:
            break
        result = _alternating_minimize(target_rank, seed=seed, max_iters=max_iters)
        if result is not None:
            U, V, W = result
            # Verify
            if _verify(U, V, W, tol=1e-5):
                return U, V, W
    return None


def _greedy_rank_reduce():
    """
    Try to find a reduced-rank decomposition using alternating minimization with multiple restarts.
    """
    global _start_time
    n, m, p = N, M, P
    rank0 = n * m * p
    
    best_rank = rank0
    best_U = None
    best_V = None
    best_W = None
    
    # Try ranks from rank0-1 down to 40
    for target_rank in range(rank0 - 1, 39, -1):
        if time.time() - _start_time > _SEARCH_DEADLINE - 15:
            break
        
        # Use more restarts for lower ranks (harder problems)
        num_restarts = 30 if target_rank > 45 else 50
        result = _try_rank_with_restarts(target_rank, num_restarts=num_restarts)
        if result is not None:
            U, V, W = result
            best_rank = target_rank
            best_U, best_V, best_W = U, V, W
            break
    
    if best_U is not None:
        # Round to nearest integer/half-integer
        U_mat = np.round(best_U * 2) / 2
        V_mat = np.round(best_V * 2) / 2
        W_mat = np.round(best_W * 2) / 2
        # Verify after rounding
        if _verify(U_mat, V_mat, W_mat, tol=1e-8):
            return U_mat, V_mat, W_mat
        # If rounding broke it, try without rounding
        if _verify(best_U, best_V, best_W, tol=1e-8):
            return best_U, best_V, best_W
        # Try finer rounding
        U_mat2 = np.round(best_U * 10) / 10
        V_mat2 = np.round(best_V * 10) / 10
        W_mat2 = np.round(best_W * 10) / 10
        if _verify(U_mat2, V_mat2, W_mat2, tol=1e-8):
            return U_mat2, V_mat2, W_mat2
    
    return None


def _schoolbook():
    """Return the schoolbook decomposition."""
    n, m, p = N, M, P
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    return U, V, W


def construct():
    global _start_time
    _start_time = time.time()
    
    # Try to find a reduced-rank decomposition
    result = None
    try:
        result = _greedy_rank_reduce()
    except Exception:
        result = None
    
    if result is not None:
        U, V, W = result
        rank = U.shape[1]
        # Convert to list format with integer/half-integer entries
        U_list = _to_fraction_list(U)
        V_list = _to_fraction_list(V)
        W_list = _to_fraction_list(W)
        # Final verification
        if _verify(U_list, V_list, W_list, tol=1e-8):
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: schoolbook decomposition
    U, V, W = _schoolbook()
    U_list = _to_fraction_list(U)
    V_list = _to_fraction_list(V)
    W_list = _to_fraction_list(W)
    return {"U": U_list, "V": V_list, "W": W_list}
