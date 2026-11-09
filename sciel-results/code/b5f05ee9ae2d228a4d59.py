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


def _greedy_rank_reduce():
    """
    Greedy approach: try to find a reduced-rank decomposition using alternating minimization.
    """
    global _start_time
    n, m, p = N, M, P
    rank0 = n * m * p
    
    # Build schoolbook U, V, W as lists of columns
    U_cols = []
    V_cols = []
    W_cols = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = [0.0] * (n * m)
                u[i * m + j] = 1.0
                v = [0.0] * (m * p)
                v[j * p + k] = 1.0
                w = [0.0] * (p * n)
                w[k * n + i] = 1.0
                U_cols.append(u)
                V_cols.append(v)
                W_cols.append(w)
    
    best_rank = rank0
    best_U = U_cols
    best_V = V_cols
    best_W = W_cols
    
    # Try ranks from rank0-1 down to 20
    for target_rank in range(rank0 - 1, max(rank0 - 10, 20), -1):
        if time.time() - _start_time > _SEARCH_DEADLINE - 10:
            break
        success = _try_rank(target_rank, U_cols, V_cols, W_cols)
        if success is not None:
            best_rank = target_rank
            best_U, best_V, best_W = success
            break
    
    if best_rank < rank0:
        U_mat = np.array(best_U).T  # (n*m, rank)
        V_mat = np.array(best_V).T  # (m*p, rank)
        W_mat = np.array(best_W).T  # (p*n, rank)
        # Round to nearest integer/half-integer
        U_mat = np.round(U_mat * 2) / 2
        V_mat = np.round(V_mat * 2) / 2
        W_mat = np.round(W_mat * 2) / 2
        return U_mat, V_mat, W_mat
    else:
        U_mat = np.array(U_cols).T
        V_mat = np.array(V_cols).T
        W_mat = np.array(W_cols).T
        return U_mat, V_mat, W_mat


def _try_rank(target_rank, U_init, V_init, W_init, max_iters=200):
    """
    Try to find a rank-`target_rank` decomposition via alternating minimization.
    Returns (U_list, V_list, W_list) as list of columns, or None if failed.
    """
    global _start_time
    n, m, p = N, M, P
    T = np.ones((n, m, p), dtype=np.float64)
    
    # Initialize with random values scaled appropriately
    rng = np.random.default_rng(42)
    U = rng.normal(size=(n * m, target_rank)) * 0.5
    V = rng.normal(size=(m * p, target_rank)) * 0.5
    W = rng.normal(size=(p * n, target_rank)) * 0.5
    
    for it in range(max_iters):
        if time.time() - _start_time > _SEARCH_DEADLINE:
            return None
        
        # Update U: for each entry (i, j), U[i*m+j, :] = argmin ||T[i,j,:] - V[j,:] ⊙ W[:,i] * U[i*m+j,:]||
        # This is a rank-1 update problem per (i,j)
        for i in range(n):
            for j in range(m):
                # T[i,j,k] for k in range(p)
                # W[:, i] gives W[k*n + i, r] for k, r
                W_i = W[:, i]  # (p, target_rank)
                # We want: sum_r U[i*m+j, r] * V[j*p+k, r] * W_i[k, r] = T[i,j,k] = 1
                # For each k: sum_r U[i*m+j, r] * (V[j*p+k, r] * W_i[k, r]) = 1
                # This is: U[i*m+j, :] @ diag(V_row * W_i_row) = 1 for each k
                # Let's form the matrix A[k, r] = V[j*p+k, r] * W_i[k, r]
                V_j = V[j * p : (j + 1) * p, :]  # (p, target_rank)
                A = V_j * W_i  # (p, target_rank)
                # Solve A @ u = 1 (least squares)
                try:
                    u, residuals, rank_a, sv = np.linalg.lstsq(A, np.ones(p), rcond=None)
                except:
                    continue
                U[i * m + j, :] = u
        
        # Update V: for each entry (j, k)
        for j in range(m):
            for k in range(p):
                U_j = U[:, j]  # (n, target_rank)  -- wait, U is (n*m, target_rank)
                # U[i*m + j, r] for i, r
                W_k = W[k * n : (k + 1) * n, :]  # (n, target_rank)
                # V[j*p+k, r] * sum_i U[i*m+j, r] * W[k*n+i, r] ... 
                # Actually: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                # For fixed j,k: sum_r V[j*p+k, r] * (sum_i ... ) no, it's per i.
                # For fixed j,k: for each i, T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                # Let's form: for each i, the equation is:
                # V[j*p+k, :] @ (U[i*m+j, :] * W[k*n+i, :]) = 1
                # Stack over i:
                B = np.zeros((n, target_rank))
                for i in range(n):
                    B[i, :] = U[i * m + j, :] * W[k * n + i, :]
                # Solve B @ v = 1
                try:
                    v, residuals, rank_b, sv = np.linalg.lstsq(B, np.ones(n), rcond=None)
                except:
                    continue
                V[j * p + k, :] = v
        
        # Update W: for each entry (k, i)
        for k in range(p):
            for i in range(n):
                V_k = V[:, k]  # wait, V is (m*p, target_rank)
                # V[j*p+k, r] for j, r
                U_i = U[:, i]  # U[i*m + j, r] for j, r -- no, U[i, :] where i is row index
                # Let's be more careful
                # W[k*n+i, r]
                # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                # For fixed k,i: for each j, T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                C = np.zeros((m, target_rank))
                for j in range(m):
                    C[j, :] = U[i * m + j, :] * V[j * p + k, :]
                # Solve C @ w = 1
                try:
                    w, residuals, rank_c, sv = np.linalg.lstsq(C, np.ones(m), rcond=None)
                except:
                    continue
                W[k * n + i, :] = w
        
        # Check convergence
        T_recon = np.zeros((n, m, p), dtype=np.float64)
        for r in range(target_rank):
            T_recon += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
        err = np.max(np.abs(T_recon - T))
        if err < 1e-6:
            # Verify more carefully
            if _verify(U.T.tolist(), V.T.tolist(), W.T.tolist(), tol=1e-5):
                return [U[:, r].tolist() for r in range(target_rank)], \
                       [V[:, r].tolist() for r in range(target_rank)], \
                       [W[:, r].tolist() for r in range(target_rank)]
    
    return None


def construct():
    global _start_time
    _start_time = time.time()
    
    # Try to find a reduced-rank decomposition
    try:
        U, V, W = _greedy_rank_reduce()
    except Exception:
        # Fallback to schoolbook
        U, V, W = None, None, None
    
    if U is not None:
        rank = U.shape[1]
        # Convert to list format with integer/half-integer entries
        U_list = _to_fraction_list(U)
        V_list = _to_fraction_list(V)
        W_list = _to_fraction_list(W)
        # Verify
        if _verify(U_list, V_list, W_list, tol=1e-8):
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: schoolbook decomposition
    n, m, p = N, M, P
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
