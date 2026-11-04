"""
Rank-minimal exact tensor decomposition for 3x4 @ 4x6 matrix multiplication.

Strategy:
1. Start from the schoolbook (rank 72) decomposition, which is always exact.
2. Use Alternating Least Squares (ALS) to search for a lower-rank decomposition.
3. Validate the resulting tensor identity and return the best (lowest-rank) valid decomposition.
"""
import time
import random
import numpy as np

TIME_LIMIT = 140  # seconds; evaluator kills at 150s, keep safety margin

DIMENSIONS = [3, 4, 6]


def _construct_schoolbook(n, m, p):
    """Return the standard rank n*m*p decomposition as (U, V, W) in list form."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W, rank


def _validate_tensor(U, V, W, n, m, p, rank):
    """
    Verify that for all i,j,k:
        T[i, j, k] = sum_{r=0}^{rank-1} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    """
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    total_tensor = np.zeros((n, m, p), dtype=np.float64)
    for r in range(rank):
        U_r = U_arr[:, r].reshape(n, m)
        V_r = V_arr[:, r].reshape(m, p)
        W_r = W_arr[:, r].reshape(p, n)
        total_tensor += np.einsum('ij,jk,ki->ijk', U_r, V_r, W_r)
    
    expected = np.ones((n, m, p), dtype=np.float64)
    return np.allclose(total_tensor, expected, atol=1e-12, rtol=1e-12)


def _als_decompose(n, m, p, rank, max_iters=50, tol=1e-10):
    """
    Attempt to find a rank-`rank` decomposition using Alternating Least Squares.
    Returns (U, V, W) as numpy arrays or None if failed.
    """
    U = np.random.randn(n * m, rank)
    V = np.random.randn(m * p, rank)
    W = np.random.randn(p * n, rank)
    
    for it in range(max_iters):
        U_new = np.zeros_like(U)
        for i in range(n):
            for j in range(m):
                A_mat = np.zeros((p, rank))
                b_vec = np.ones(p)
                for k in range(p):
                    A_mat[k, :] = V[j * p + k, :] * W[k * n + i, :]
                u_sol, residuals, rank_lsq, s = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                U_new[i * m + j, :] = u_sol
        
        V_new = np.zeros_like(V)
        for j in range(m):
            for k in range(p):
                A_mat = np.zeros((n, rank))
                b_vec = np.ones(n)
                for i in range(n):
                    A_mat[i, :] = U[i * m + j, :] * W[k * n + i, :]
                v_sol, residuals, rank_lsq, s = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                V_new[j * p + k, :] = v_sol
        
        W_new = np.zeros_like(W)
        for k in range(p):
            for i in range(n):
                A_mat = np.zeros((m, rank))
                b_vec = np.ones(m)
                for j in range(m):
                    A_mat[j, :] = U[i * m + j, :] * V[j * p + k, :]
                w_sol, residuals, rank_lsq, s = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                W_new[k * n + i, :] = w_sol
        
        U, V, W = U_new, V_new, W_new
        
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                    error += (val - 1.0) ** 2
        
        if error < tol:
            break
            
    error = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                error += (val - 1.0) ** 2
                
    if error < 1e-8:
        return U, V, W
    return None


def construct():
    """
    Construct a rank-minimal (or near-minimal) exact tensor decomposition
    for multiplying a 3x4 matrix by a 4x6 matrix.
    
    Returns:
        dict with keys 'U', 'V', 'W' containing the factor matrices.
        U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
        Entries are integers or half-integers.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    U_school, V_school, W_school, rank_school = _construct_schoolbook(n, m, p)
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school
    
    # Try to find a lower rank using ALS
    for rank in range(rank_school - 1, max(rank_school - 20, 40), -1):
        if time.time() - start_time > TIME_LIMIT - 10:
            break
            
        found = False
        for restart in range(3):
            if time.time() - start_time > TIME_LIMIT - 10:
                break
                
            result = _als_decompose(n, m, p, rank, max_iters=20, tol=1e-10)
            if result is not None:
                U, V, W = result
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                U_list = U_rounded.tolist()
                V_list = V_rounded.tolist()
                W_list = W_rounded.tolist()
                
                if _validate_tensor(U_list, V_list, W_list, n, m, p, rank):
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = rank
                    found = True
                    break
        
        if found:
            break

    return best_result
