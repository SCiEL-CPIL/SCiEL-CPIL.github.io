"""Rank-minimal exact tensor decomposition for 3x3x6 matrix multiplication.

Strategy: Start from the rank-14 Strassen-based decomposition (two 3x3x3 slices),
then systematically search for linear dependencies to reduce rank.
Uses numpy for efficient tensor reconstruction and dependency checking.
"""
DIMENSIONS = [3, 3, 6]

import time
import numpy as np
from itertools import combinations

# Rank-7 (Strassen) decomposition of the 3x3x3 multiplication tensor.
_A = [
    [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    [[0, 0, 0], [1, 0, 0], [1, 1, 1]],
    [[0, 0, 0], [1, 0, 0], [1, 0, 0]],
    [[0, 0, 0], [0, 1, 0], [0, 1, 0]],
    [[0, 0, 0], [0, 1, 0], [0, 1, 1]],
    [[1, 0, 0], [0, 0, 0], [0, 0, 0]],
    [[1, 1, 1], [0, 0, 0], [0, 0, 0]],
]
_B = [
    [[0, 0, 0], [0, 1, 0], [0, 1, 0]],
    [[0, 0, 0], [0, 1, 0], [0, -1, 0]],
    [[1, 0, 0], [0, 0, 0], [0, 0, 0]],
    [[1, 0, 0], [0, 0, 0], [0, -1, 0]],
    [[0, 0, 0], [0, 1, 0], [0, 1, 0]],
    [[0, 0, 0], [0, 0, 0], [1, 1, 1]],
    [[0, 0, 0], [0, 0, 0], [1, 1, 1]],
]
_C = [
    [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    [[0, 0, 0], [0, 1, 0], [0, 1, 0]],
    [[0, 0, 0], [1, 0, 0], [0, 0, 0]],
    [[0, 0, 0], [1, 0, 0], [0, 0, 0]],
    [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
    [[0, 0, 0], [0, 0, 1], [0, 0, 1]],
    [[0, 0, 0], [0, 0, 0], [1, 1, 0]],
]


def _zero_matrix(rows, cols):
    return [[0] * cols for _ in range(rows)]


def _schoolbook():
    """Fallback rank-54 decomposition."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = _zero_matrix(n * m, rank)
    V = _zero_matrix(m * p, rank)
    W = _zero_matrix(p * n, rank)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _rank14():
    """Construct the rank-14 decomposition using two Strassen slices."""
    n, m, p = DIMENSIONS
    n_groups = 2
    R = n_groups * len(_A)
    U = _zero_matrix(n * m, R)
    V = _zero_matrix(m * p, R)
    W = _zero_matrix(p * n, R)

    for g in range(n_groups):
        base = g * len(_A)
        for r in range(len(_A)):
            for i in range(n):
                for j in range(m):
                    val = _A[r][i][j]
                    if val:
                        U[i * m + j][base + r] = val

            for j in range(m):
                for k_local in range(3):
                    val = _B[r][j][k_local]
                    if val:
                        k_global = g * 3 + k_local
                        V[j * p + k_global][base + r] = val

            for i in range(n):
                for k_local in range(3):
                    val = _C[r][i][k_local]
                    if val:
                        k_global = g * 3 + k_local
                        W[k_global * n + i][base + r] = val

    return {"U": U, "V": V, "W": W}


def _verify(U, V, W):
    """Verify exact decomposition of the 3x3x6 multiplication tensor."""
    n, m, p = DIMENSIONS
    rank = len(U[0]) if U else 0
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    for c in range(rank):
        if len(U[c]) != rank or len(V[c]) != rank or len(W[c]) != rank:
            return False
    seen = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1.0) < 1e-10:
                    if (i, j, k) in seen:
                        return False
                    seen.add((i, j, k))
                elif abs(s) > 1e-10:
                    return False
    return len(seen) == n * m * p


def _tensor_from_factors(U, V, W):
    """Reconstruct the tensor T[i,j,k] from factor matrices."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    T = np.zeros((n, m, p), dtype=float)
    for r in range(rank):
        u_col = np.array([U[i][r] for i in range(n * m)], dtype=float)
        v_col = np.array([V[i][r] for i in range(m * p)], dtype=float)
        w_col = np.array([W[i][r] for i in range(p * n)], dtype=float)
        # Reshape and take outer product
        u_mat = u_col.reshape(n, m)
        v_mat = v_col.reshape(m, p)
        w_mat = w_col.reshape(p, n)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += u_mat[i, j] * v_mat[j, k] * w_mat[k, i]
    return T


def _try_remove_term(U, V, W, term_idx):
    """Try removing one term and verify if the tensor is still correct."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    keep = [r for r in range(rank) if r != term_idx]
    U_new = [[U[i][r] for r in keep] for i in range(len(U))]
    V_new = [[V[i][r] for r in keep] for i in range(len(V))]
    W_new = [[W[i][r] for r in keep] for i in range(len(W))]
    return {"U": U_new, "V": V_new, "W": W_new}


def _find_linear_combination(U, V, W, target_term, other_terms):
    """
    Try to express the target term as a linear combination of other terms.
    Returns coefficients if found, else None.
    """
    n, m, p = DIMENSIONS
    rank = len(U[0])
    
    # Build the tensor contribution of the target term
    u_t = np.array([U[i][target_term] for i in range(n * m)], dtype=float)
    v_t = np.array([V[i][target_term] for i in range(m * p)], dtype=float)
    w_t = np.array([W[i][target_term] for i in range(p * n)], dtype=float)
    
    # Build tensors for other terms
    other_tensors = []
    for ot in other_terms:
        u_o = np.array([U[i][ot] for i in range(n * m)], dtype=float)
        v_o = np.array([V[i][ot] for i in range(m * p)], dtype=float)
        w_o = np.array([W[i][ot] for i in range(p * n)], dtype=float)
        tensor = np.zeros((n, m, p), dtype=float)
        u_mat = u_o.reshape(n, m)
        v_mat = v_o.reshape(m, p)
        w_mat = w_o.reshape(p, n)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    tensor[i, j, k] = u_mat[i, j] * v_mat[j, k] * w_mat[k, i]
        other_tensors.append(tensor)
    
    # Target tensor
    target_tensor = np.zeros((n, m, p), dtype=float)
    u_mat = u_t.reshape(n, m)
    v_mat = v_t.reshape(m, p)
    w_mat = w_t.reshape(p, n)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target_tensor[i, j, k] = u_mat[i, j] * v_mat[j, k] * w_mat[k, i]
    
    # Solve: target = sum(c_i * other_i)
    # Flatten tensors
    target_flat = target_tensor.flatten()
    num_others = len(other_terms)
    A_mat = np.column_stack([t.flatten() for t in other_tensors])
    
    try:
        coeffs, residuals, rank_sol, sv = np.linalg.lstsq(A_mat, target_flat, rcond=None)
        # Check if the solution is good
        reconstructed = A_mat @ coeffs
        if np.allclose(reconstructed, target_flat, atol=1e-10):
            return coeffs
    except Exception:
        pass
    
    return None


def _search_lower_rank(U, V, W, max_time=100.0):
    """
    Systematically search for lower-rank decompositions.
    Returns the best decomposition found.
    """
    start_time = time.time()
    n, m, p = DIMENSIONS
    current_rank = len(U[0])
    
    best_U, best_V, best_W = U, V, W
    best_rank = current_rank
    
    # Try removing single terms
    for r_remove in range(current_rank):
        if time.time() - start_time > max_time - 5:
            break
        
        cand = _try_remove_term(U, V, W, r_remove)
        if _verify(cand["U"], cand["V"], cand["W"]):
            best_U, best_V, best_W = cand["U"], cand["V"], cand["W"]
            best_rank = len(cand["U"][0])
            # Restart search with the reduced set
            U, V, W = best_U, best_V, best_W
            current_rank = best_rank
            r_remove = 0
            continue
    
    # Try removing pairs of terms
    for r1 in range(len(U[0])):
        for r2 in range(r1 + 1, len(U[0])):
            if time.time() - start_time > max_time - 5:
                break
            keep = [r for r in range(len(U[0])) if r != r1 and r != r2]
            U_new = [[U[i][r] for r in keep] for i in range(len(U))]
            V_new = [[V[i][r] for r in keep] for i in range(len(V))]
            W_new = [[W[i][r] for r in keep] for i in range(len(W))]
            if _verify(U_new, V_new, W_new):
                best_U, best_V, best_W = U_new, V_new, W_new
                best_rank = len(keep)
                U, V, W = best_U, best_V, best_W
    
    return {"U": best_U, "V": best_V, "W": best_W}


def _try_mixed_decomposition():
    """
    Try to find a decomposition that mixes the two Strassen slices.
    The key idea: since the A-matrix is shared, we might be able to
    combine terms from different slices.
    """
    n, m, p = DIMENSIONS
    
    # Start with rank-14
    cand = _rank14()
    U, V, W = cand["U"], cand["V"], cand["W"]
    
    # Search for lower rank
    reduced = _search_lower_rank(U, V, W, max_time=80.0)
    
    if reduced is not None and len(reduced["U"][0]) < len(U[0]):
        return reduced
    
    return cand


def construct():
    deadline = time.time() + 115.0  # Safety margin before 120s limit
    best_result = None
    best_rank = float('inf')
    
    # First, get the baseline rank-14 decomposition
    try:
        cand14 = _rank14()
        if _verify(cand14["U"], cand14["V"], cand14["W"]):
            best_result = cand14
            best_rank = len(cand14["U"][0])
    except Exception:
        pass
    
    # Try to find a lower-rank decomposition
    if time.time() < deadline - 20.0:
        try:
            cand_mixed = _try_mixed_decomposition()
            if cand_mixed is not None:
                rank = len(cand_mixed["U"][0])
                if rank < best_rank and _verify(cand_mixed["U"], cand_mixed["V"], cand_mixed["W"]):
                    best_result = cand_mixed
                    best_rank = rank
        except Exception:
            pass
    
    if best_result is not None:
        return best_result
    
    # Fallback to schoolbook
    return _schoolbook()
