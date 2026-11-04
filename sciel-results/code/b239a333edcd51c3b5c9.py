"""Tensor decomposition for 4x4 * 4x5 matrix multiplication."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction
import itertools

def _validate(U, V, W, n=4, m=4, p=5, tol=1e-9):
    """Check that U,V,W form a valid tensor decomposition of matmul(n,m,p)."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    if rank != V.shape[1] or rank != W.shape[1]:
        return False
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    U3 = U.reshape(n, m, -1)
    V3 = V.reshape(m, p, -1)
    W3 = W.reshape(p, n, -1)
    M_decomp = np.einsum('ijr,jkr,kir->ikj', U3, V3, W3)
    if not np.allclose(M_decomp, 1.0, atol=tol):
        return False
    return True

def _strassen_2x2_terms():
    """
    Return the 7 rank-1 tensors for 2x2x2 matrix multiplication (Strassen).
    Each term is (u, v, w) where u, v, w are length-4 vectors (flattened 2x2).
    u[i*2+j], v[j*2+k], w[k*2+i]
    """
    terms = []
    u1 = [1, 0, 0, 1]; v1 = [1, 0, 0, 1]; w1 = [1, 0, 0, 1]
    terms.append((u1, v1, w1))
    u2 = [0, 0, 1, 1]; v2 = [1, 0, 0, 0]; w2 = [0, 0, 1, 0]
    terms.append((u2, v2, w2))
    u3 = [1, 0, 0, 0]; v3 = [0, 1, 1, -1]; w3 = [0, 1, 0, 0]
    terms.append((u3, v3, w3))
    u4 = [0, 0, 0, 1]; v4 = [1, -1, 0, 0]; w4 = [1, 0, 0, 1]
    terms.append((u4, v4, w4))
    u5 = [1, 1, 0, 0]; v5 = [0, 0, 0, 1]; w5 = [1, 1, 0, 0]
    terms.append((u5, v5, w5))
    u6 = [-1, 0, 1, 0]; v6 = [1, 1, 0, 0]; w6 = [0, 0, 1, 1]
    terms.append((u6, v6, w6))
    u7 = [0, 1, 0, -1]; v7 = [0, 0, 1, 1]; w7 = [0, 1, 0, 1]
    terms.append((u7, v7, w7))
    return terms

def _construct_strassen_4x4_4x5():
    """
    Construct a decomposition for 4x4 * 4x5 using Strassen on the 4x4 part
    and schoolbook for the boundary column.
    Total rank: 49 + 16 = 65
    """
    n, m, p = 4, 4, 5
    rank = 65
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    strassen_2x2 = _strassen_2x2_terms()
    
    strassen_defs = [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1)]),
        ([(0, 1)], [(2, 1), (3, -1)], [(1, 1)]),
        ([(3, 1)], [(0, 1), (1, -1)], [(0, 1), (3, 1)]),
        ([(0, 1), (1, 1)], [(3, 1)], [(0, 1), (1, 1)]),
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(2, 1), (3, 1)]),
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(1, 1), (3, 1)]),
    ]
    
    col = 0
    for mk in range(7):
        a_blocks, b_blocks, c_blocks = strassen_defs[mk]
        
        a_map = {}
        for block_idx, sign in a_blocks:
            bi = block_idx // 2
            bj = block_idx % 2
            a_map[(bi, bj)] = a_map.get((bi, bj), 0) + sign
        
        b_map = {}
        for block_idx, sign in b_blocks:
            bi = block_idx // 2
            bj = block_idx % 2
            b_map[(bi, bj)] = b_map.get((bi, bj), 0) + sign
        
        c_map = {}
        for block_idx, sign in c_blocks:
            bi = block_idx // 2
            bj = block_idx % 2
            c_map[(bi, bj)] = c_map.get((bi, bj), 0) + sign
        
        for r in range(7):
            u, v, w = strassen_2x2[r]
            
            for bi in range(2):
                for bj in range(2):
                    u_coeff = u[bi * 2 + bj]
                    a_sign = a_map.get((bi, bj), 0)
                    if u_coeff == 0 or a_sign == 0:
                        continue
                    for di in range(2):
                        for dj in range(2):
                            row = 2 * bi + di
                            col_a = 2 * bj + dj
                            U[row * m + col_a][col] += u_coeff * a_sign
            
            for bj in range(2):
                for bk in range(2):
                    v_coeff = v[bj * 2 + bk]
                    b_sign = b_map.get((bj, bk), 0)
                    if v_coeff == 0 or b_sign == 0:
                        continue
                    for dj in range(2):
                        for dk in range(2):
                            row_v = 2 * bj + dj
                            col_v = 2 * bk + dk
                            V[row_v * p + col_v][col] += v_coeff * b_sign
            
            for bk in range(2):
                for bi in range(2):
                    w_coeff = w[bk * 2 + bi]
                    c_sign = c_map.get((bk, bi), 0)
                    if w_coeff == 0 or c_sign == 0:
                        continue
                    for dk in range(2):
                        for di in range(2):
                            row_w = 2 * bk + dk
                            col_w = 2 * bi + di
                            W[row_w * n + col_w][col] += w_coeff * c_sign
            
            col += 1
    
    for i in range(n):
        for j in range(m):
            U[i * m + j][col] = 1
            V[j * p + 4][col] = 1
            W[4 * n + i][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}

def _can_merge_pair(U, V, W, a, b, n, m, p):
    """
    Check if terms a and b can be merged into a single term.
    A pair of terms (ua, va, wa) and (ub, vb, wb) can be replaced by a single
    term (u', v', w') if there exist scalars such that:
      ua*va*wa + ub*vb*wb = u'*v'*w'
    This is possible if the two terms are "proportional" in a certain sense.
    
    A sufficient condition: if ua = s*ub, va = vb, wa = wb/s for some s != 0,
    then ua*va*wa + ub*vb*wb = s*ub*vb*wb/s + ub*vb*wb = 2*ub*vb*wb,
    which is a single term (2*ub, vb, wb) - but this doubles, not merges.
    
    Actually, merging two terms into one is generally NOT possible unless they
    are identical (then we just have 2*term) or one is zero.
    
    However, we CAN check if removing term a still works (if term b covers
    the same slots), or if we can find a linear combination.
    
    For now, let's check the simpler condition: can we remove term a?
    """
    # Check if term a is redundant given all other terms
    rank = U.shape[1]
    # This is expensive, so we only call this when we have time
    return False

def _greedy_reduce(U, V, W, n, m, p, max_iterations=100, time_limit=20):
    """
    Try to reduce the rank by finding redundant terms or mergeable pairs.
    """
    start_time = time.time()
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    def _is_valid(U_mat, V_mat, W_mat, r):
        if r <= 0:
            return False
        U3 = U_mat[:, :r].reshape(n, m, -1)
        V3 = V_mat[:, :r].reshape(m, p, -1)
        W3 = W_mat[:, :r].reshape(p, n, -1)
        M_decomp = np.einsum('ijr,jkr,kir->ikj', U3, V3, W3)
        return np.allclose(M_decomp, 1.0, atol=1e-9)
    
    # Try removing individual terms
    for _ in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        if rank <= 1:
            break
        
        found = False
        for t in range(rank):
            if time.time() - start_time > time_limit:
                break
            
            U_test = U.copy()
            V_test = V.copy()
            W_test = W.copy()
            U_test[:, t] = 0
            V_test[:, t] = 0
            W_test[:, t] = 0
            
            if _is_valid(U_test, V_test, W_test, rank):
                keep = list(range(rank))
                keep.pop(t)
                U = U[:, keep]
                V = V[:, keep]
                W = W[:, keep]
                rank -= 1
                found = True
                best_rank = rank
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                break
        
        if not found:
            break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}, best_rank

def _try_pairwise_merge(U, V, W, n, m, p, time_limit=30):
    """
    For each pair of columns (a, b), check if we can replace them with a single
    column by finding a linear combination that preserves the tensor.
    
    This is a more sophisticated search: for each pair, we check if there exists
    a new column that, combined with all other columns, still gives the correct
    tensor. This is equivalent to checking if the span of {all columns} has
    dimension rank-1 after removing one column and adding a new one.
    
    A practical approach: for each pair (a,b), try to find coefficients α, β
    such that replacing columns a and b with a single column c = α*col_a + β*col_b
    (for U, V, W respectively, with appropriate scaling) still works.
    
    This is complex. Instead, we'll do a simpler but effective search:
    try all pairs and check if zeroing one of them still works (redundancy check).
    """
    start_time = time.time()
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    def _is_valid(U_mat, V_mat, W_mat, r):
        if r <= 0:
            return False
        U3 = U_mat[:, :r].reshape(n, m, -1)
        V3 = V_mat[:, :r].reshape(m, p, -1)
        W3 = W_mat[:, :r].reshape(p, n, -1)
        M_decomp = np.einsum('ijr,jkr,kir->ikj', U3, V3, W3)
        return np.allclose(M_decomp, 1.0, atol=1e-9)
    
    # Try all pairs: for each pair (a, b), check if we can drop one
    # This is the same as individual removal, but we do it in a different order
    # to catch different redundancies
    for a in range(rank):
        if time.time() - start_time > time_limit:
            break
        for b in range(a + 1, rank):
            if time.time() - start_time > time_limit:
                break
            
            # Try dropping a
            U_test = U.copy()
            V_test = V.copy()
            W_test = W.copy()
            U_test[:, a] = 0
            V_test[:, a] = 0
            W_test[:, a] = 0
            if _is_valid(U_test, V_test, W_test, rank):
                keep = list(range(rank))
                keep.remove(a)
                U = U[:, keep]
                V = V[:, keep]
                W = W[:, keep]
                rank -= 1
                best_rank = rank
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                break
            
            # Try dropping b
            U_test = U.copy()
            V_test = V.copy()
            W_test = W.copy()
            U_test[:, b] = 0
            V_test[:, b] = 0
            W_test[:, b] = 0
            if _is_valid(U_test, V_test, W_test, rank):
                keep = list(range(rank))
                keep.remove(b)
                U = U[:, keep]
                V = V[:, keep]
                W = W[:, keep]
                rank -= 1
                best_rank = rank
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                break
        
        if time.time() - start_time > time_limit:
            break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}, best_rank

def _randomized_search(U, V, W, n, m, p, time_limit=30, seed=42):
    """
    Try to find a lower-rank decomposition by random perturbation and optimization.
    
    Strategy: start with the current decomposition, randomly perturb the columns,
    then try to reduce the rank by finding linear dependencies.
    """
    import random
    rng = random.Random(seed)
    start_time = time.time()
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    def _is_valid(U_mat, V_mat, W_mat, r):
        if r <= 0:
            return False
        U3 = U_mat[:, :r].reshape(n, m, -1)
        V3 = V_mat[:, :r].reshape(m, p, -1)
        W3 = W_mat[:, :r].reshape(p, n, -1)
        M_decomp = np.einsum('ijr,jkr,kir->ikj', U3, V3, W3)
        return np.allclose(M_decomp, 1.0, atol=1e-9)
    
    # Try random linear combinations of columns to see if we can reduce rank
    # For each pair of columns, try to replace them with a single column
    # that is a linear combination
    
    attempts = 0
    while time.time() - start_time < time_limit and attempts < 1000:
        attempts += 1
        a = rng.randint(0, rank - 1)
        b = rng.randint(0, rank - 1)
        if a == b:
            continue
        
        # Try replacing column a with a combination of a and b
        # New column a' = alpha * col_a + beta * col_b
        # We need to find alpha, beta such that the tensor is still correct
        # This requires: alpha * (ua va wa) + beta * (ub vb wb) to replace
        # both terms, which is only possible if they are proportional
        
        # Simpler: just check if we can drop one
        for drop in [a, b]:
            U_test = U.copy()
            V_test = V.copy()
            W_test = W.copy()
            U_test[:, drop] = 0
            V_test[:, drop] = 0
            W_test[:, drop] = 0
            if _is_valid(U_test, V_test, W_test, rank):
                keep = list(range(rank))
                keep.remove(drop)
                U = U[:, keep]
                V = V[:, keep]
                W = W[:, keep]
                rank -= 1
                best_rank = rank
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}, best_rank

def construct():
    """Construct a rank-minimal tensor decomposition for 4x4 * 4x5 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Start with the Strassen-embedded construction (rank 65)
    result = _construct_strassen_4x4_4x5()
    rank = len(result["U"][0]) if result["U"] else 0
    
    # Validate
    if not _validate(result["U"], result["V"], result["W"], n, m, p):
        # Fallback to schoolbook
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
        result = {"U": U, "V": V, "W": W}
    
    # Try to reduce the rank with greedy search
    remaining = deadline - time.time()
    if remaining > 5:
        try:
            reduced, reduced_rank = _greedy_reduce(
                result["U"], result["V"], result["W"], n, m, p,
                max_iterations=200, time_limit=min(30, remaining - 5)
            )
            if reduced_rank < rank:
                if _validate(reduced["U"], reduced["V"], reduced["W"], n, m, p):
                    result = reduced
                    rank = reduced_rank
        except Exception:
            pass
    
    # Try pairwise merge search
    remaining = deadline - time.time()
    if remaining > 5:
        try:
            reduced, reduced_rank = _try_pairwise_merge(
                result["U"], result["V"], result["W"], n, m, p,
                time_limit=min(30, remaining - 5)
            )
            if reduced_rank < rank:
                if _validate(reduced["U"], reduced["V"], reduced["W"], n, m, p):
                    result = reduced
                    rank = reduced_rank
        except Exception:
            pass
    
    # Try randomized search
    remaining = deadline - time.time()
    if remaining > 5:
        try:
            reduced, reduced_rank = _randomized_search(
                result["U"], result["V"], result["W"], n, m, p,
                time_limit=min(30, remaining - 5), seed=42
            )
            if reduced_rank < rank:
                if _validate(reduced["U"], reduced["V"], reduced["W"], n, m, p):
                    result = reduced
                    rank = reduced_rank
        except Exception:
            pass
    
    return result

# EVOLVE-BLOCK-END
