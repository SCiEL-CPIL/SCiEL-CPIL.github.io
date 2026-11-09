"""Rank-minimal tensor decomposition for 4x4 @ 4x9 multiplication.

Strategy:
1. Build a known valid decomposition using Strassen(2x2)⊗Strassen(2x2) for
   4x4x4 (rank 49) and schoolbook for the remaining columns.
   Use 4+4+1 column split: total rank = 49 + 49 + 16 = 114.
2. Attempt to reduce rank via SVD-based linear dependency detection.
"""

import numpy as np
import time

DIMENSIONS = [4, 4, 9]

def _strassen_2x2():
    """Return the 7 Strassen terms for 2x2 @ 2x2.
    
    Each term is (U_row, V_row, W_row) where:
    - U_row is length 4 (flattened 2x2 A matrix, row-major)
    - V_row is length 4 (flattened 2x2 B matrix, row-major)  
    - W_row is length 4 (flattened 2x2 C matrix, row-major)
    """
    terms = []
    
    # M1 = (A11+A22)*(B11+B22)
    terms.append((np.array([1,0,0,1]), np.array([1,0,0,1]), np.array([1,1,0,1])))
    # M2 = (A21+A22)*B11
    terms.append((np.array([0,0,1,1]), np.array([1,0,0,0]), np.array([0,0,1,0])))
    # M3 = A11*(B12-B22)
    terms.append((np.array([1,0,0,0]), np.array([0,1,0,-1]), np.array([0,1,0,1])))
    # M4 = A22*(B21-B11)
    terms.append((np.array([0,0,0,1]), np.array([-1,0,1,0]), np.array([1,0,1,0])))
    # M5 = (A11+A12)*B22
    terms.append((np.array([1,1,0,0]), np.array([0,0,0,1]), np.array([-1,1,0,0])))
    # M6 = (A21-A11)*(B11+B12)
    terms.append((np.array([-1,0,1,0]), np.array([1,1,0,0]), np.array([0,0,0,1])))
    # M7 = (A12-A22)*(B21+B22)
    terms.append((np.array([0,1,0,-1]), np.array([0,0,1,1]), np.array([1,0,0,0])))
    
    return terms

def _build_4x4x4_strassen():
    """Build Strassen-based decomposition for 4x4 @ 4x4 -> 4x4.
    
    Rank = 7*7 = 49.
    """
    s_terms = _strassen_2x2()
    rank = 7 * 7
    U = np.zeros((16, rank))
    V = np.zeros((16, rank))
    W = np.zeros((16, rank))
    
    col = 0
    for s1 in range(7):
        for s2 in range(7):
            u1, v1, w1 = s_terms[s1]
            u2, v2, w2 = s_terms[s2]
            
            for b in range(4):
                for p in range(4):
                    block_row = b // 2
                    block_col = b % 2
                    inner_row = p // 2
                    inner_col = p % 2
                    i = block_row * 2 + inner_row
                    j = block_col * 2 + inner_col
                    full_idx = i * 4 + j
                    
                    U[full_idx, col] = u1[b] * u2[p]
                    V[full_idx, col] = v1[b] * v2[p]
                    W[full_idx, col] = w1[b] * w2[p]
            
            col += 1
    
    return U, V, W, rank

def _build_4x4x1_schoolbook():
    """Build schoolbook decomposition for 4x4 @ 4x1 -> 4x1.
    Rank = 16.
    """
    n, m, p = 4, 4, 1
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return U, V, W, rank

def _build_4x4x5_schoolbook():
    """Build schoolbook decomposition for 4x4 @ 4x5 -> 4x5.
    Rank = 80.
    """
    n, m, p = 4, 4, 5
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return U, V, W, rank

def _verify_decomposition(U, V, W, n, m, p, A_test, B_test, C_expected):
    """Verify that the decomposition correctly computes C = A @ B."""
    D = np.zeros((n, p))
    rank = U.shape[1]
    for r in range(rank):
        a_r = np.sum(U[:, r].reshape(n, m) * A_test)
        b_r = np.sum(V[:, r].reshape(m, p) * B_test)
        w_vec = W[:, r].reshape(p, n)  # W[k*n+i]
        D += a_r * b_r * w_vec.T
    err = np.max(np.abs(D - C_expected))
    return err

def _try_reduce_rank(U, V, W, n, m, p, A_test, B_test, C_expected, max_iter=200, tol=1e-8):
    """Attempt to reduce rank by finding linear dependencies among the factor vectors.
    
    We look for columns of the combined factor matrix that can be expressed
    as linear combinations of other columns, while preserving the tensor identity.
    
    The key insight: if we have factors (U_r, V_r, W_r) for r=1..R, and we find
    a linear relation sum_r c_r * (U_r, V_r, W_r) = 0 in the "tensor sense",
    we can eliminate one factor.
    
    A practical approach: treat each factor as a vector in R^(nm + mp + pn) and
    look for the rank of this combined matrix. If rank < R, we can reduce.
    """
    R = U.shape[1]
    combined = np.vstack([U, V, W])  # shape (nm+mp+pn, R)
    
    # Compute SVD to find the numerical rank
    try:
        U_svd, s_vals, Vt = np.linalg.svd(combined, full_matrices=False)
    except np.linalg.LinAlgError:
        return U, V, W, R
    
    # Find the numerical rank
    num_rank = np.sum(s_vals > tol * s_vals[0])
    
    if num_rank >= R:
        return U, V, W, R
    
    # We can potentially reduce to num_rank factors
    # Use the top num_rank right singular vectors as the new basis
    new_R = int(num_rank)
    
    # Transform the factors to the new basis
    # Vt[:new_R] gives the new basis in the factor space
    # We need to express the original factors in terms of this basis
    
    # Actually, a cleaner approach: if the combined matrix has rank < R,
    # there exist non-trivial linear relations among the columns.
    # Let's find a null space vector and use it to eliminate one factor.
    
    # For simplicity, let's try a greedy approach:
    # Try removing each factor and see if we can still satisfy the identity
    # with a reweighting of the remaining factors.
    
    # This is complex, so for now just return the original if no clear reduction.
    # A more sophisticated approach would be needed for actual rank reduction.
    
    return U, V, W, R

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 9
    deadline = time.time() + 145
    
    # Build the 4+4+1 split decomposition: rank 49 + 49 + 16 = 114
    U_444a, V_444a, W_444a, r_444a = _build_4x4x4_strassen()
    U_444b, V_444b, W_444b, r_444b = _build_4x4x4_strassen()
    U_441, V_441, W_441, r_441 = _build_4x4x1_schoolbook()
    
    total_rank = r_444a + r_444b + r_441  # 49 + 49 + 16 = 114
    
    U_full = np.zeros((16, total_rank))
    V_full = np.zeros((36, total_rank))
    W_full = np.zeros((36, total_rank))
    
    # First 4x4x4: columns 0-3
    U_full[:, :r_444a] = U_444a
    for j in range(4):
        for k in range(4):
            V_full[j * 9 + k, :r_444a] = V_444a[j * 4 + k, :]
    for k in range(4):
        for i in range(4):
            W_full[k * 4 + i, :r_444a] = W_444a[k * 4 + i, :]
    
    # Second 4x4x4: columns 4-7
    U_full[:, r_444a:r_444a+r_444b] = U_444b
    for j in range(4):
        for k in range(4):
            V_full[j * 9 + (k + 4), r_444a:r_444a+r_444b] = V_444b[j * 4 + k, :]
    for k in range(4):
        for i in range(4):
            W_full[(k + 4) * 4 + i, r_444a:r_444a+r_444b] = W_444b[k * 4 + i, :]
    
    # 4x4x1: column 8
    U_full[:, r_444a+r_444b:] = U_441
    for j in range(4):
        V_full[j * 9 + 8, r_444a+r_444b:] = V_441[j * 1 + 0, :]
    for i in range(4):
        W_full[8 * 4 + i, r_444a+r_444b:] = W_441[0 * 4 + i, :]
    
    # Verify
    A_test = np.random.randn(4, 4)
    B_test = np.random.randn(4, 9)
    C_expected = A_test @ B_test
    
    err = _verify_decomposition(U_full, V_full, W_full, n, m, p, A_test, B_test, C_expected)
    
    if err > 1e-9:
        # Fall back to pure schoolbook
        rank = n * m * p
        U_mat = np.zeros((n * m, rank))
        V_mat = np.zeros((m * p, rank))
        W_mat = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_mat[i * m + j, col] = 1
                    V_mat[j * p + k, col] = 1
                    W_mat[k * n + i, col] = 1
                    col += 1
        return {"U": U_mat.tolist(), "V": V_mat.tolist(), "W": W_mat.tolist()}
    
    # Try to reduce rank
    if time.time() < deadline - 30:
        U_full, V_full, W_full, total_rank = _try_reduce_rank(
            U_full, V_full, W_full, n, m, p, A_test, B_test, C_expected
        )
        if total_rank < 114:
            # Re-verify after reduction
            err2 = _verify_decomposition(U_full, V_full, W_full, n, m, p, A_test, B_test, C_expected)
            if err2 > 1e-9:
                # Reduction failed, revert
                U_full, V_full, W_full, total_rank = _build_4x4x4_strassen()[0], None, None, 0
                # Rebuild the 114-rank decomposition
                U_444a, V_444a, W_444a, r_444a = _build_4x4x4_strassen()
                U_444b, V_444b, W_444b, r_444b = _build_4x4x4_strassen()
                U_441, V_441, W_441, r_441 = _build_4x4x1_schoolbook()
                total_rank = r_444a + r_444b + r_441
                U_full = np.zeros((16, total_rank))
                V_full = np.zeros((36, total_rank))
                W_full = np.zeros((36, total_rank))
                U_full[:, :r_444a] = U_444a
                for j in range(4):
                    for k in range(4):
                        V_full[j * 9 + k, :r_444a] = V_444a[j * 4 + k, :]
                for k in range(4):
                    for i in range(4):
                        W_full[k * 4 + i, :r_444a] = W_444a[k * 4 + i, :]
                U_full[:, r_444a:r_444a+r_444b] = U_444b
                for j in range(4):
                    for k in range(4):
                        V_full[j * 9 + (k + 4), r_444a:r_444a+r_444b] = V_444b[j * 4 + k, :]
                for k in range(4):
                    for i in range(4):
                        W_full[(k + 4) * 4 + i, r_444a:r_444a+r_444b] = W_444b[k * 4 + i, :]
                U_full[:, r_444a+r_444b:] = U_441
                for j in range(4):
                    V_full[j * 9 + 8, r_444a+r_444b:] = V_441[j * 1 + 0, :]
                for i in range(4):
                    W_full[8 * 4 + i, r_444a+r_444b:] = W_441[0 * 4 + i, :]
    
    return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
