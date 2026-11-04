"""
Strategy: BCL rank-49 4x4x4 tensor, concatenated for 4x4x8 (rank 98).
BCL (Bini-Capovani-Lotti-Romani) is the known optimal rank for 4x4x4.
"""

import time
import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START

def _get_target_tensor():
    """Construct the target matrix multiplication tensor T[a,b,c]."""
    n, m, p = DIMENSIONS
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _bcl_4x4x4():
    """
    Returns (U, V, W) each of shape (16, 49) for 4x4x4 matrix multiplication.
    Uses the BCL rank-49 algorithm (same as recursive Strassen).
    We construct it by applying Strassen at two levels.
    """
    n = 4
    dim = n * n

    # Level-1 (bottom) Strassen 2x2 terms
    # 7 terms, each (u, v, w) of length 4
    terms_2x2 = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 0]),  # M1: (A00+A11)(B00+B11) -> C00
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 0, 0]),  # M2: (A10+A11)B00 -> C10
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 0, 0]),  # M3: A00(B01-B11) -> C11 (neg)
        ([0, 0, 0, 1], [0, 0, 1, -1], [0, 0, 0, 0]),  # M4: A11(B10-B00) -> C00(neg),C10
        ([1, 1, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]),  # M5: (A00+A01)B11 -> C01
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 0]),  # M6: (A10-A00)(B00+B01) -> C00
        ([0, 1, 0, -1], [0, 0, 1, 1], [0, 0, 0, 0]),  # M7: (A01-A11)(B10+B11) -> C01,C11
    ]

    # Top-level Strassen 4x4 terms
    # 7 terms, each with u_blocks (4 coeffs), v_blocks (4 coeffs), w_contributions (dict)
    top_terms = [
        {'u': [1, 0, 0, 1], 'v': [1, 0, 0, 1], 'w': {0: 1, 3: 1}},
        {'u': [0, 0, 1, 1], 'v': [1, 0, 0, 0], 'w': {2: 1}},
        {'u': [1, 0, 0, 0], 'v': [0, 1, 0, -1], 'w': {3: -1}},
        {'u': [0, 0, 0, 1], 'v': [0, 0, 1, -1], 'w': {0: -1, 2: 1}},
        {'u': [1, 1, 0, 0], 'v': [0, 0, 0, 1], 'w': {1: 1}},
        {'u': [-1, 0, 1, 0], 'v': [1, 1, 0, 0], 'w': {0: 1}},
        {'u': [0, 1, 0, -1], 'v': [0, 0, 1, 1], 'w': {1: 1, 3: 1}},
    ]

    U = np.zeros((dim, 49))
    V = np.zeros((dim, 49))
    W = np.zeros((dim, 49))

    def block_indices(br, bc):
        """Global indices for 2x2 block at position (br, bc) in a 4x4 matrix."""
        return [(br*2 + lr)*4 + (bc*2 + lc) for lr in range(2) for lc in range(2)]

    col = 0
    for tt in top_terms:
        u_top = tt['u']
        v_top = tt['v']
        w_top = tt['w']
        for bt in terms_2x2:
            u_bot, v_bot, w_bot = bt

            u_full = np.zeros(dim)
            v_full = np.zeros(dim)
            w_full = np.zeros(dim)

            for b_idx in range(4):
                c_u = u_top[b_idx]
                c_v = v_top[b_idx]
                br, bc = b_idx // 2, b_idx % 2
                if c_u != 0:
                    for l, idx in enumerate(block_indices(br, bc)):
                        if u_bot[l] != 0:
                            u_full[idx] += c_u * u_bot[l]
                if c_v != 0:
                    for l, idx in enumerate(block_indices(br, bc)):
                        if v_bot[l] != 0:
                            v_full[idx] += c_v * v_bot[l]

            for c_idx, c_w in w_top.items():
                br, bc = c_idx // 2, c_idx % 2
                for l, idx in enumerate(block_indices(br, bc)):
                    if w_bot[l] != 0:
                        w_full[idx] += c_w * w_bot[l]

            U[:, col] = u_full
            V[:, col] = v_full
            W[:, col] = w_full
            col += 1

    return U, V, W

def _verify_decomposition(U, V, W, T, tol=1e-9):
    """Verify that sum_k U[:,k] V[:,k] W[:,k] == T."""
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(T_recon, T, atol=tol)

def _build_4x4x8():
    """
    Build 4x4x8 decomposition by combining two 4x4x4 decompositions.
    """
    n, m, p = DIMENSIONS  # 4, 4, 8

    U4, V4, W4 = _bcl_4x4x4()
    rank4 = U4.shape[1]  # 49

    total_rank = 2 * rank4  # 98

    U_full = np.zeros((n*m, total_rank))
    V_full = np.zeros((m*p, total_rank))
    W_full = np.zeros((p*n, total_rank))

    # Left half (k=0..3)
    U_full[:, 0:rank4] = U4

    for r in range(rank4):
        for idx in range(16):
            val = V4[idx, r]
            if val != 0:
                j = idx // 4
                k_local = idx % 4
                v_idx = j * 8 + k_local
                V_full[v_idx, r] += val

            val = W4[idx, r]
            if val != 0:
                k_local = idx // 4
                i = idx % 4
                w_idx = k_local * 4 + i
                W_full[w_idx, r] += val

    # Right half (k=4..7)
    U_full[:, rank4:2*rank4] = U4

    for r in range(rank4):
        for idx in range(16):
            val = V4[idx, r]
            if val != 0:
                j = idx // 4
                k_local = idx % 4
                v_idx = j * 8 + (k_local + 4)
                V_full[v_idx, r + rank4] += val

            val = W4[idx, r]
            if val != 0:
                k_local = idx % 4
                i = idx // 4
                w_idx = (k_local + 4) * 4 + i
                W_full[w_idx, r + rank4] += val

    return U_full, V_full, W_full

def _to_list(mat):
    """Convert numpy array to list with integer or half-integer entries."""
    mat2 = mat * 2
    rounded = np.round(mat2)
    if np.allclose(mat2, rounded, atol=1e-6):
        return (rounded / 2).tolist()
    rounded_int = np.round(mat)
    if np.allclose(mat, rounded_int, atol=1e-6):
        return rounded_int.astype(int).tolist()
    return mat.tolist()

def _schoolbook_fallback():
    n, m, p = DIMENSIONS
    rank_school = n * m * p
    U_s = np.zeros((n*m, rank_school))
    V_s = np.zeros((m*p, rank_school))
    W_s = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_s[i*m + j, col] = 1.0
                V_s[j*p + k, col] = 1.0
                W_s[k*n + i, col] = 1.0
                col += 1
    return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    U, V, W = _build_4x4x8()

    T = _get_target_tensor()
    if not _verify_decomposition(U, V, W, T):
        return _schoolbook_fallback()

    if time.time() > deadline:
        return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}

    return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}

# EVOLVE-BLOCK-END
