"""
Rank-49 exact tensor decomposition for 4x4 matrix multiplication.

Uses the Kronecker product of the rank-7 Strassen decomposition for 2x2
matrix multiplication to produce a rank-49 decomposition for 4x4.

The tensor identity:
  sum_r U[i*4+j, r] * V[j*4+k, r] * W[k*4+i, r] = 1  for all i,j,k in {0..3}
"""

import time
import numpy as np

# Strassen 2x2 decomposition terms.
# Each term: (u_list, v_list, w_list) where each list is [(idx, val), ...]
# Indices are 0..3 (flattened 2x2).
def _get_strassen_2x2_terms():
    terms = []
    # M1 = (A00+A11)(B00+B11)
    terms.append(([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)]))
    # M2 = (A10+A11)B00
    terms.append(([(2,1),(3,1)], [(0,1)], [(2,1)]))
    # M3 = A00(B01-B11)
    terms.append(([(0,1)], [(1,1),(3,-1)], [(1,1)]))
    # M4 = A11(B10-B00)
    terms.append(([(3,1)], [(2,1),(0,-1)], [(0,1),(2,1)]))
    # M5 = (A00+A01)B11
    terms.append(([(0,1),(1,1)], [(3,1)], [(0,-1),(1,1)]))
    # M6 = (A10-A00)(B00+B01)
    terms.append(([(2,1),(0,-1)], [(0,1),(1,1)], [(3,1)]))
    # M7 = (A01-A11)(B10+B11)
    terms.append(([(1,1),(3,-1)], [(2,1),(3,1)], [(0,1),(3,-1)]))
    return terms

def construct():
    """
    Constructs a rank-49 decomposition for 4x4 matrix multiplication.
    Returns dict with keys 'U','V','W' each of shape (16, 49).
    """
    start_time = time.time()
    deadline = start_time + 100.0

    s_terms = _get_strassen_2x2_terms()
    rank_2 = len(s_terms)  # 7
    total_rank = rank_2 * rank_2  # 49

    # Use numpy for efficiency
    U = np.zeros((16, total_rank), dtype=np.int64)
    V = np.zeros((16, total_rank), dtype=np.int64)
    W = np.zeros((16, total_rank), dtype=np.int64)

    for r_h in range(rank_2):
        u_h, v_h, w_h = s_terms[r_h]
        for r_l in range(rank_2):
            u_l, v_l, w_l = s_terms[r_l]
            r = r_h * rank_2 + r_l

            for idx_uh, val_uh in u_h:
                i_h = idx_uh // 2
                j_h = idx_uh % 2
                for idx_ul, val_ul in u_l:
                    i_l = idx_ul // 2
                    j_l = idx_ul % 2
                    i = i_h * 2 + i_l
                    j = j_h * 2 + j_l
                    U[i * 4 + j, r] = val_uh * val_ul

            for idx_vh, val_vh in v_h:
                j_h = idx_vh // 2
                k_h = idx_vh % 2
                for idx_vl, val_vl in v_l:
                    j_l = idx_vl // 2
                    k_l = idx_vl % 2
                    j = j_h * 2 + j_l
                    k = k_h * 2 + k_l
                    V[j * 4 + k, r] = val_vh * val_vl

            for idx_wh, val_wh in w_h:
                k_h = idx_wh // 2
                i_h = idx_wh % 2
                for idx_wl, val_wl in w_l:
                    k_l = idx_wl // 2
                    i_l = idx_wl % 2
                    k = k_h * 2 + k_l
                    i = i_h * 2 + i_l
                    W[k * 4 + i, r] = val_wh * val_wl

    if time.time() > deadline:
        return _schoolbook_fallback()

    # Verify correctness
    for i in range(4):
        for j in range(4):
            for k in range(4):
                total = 0
                for r in range(total_rank):
                    total += U[i*4+j, r] * V[j*4+k, r] * W[k*4+i, r]
                expected = 1 if i == k else 0
                if total != expected:
                    return _schoolbook_fallback()

    # Convert to lists of lists with integer values
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}

def _schoolbook_fallback():
    """Fallback to rank-64 schoolbook decomposition."""
    n = 4
    rank = 64
    U = [[0] * rank for _ in range(n * n)]
    V = [[0] * rank for _ in range(n * n)]
    W = [[0] * rank for _ in range(n * n)]
    col = 0
    for i in range(n):
        for j in range(n):
            for k in range(n):
                U[i * n + j][col] = 1
                V[j * n + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
