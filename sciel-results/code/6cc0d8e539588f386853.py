import time
import sys

_START_TIME = time.time()
TIME_LIMIT = 150.0

def construct():
    """
    Returns a rank-65 exact tensor decomposition for 4x4 * 4x5 matrix multiplication.
    Uses double-Strassen (rank 49) for the 4x4x4 part + schoolbook (rank 16) for 4x4x1.
    """
    return _construct_rank_65()

def _construct_rank_65():
    """
    Rank-65 decomposition: double-Strassen (49) for 4x4x4 + schoolbook (16) for 4x4x1.
    """
    n, m, p = 4, 4, 5
    rank = 65
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen parameters for 2x2 block decomposition
    # Each entry: (p_coeffs, q_coeffs, c_blocks)
    # p_coeffs: [A00, A01, A10, A11] coefficients for the super-term
    # q_coeffs: [B00, B01, B10, B11] coefficients for the super-term
    # c_blocks: list of (block_row, block_col, sign) for C contributions
    
    strassen_s = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [(0, 0, 1), (1, 1, 1)]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [(1, 0, 1), (1, 1, -1)]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [(0, 1, 1), (1, 1, 1)]),
        ([0, 0, 0, 1], [-1, 0, 1, 0], [(0, 0, 1), (1, 0, 1)]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [(0, 0, -1), (0, 1, 1)]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [(1, 1, 1)]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [(0, 0, 1)]),
    ]
    
    # Inner Strassen: each entry is (p_inner, q_inner, c_inner)
    # p_inner: list of (local_elem, coeff) for 2x2 A block
    # q_inner: list of (local_elem, coeff) for 2x2 B block
    # c_inner: list of (local_row, local_col, coeff) for 2x2 C block
    
    strassen_i = [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 0, 1), (1, 1, 1)]),
        ([(2, 1), (3, 1)], [(0, 1)], [(1, 0, 1), (1, 1, -1)]),
        ([(0, 1)], [(1, 1), (3, -1)], [(0, 1, 1), (1, 1, 1)]),
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 0, 1), (1, 0, 1)]),
        ([(0, 1), (1, 1)], [(3, 1)], [(0, 0, -1), (0, 1, 1)]),
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(1, 1, 1)]),
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 0, 1)]),
    ]
    
    for s in range(7):
        p_coeffs, q_coeffs, c_blocks = strassen_s[s]
        for i in range(7):
            p_inner, q_inner, c_inner = strassen_i[i]
            
            # Build U contributions
            for p_elem, p_coeff in p_inner:
                r = p_elem // 2
                c = p_elem % 2
                for b_idx, b_coeff in enumerate(p_coeffs):
                    if b_coeff == 0:
                        continue
                    if b_idx == 0:
                        gi, gj = r, c
                    elif b_idx == 1:
                        gi, gj = r, 2 + c
                    elif b_idx == 2:
                        gi, gj = 2 + r, c
                    else:
                        gi, gj = 2 + r, 2 + c
                    coeff = p_coeff * b_coeff
                    if coeff != 0:
                        U[gi * m + gj][col] += coeff
            
            # Build V contributions (only for first 4 columns of B)
            for q_elem, q_coeff in q_inner:
                r = q_elem // 2
                c = q_elem % 2
                for b_idx, b_coeff in enumerate(q_coeffs):
                    if b_coeff == 0:
                        continue
                    if b_idx == 0:
                        gj, gk = r, c
                    elif b_idx == 1:
                        gj, gk = r, 2 + c
                    elif b_idx == 2:
                        gj, gk = 2 + r, c
                    else:
                        gj, gk = 2 + r, 2 + c
                    if gk < 4:
                        coeff = q_coeff * b_coeff
                        if coeff != 0:
                            V[gj * p + gk][col] += coeff
            
            # Build W contributions
            for br, bc, sign in c_blocks:
                for lr, lc, local_coeff in c_inner:
                    gi = br * 2 + lr
                    gk = bc * 2 + lc
                    coeff = sign * local_coeff
                    if coeff != 0:
                        W[gk * n + gi][col] += coeff
            
            col += 1
    
    # Schoolbook for 4x4 * 4x1 (rank 16)
    for i in range(4):
        for j in range(4):
            U[i * 4 + j][col] = 1
            V[j * 5 + 4][col] = 1
            W[4 * 4 + i][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}
