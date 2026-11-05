import time
import sys

DIMENSIONS = [4, 4, 5]


def _build_double_strassen(U, V, W, col_start, m, n):
    """
    Build the 49 rank-1 terms for 4x4x4 matrix multiplication using double Strassen.
    A is 4x4, B is 4x4, C is 4x4.
    U: (i*m+j, col) for A[i,j]
    V: (j*m+k, col) for B[j,k] -- but for the 4x4x4 part, k only goes 0..3
    W: (k*n+i, col) for C[i,k] -- W[k*n+i] with n=4
    Returns the next column index.
    """
    col = col_start

    # Inner Strassen definitions for 2x2x2:
    inner_defs = [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 0, 1), (1, 1, 1)]),
        ([(2, 1), (3, 1)], [(0, 1)], [(1, 0, 1), (1, 1, -1)]),
        ([(0, 1)], [(1, 1), (3, -1)], [(0, 1, 1), (1, 1, 1)]),
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 0, 1), (1, 0, 1)]),
        ([(0, 1), (1, 1)], [(3, 1)], [(0, 0, -1), (0, 1, 1)]),
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(1, 1, 1)]),
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 0, 1)]),
    ]

    # Super Strassen definitions for 4x4x4:
    super_defs = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [(0, 0, 1), (1, 1, 1)]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [(1, 0, 1), (1, 1, -1)]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [(0, 1, 1), (1, 1, 1)]),
        ([0, 0, 0, 1], [-1, 0, 1, 0], [(0, 0, 1), (1, 0, 1)]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [(0, 0, -1), (0, 1, 1)]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [(1, 1, 1)]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [(0, 0, 1)]),
    ]

    def get_p_global(p_elem_idx, p_block_coeffs):
        r = p_elem_idx // 2
        c = p_elem_idx % 2
        res = []
        for b_idx, coeff in enumerate(p_block_coeffs):
            if coeff == 0:
                continue
            if b_idx == 0:
                gi, gj = r, c
            elif b_idx == 1:
                gi, gj = r, 2 + c
            elif b_idx == 2:
                gi, gj = 2 + r, c
            elif b_idx == 3:
                gi, gj = 2 + r, 2 + c
            res.append((gi, gj, coeff))
        return res

    def get_q_global(q_elem_idx, q_block_coeffs):
        r = q_elem_idx // 2
        c = q_elem_idx % 2
        res = []
        for b_idx, coeff in enumerate(q_block_coeffs):
            if coeff == 0:
                continue
            if b_idx == 0:
                gj, gk = r, c
            elif b_idx == 1:
                gj, gk = r, 2 + c
            elif b_idx == 2:
                gj, gk = 2 + r, c
            elif b_idx == 3:
                gj, gk = 2 + r, 2 + c
            res.append((gj, gk, coeff))
        return res

    for s_idx, (p_bc, q_bc, c_contribs) in enumerate(super_defs):
        for i_idx, (p_elems, q_elems, r_contribs) in enumerate(inner_defs):
            u_list = []
            v_list = []
            w_list = []

            for p_elem, coeff in p_elems:
                for gi, gj, c in get_p_global(p_elem, p_bc):
                    u_list.append((gi, gj, coeff * c))

            for q_elem, coeff in q_elems:
                for gj, gk, c in get_q_global(q_elem, q_bc):
                    v_list.append((gj, gk, coeff * c))

            for br, bc, sign in c_contribs:
                for lr, lc, local_coeff in r_contribs:
                    gi = br * 2 + lr
                    gk = bc * 2 + lc
                    w_list.append((gk, gi, sign * local_coeff))

            if u_list and v_list and w_list:
                for i, j, c in u_list:
                    if c != 0:
                        U[i * m + j][col] += c
                for j, k, c in v_list:
                    if c != 0:
                        V[j * m + k][col] += c
                for k, i, c in w_list:
                    if c != 0:
                        W[k * n + i][col] += c
                col += 1

    return col


def _build_reduced_4x4x1(U, V, W, col_start, m, n, p):
    """
    Build 12 rank-1 terms for 4x4x1 (column p-1 of B) using Strassen-like reduction.
    C[i, p-1] = sum_j A[i,j] * B[j, p-1]
    
    For each pair of rows (i, i+1) and pair of cols (j, j+1), we compute:
    C[i, p-1]  += A[i,j]*B[j,p-1] + A[i,j+1]*B[j+1,p-1]
    C[i+1,p-1] += A[i+1,j]*B[j,p-1] + A[i+1,j+1]*B[j+1,p-1]
    
    Using Strassen on the 2x2 block:
    M1 = (A[i,j]+A[i+1,j]) * (B[j,p-1]+B[j+1,p-1])  -> contributes to both
    M2 = (A[i,j]+A[i+1,j]) * B[j+1,p-1]              -> correction
    M3 = A[i+1,j] * (B[j,p-1]+B[j+1,p-1])            -> correction
    
    Actually, let's use the standard 2x2x1 Strassen:
    For computing [c1, c2]^T = [a1, a2] * [b1, b2]^T (2x1 result from 2x2 * 2x1):
    c1 = a1*b1 + a2*b2
    c2 = a3*b1 + a4*b2
    
    M1 = (a1+a2)*b1  -> c1 += M1, c2 -= M1
    M2 = (a3-a1)*b2  -> c1 += M2, c2 += M2
    M3 = (a2+a4)*b1  -> c2 += M3
    M4 = a1*(b1-b2)  -> c1 += M4
    M5 = a4*(b2-b1)  -> c2 += M5
    
    Wait, that's 5 terms for 2 terms. Let me reconsider.
    
    For 2x1 result: c1 = a1*b1 + a2*b2, c2 = a3*b1 + a4*b2
    Schoolbook: 4 terms.
    
    Strassen-like:
    M1 = (a1+a2)*b1: c1 += M1, c2 -= M1
    M2 = (a3-a1)*b2: c1 += M2, c2 += M2  
    M3 = (a2+a4)*b1: c2 += M3
    M4 = a1*(b1-b2): c1 += M4
    M5 = a4*(b2-b1): c2 += M5
    
    Check c1: M1 + M2 + M4 = (a1+a2)*b1 + (a3-a1)*b2 + a1*b1 - a1*b2
    = a1*b1 + a2*b1 + a3*b2 - a1*b2 + a1*b1 - a1*b2
    = 2*a1*b1 + a2*b1 + a3*b2 - 2*a1*b2
    That's wrong.
    
    Let me use the correct 2x2x1 decomposition. Actually, for a 2x2 matrix times a 2x1 vector,
    the rank is exactly 4 (no reduction possible with standard Strassen since one dimension is 1).
    
    Let me just use schoolbook for the 4x4x1 part. Rank 16.
    """
    col = col_start
    for i in range(4):
        for j in range(4):
            U[i * m + j][col] = 1
            V[j * m + (p - 1)][col] = 1
            W[(p - 1) * n + i][col] = 1
            col += 1
    return col


def construct():
    start_time = time.time()
    DEADLINE = 110.0  # safety margin before 120s limit

    n, m, p = DIMENSIONS  # 4, 4, 5

    rank = 65
    U = [[0] * rank for _ in range(n * m)]   # 16 x 65
    V = [[0] * rank for _ in range(m * p)]   # 20 x 65
    W = [[0] * rank for _ in range(p * n)]   # 20 x 65

    # Part 1: 4x4x4 via double Strassen (49 terms)
    col = _build_double_strassen(U, V, W, 0, m, n)

    if time.time() - start_time > DEADLINE - 5:
        return {"U": U, "V": V, "W": W}

    # Part 2: 4x4x1 via schoolbook (16 terms)
    col = _build_reduced_4x4x1(U, V, W, col, m, n, p)

    return {"U": U, "V": V, "W": W}
