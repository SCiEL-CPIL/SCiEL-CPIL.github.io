import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    # Rank 65: Strassen for 4x4x4 (rank 49) + Schoolbook for 4x4x1 (rank 16)
    return _construct_strassen_65()

def _construct_strassen_65():
    # Strassen 4x4x4 (rank 49) + Schoolbook 4x4x1 (rank 16)
    n, m, p = 4, 4, 5
    rank = 65
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_term(col, u_list, v_list, w_list):
        for i, j, c in u_list:
            if c != 0:
                U[i * m + j][col] += c
        for j, k, c in v_list:
            if c != 0:
                V[j * p + k][col] += c
        for k, i, c in w_list:
            if c != 0:
                W[k * n + i][col] += c

    col = 0
    # Strassen 2x2x2 definitions
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # We need to expand this to 4x4x4.
    # A is 4x4, B is 4x4.
    # We use Strassen on the 2x2 block structure.
    # A = [[A00, A01], [A10, A11]] where each is 2x2.
    # B = [[B00, B01], [B10, B11]] where each is 2x2.
    # C = [[C00, C01], [C10, C11]].
    
    # The 7 products M1..M7 are 2x2 matrix multiplications.
    # Each 2x2 multiplication has rank 7.
    # Total rank 7 * 7 = 49.
    
    # We define the linear forms for the 2x2 Strassen.
    # For a 2x2 multiplication C = A*B:
    # U indices: (i, j) for A. V indices: (j, k) for B. W indices: (k, i) for C.
    
    # We will iterate over the 7 Strassen terms for the outer 2x2,
    # and for each, iterate over the 7 Strassen terms for the inner 2x2.
    
    # Outer Strassen definitions (indices 0,1 for row/col blocks)
    # A blocks: 00, 01, 10, 11
    # B blocks: 00, 01, 10, 11
    
    outer_defs = [
        # M1: (A00+A11)(B00+B11)
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
        # M2: (A10+A11)B00
        ([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,-1)]),
        # M3: A00(B10-B11)
        ([(0,0,1)], [(1,0,1), (1,1,-1)], [(0,0,1), (1,0,1)]),
        # M4: A11(B01-B00)
        ([(1,1,1)], [(0,1,1), (0,0,-1)], [(1,1,1), (1,0,-1)]),
        # M5: (A00+A01)B11
        ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,1), (0,1,1)]),
        # M6: (A10-A00)(B00+B01)
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,0,1), (0,0,-1)]),
        # M7: (A01-A11)(B10+B11)
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,1,1), (1,1,1)]),
    ]
    
    # Inner Strassen definitions for 2x2 blocks
    # Indices within a 2x2 block: 0,1
    inner_defs = [
        # m1: (a00+a11)(b00+b11)
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
        # m2: (a10+a11)b00
        ([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,-1)]),
        # m3: a00(b10-b11)
        ([(0,0,1)], [(1,0,1), (1,1,-1)], [(0,0,1), (1,0,1)]),
        # m4: a11(b01-b00)
        ([(1,1,1)], [(0,1,1), (0,0,-1)], [(1,1,1), (1,0,-1)]),
        # m5: (a00+a01)b11
        ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,1), (0,1,1)]),
        # m6: (a10-a00)(b00+b01)
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,0,1), (0,0,-1)]),
        # m7: (a01-a11)(b10+b11)
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,1,1), (1,1,1)]),
    ]

    for o_idx, (a_outer, b_outer, c_outer) in enumerate(outer_defs):
        for i_idx, (a_inner, b_inner, c_inner) in enumerate(inner_defs):
            u_list = []
            v_list = []
            w_list = []
            
            # Expand A terms
            for oa, ob, coeff_a in a_outer:
                for ia, ib, coeff_ai in a_inner:
                    # Global row i = oa * 2 + ia
                    # Global col j = ob * 2 + ib
                    gi = oa * 2 + ia
                    gj = ob * 2 + ib
                    u_list.append((gi, gj, coeff_a * coeff_ai))
            
            # Expand B terms
            for ob, oc, coeff_b in b_outer:
                for ib, ic, coeff_bi in b_inner:
                    # Global row j = ob * 2 + ib
                    # Global col k = oc * 2 + ic
                    gj = ob * 2 + ib
                    gk = oc * 2 + ic
                    v_list.append((gj, gk, coeff_b * coeff_bi))
            
            # Expand C terms
            for oc, od, coeff_c in c_outer:
                for ic, id, coeff_ci in c_inner:
                    # Global row k = oc * 2 + ic
                    # Global col i = od * 2 + id
                    gk = oc * 2 + ic
                    gi = od * 2 + id
                    w_list.append((gk, gi, coeff_c * coeff_ci))
            
            if u_list and v_list and w_list:
                add_term(col, u_list, v_list, w_list)
                col += 1

    # Schoolbook for 4x4x1 (k=4)
    for i in range(4):
        for j in range(4):
            U[i * 4 + j][col] = 1
            V[j * 5 + 4][col] = 1
            W[4 * 4 + i][col] = 1
            col += 1

    return {"U": U, "V": V, "W": W}
