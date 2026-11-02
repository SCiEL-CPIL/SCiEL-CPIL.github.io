import numpy as np

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    rank = 43
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_indices, v_indices, w_indices):
        nonlocal col
        for idx, coeff in u_indices:
            U[idx][col] += coeff
        for idx, coeff in v_indices:
            V[idx][col] += coeff
        for idx, coeff in w_indices:
            W[idx][col] += coeff
        col += 1

    # 1. A[0:2, 0:2] * B[0:2, 0:2] (Strassen)
    # A indices: 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
    # B indices: 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
    # C indices: 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
    
    # M1 = (a00+a11)(b00+b11)
    add_term([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)])
    # M2 = (a10+a11)b11
    add_term([(2, 1), (3, 1)], [(3, 1)], [(2, 1), (3, -1)]) # C10 += M2, C11 -= M2
    # M3 = a00(b10-b00)
    add_term([(0, 1)], [(2, 1), (0, -1)], [(1, 1), (3, 1)]) # C01 += M3, C11 += M3
    # M4 = a11(b01-b00)
    add_term([(3, 1)], [(1, 1), (0, -1)], [(0, -1), (2, 1)]) # C00 -= M4, C10 += M4
    # M5 = (a00+a01)b00
    add_term([(0, 1), (1, 1)], [(0, 1)], [(0, 1)]) # C00 += M5
    # M6 = (a10-a01)(b01+b11)
    add_term([(2, 1), (1, -1)], [(1, 1), (3, 1)], [(3, 1)]) # C11 += M6
    # M7 = (a01+a11)b10
    add_term([(1, 1), (3, 1)], [(2, 1)], [(0, -1), (1, 1)]) # C00 -= M7, C01 += M7

    # 2. A[0:2, 0:2] * B[0:2, 2:4] (Strassen)
    # B indices: 4: (0,2), 5: (0,3), 6: (1,2), 7: (1,3)
    # C indices: 4: (0,2), 5: (0,3), 6: (1,2), 7: (1,3)
    
    add_term([(0, 1), (3, 1)], [(4, 1), (7, 1)], [(4, 1), (7, 1)])
    add_term([(2, 1), (3, 1)], [(7, 1)], [(6, 1), (7, -1)])
    add_term([(0, 1)], [(6, 1), (4, -1)], [(5, 1), (7, 1)])
    add_term([(3, 1)], [(5, 1), (4, -1)], [(4, -1), (6, 1)])
    add_term([(0, 1), (1, 1)], [(4, 1)], [(4, 1)])
    add_term([(2, 1), (1, -1)], [(5, 1), (7, 1)], [(7, 1)])
    add_term([(1, 1), (3, 1)], [(6, 1)], [(4, -1), (5, 1)])

    # 3. A[0:2, 0:2] * B[0:2, 4:5] (Schoolbook)
    # B indices: 8: (0,4), 9: (1,4)
    # C indices: 8: (0,4), 9: (1,4)
    
    add_term([(0, 1)], [(8, 1)], [(8, 1)]) # a00 b04
    add_term([(1, 1)], [(9, 1)], [(8, 1)]) # a01 b14
    add_term([(2, 1)], [(8, 1)], [(9, 1)]) # a10 b04
    add_term([(3, 1)], [(9, 1)], [(9, 1)]) # a11 b14

    # 4. A[0:2, 2:3] * B[2:3, 0:5]
    # A indices: 4: (0,2), 5: (1,2)
    # B indices: 10: (2,0), 11: (2,1), 12: (2,2), 13: (2,3), 14: (2,4)
    # C indices: 0-4 (row 0), 5-9 (row 1)
    
    for k in range(5):
        b_idx = 10 + k
        c0_idx = k
        c1_idx = 5 + k
        add_term([(4, 1)], [(b_idx, 1)], [(c0_idx, 1)])
        add_term([(5, 1)], [(b_idx, 1)], [(c1_idx, 1)])

    # 5. A[2:3, 0:2] * B[0:2, 0:5]
    # A indices: 6: (2,0), 7: (2,1)
    # B indices: 0: (0,0), 1: (0,1), 2: (0,2), 3: (0,3), 4: (0,4)
    #             5: (1,0), 6: (1,1), 7: (1,2), 8: (1,3), 9: (1,4)
    # C indices: 10-14 (row 2)
    
    for k in range(5):
        b0_idx = k
        b1_idx = 5 + k
        c_idx = 10 + k
        add_term([(6, 1)], [(b0_idx, 1)], [(c_idx, 1)])
        add_term([(7, 1)], [(b1_idx, 1)], [(c_idx, 1)])

    # 6. A[2:3, 2:3] * B[2:3, 0:5]
    # A indices: 8: (2,2)
    # B indices: 10-14
    # C indices: 10-14
    
    for k in range(5):
        b_idx = 10 + k
        c_idx = 10 + k
        add_term([(8, 1)], [(b_idx, 1)], [(c_idx, 1)])

    return {"U": U, "V": V, "W": W}
