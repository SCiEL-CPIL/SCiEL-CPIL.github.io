import numpy as np

def construct():
    n, m, p = 2, 4, 4
    rank = 28
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Helper to add a term
    # term_id: 0..27
    # u_idx: list of (row, coeff)
    # v_idx: list of (row, coeff)
    # w_idx: list of (row, coeff)
    
    def add_term(term_id, u_coeffs, v_coeffs, w_coeffs):
        for idx, c in u_coeffs:
            U[idx, term_id] = c
        for idx, c in v_coeffs:
            V[idx, term_id] = c
        for idx, c in w_coeffs:
            W[idx, term_id] = c
            
    # Strassen products for 2x2 * 2x2
    # Inputs: A (2x2), B (2x2)
    # A indices in global U:
    # a00: 0, a01: 1, a10: 4, a11: 5 (for block 0, j=0,1)
    # a00: 2, a01: 3, a10: 6, a11: 7 (for block 1, j=2,3) -- wait, j=2,3 means indices 2,3,6,7?
    # U row index = i*m + j = i*4 + j.
    # i=0: 0,1,2,3
    # i=1: 4,5,6,7
    # Block 0 (j=0,1):
    # a00: 0, a01: 1
    # a10: 4, a11: 5
    # Block 1 (j=2,3):
    # a00: 2, a01: 3
    # a10: 6, a11: 7
    
    # B indices in global V:
    # V row index = j*p + k = j*4 + k.
    # Block 0 (j=0,1):
    # j=0: b00:0, b01:1, b02:2, b03:3
    # j=1: b10:4, b11:5, b12:6, b13:7
    # Block 1 (j=2,3):
    # j=2: b20:8, b21:9, b22:10, b23:11
    # j=3: b30:12, b31:13, b32:14, b33:15
    
    # C indices in global W:
    # W row index = k*n + i = k*2 + i.
    # k=0: c00:0, c10:1
    # k=1: c01:2, c11:3
    # k=2: c02:4, c12:5
    # k=3: c03:6, c13:7
    
    # We will generate terms for Block 0 (j=0,1) and Block 1 (j=2,3).
    # For each block, we have two 2x2 sub-blocks of B (k=0,1 and k=2,3).
    
    # Block 0 (j=0,1)
    # Sub-block 0 (k=0,1): B cols 0,1
    # B elements: b00(0), b01(1), b10(4), b11(5)
    # A elements: a00(0), a01(1), a10(4), a11(5)
    
    # M1 = (a00+a11)(b00+b11)
    # U: 0:1, 5:1
    # V: 0:1, 5:1
    # W: c00(0):1, c11(3):1
    add_term(0, [(0,1), (5,1)], [(0,1), (5,1)], [(0,1), (3,1)])
    
    # M2 = (a10+a11)b00
    # U: 4:1, 5:1
    # V: 0:1
    # W: c00(0):-1, c10(1):1
    add_term(1, [(4,1), (5,1)], [(0,1)], [(0,-1), (1,1)])
    
    # M3 = a00(b01-b10)
    # U: 0:1
    # V: 1:1, 4:-1
    # W: c01(2):1, c11(3):1
    add_term(2, [(0,1)], [(1,1), (4,-1)], [(2,1), (3,1)])
    
    # M4 = a11(b10-b00)
    # U: 5:1
    # V: 4:1, 0:-1
    # W: c00(0):1, c11(3):-1
    add_term(3, [(5,1)], [(4,1), (0,-1)], [(0,1), (3,-1)])
    
    # M5 = (a00+a01)b11
    # U: 0:1, 1:1
    # V: 5:1
    # W: c01(2):1, c11(3):-1
    add_term(4, [(0,1), (1,1)], [(5,1)], [(2,1), (3,-1)])
    
    # M6 = (a10-a11)(b01+b11)
    # U: 4:1, 5:-1
    # V: 1:1, 5:1
    # W: c00(0):1, c10(1):-1
    add_term(5, [(4,1), (5,-1)], [(1,1), (5,1)], [(0,1), (1,-1)])
    
    # M7 = (a01-a10)(b00+b10)
    # U: 1:1, 4:-1
    # V: 0:1, 4:1
    # W: c10(1):1, c11(3):1
    add_term(6, [(1,1), (4,-1)], [(0,1), (4,1)], [(1,1), (3,1)])
    
    # Sub-block 1 (k=2,3): B cols 2,3
    # B elements: b02(2), b03(3), b12(6), b13(7)
    # A elements: same as above
    
    # N1 = (a00+a11)(b02+b13)
    add_term(7, [(0,1), (5,1)], [(2,1), (7,1)], [(4,1), (7,1)]) # c02(4), c13(7)
    
    # N2 = (a10+a11)b02
    add_term(8, [(4,1), (5,1)], [(2,1)], [(4,-1), (5,1)]) # c02(4), c12(5)
    
    # N3 = a00(b03-b12)
    add_term(9, [(0,1)], [(3,1), (6,-1)], [(6,1), (7,1)]) # c03(6), c13(7)
    
    # N4 = a11(b12-b02)
    add_term(10, [(5,1)], [(6,1), (2,-1)], [(4,1), (7,-1)]) # c02(4), c13(7)
    
    # N5 = (a00+a01)b13
    add_term(11, [(0,1), (1,1)], [(7,1)], [(6,1), (7,-1)]) # c03(6), c13(7)
    
    # N6 = (a10-a11)(b03+b13)
    add_term(12, [(4,1), (5,-1)], [(3,1), (7,1)], [(4,1), (5,-1)]) # c02(4), c12(5)
    
    # N7 = (a01-a10)(b02+b12)
    add_term(13, [(1,1), (4,-1)], [(2,1), (6,1)], [(5,1), (7,1)]) # c12(5), c13(7)
    
    # Block 1 (j=2,3)
    # A elements: a00(2), a01(3), a10(6), a11(7)
    # Sub-block 0 (k=0,1): B cols 0,1, rows 2,3
    # B elements: b20(8), b21(9), b30(12), b31(13)
    
    # P1 = (a00+a11)(b20+b31)
    add_term(14, [(2,1), (7,1)], [(8,1), (13,1)], [(0,1), (3,1)])
    
    # P2 = (a10+a11)b20
    add_term(15, [(6,1), (7,1)], [(8,1)], [(0,-1), (1,1)])
    
    # P3 = a00(b21-b30)
    add_term(16, [(2,1)], [(9,1), (12,-1)], [(2,1), (3,1)])
    
    # P4 = a11(b30-b20)
    add_term(17, [(7,1)], [(12,1), (8,-1)], [(0,1), (3,-1)])
    
    # P5 = (a00+a01)b31
    add_term(18, [(2,1), (3,1)], [(13,1)], [(2,1), (3,-1)])
    
    # P6 = (a10-a11)(b21+b31)
    add_term(19, [(6,1), (7,-1)], [(9,1), (13,1)], [(0,1), (1,-1)])
    
    # P7 = (a01-a10)(b20+b30)
    add_term(20, [(3,1), (6,-1)], [(8,1), (12,1)], [(1,1), (3,1)])
    
    # Sub-block 1 (k=2,3): B cols 2,3, rows 2,3
    # B elements: b22(10), b23(11), b32(14), b33(15)
    
    # Q1 = (a00+a11)(b22+b33)
    add_term(21, [(2,1), (7,1)], [(10,1), (15,1)], [(4,1), (7,1)])
    
    # Q2 = (a10+a11)b22
    add_term(22, [(6,1), (7,1)], [(10,1)], [(4,-1), (5,1)])
    
    # Q3 = a00(b23-b32)
    add_term(23, [(2,1)], [(11,1), (14,-1)], [(6,1), (7,1)])
    
    # Q4 = a11(b32-b22)
    add_term(24, [(7,1)], [(14,1), (10,-1)], [(4,1), (7,-1)])
    
    # Q5 = (a00+a01)b33
    add_term(25, [(2,1), (3,1)], [(15,1)], [(6,1), (7,-1)])
    
    # Q6 = (a10-a11)(b23+b33)
    add_term(26, [(6,1), (7,-1)], [(11,1), (15,1)], [(4,1), (5,-1)])
    
    # Q7 = (a01-a10)(b22+b32)
    add_term(27, [(3,1), (6,-1)], [(10,1), (14,1)], [(5,1), (7,1)])
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
