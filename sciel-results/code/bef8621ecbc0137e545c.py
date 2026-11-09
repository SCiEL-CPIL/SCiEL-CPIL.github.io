import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    rank = 22
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Indices
    # U: i*m + j
    # V: j*p + k
    # W: k*n + i
    
    col = 0
    
    # Helper to add a term
    def add_term(u_coeffs, v_coeffs, w_coeffs):
        nonlocal col
        for idx, coeff in u_coeffs.items():
            U[idx][col] += coeff
        for idx, coeff in v_coeffs.items():
            V[idx][col] += coeff
        for idx, coeff in w_coeffs.items():
            W[idx][col] += coeff
        col += 1

    # Block 1: C1 = AR * BB1
    # AR: a00, a01, a10, a11
    # BB1: b00, b01, b10, b11
    
    # M1 = (a00 + a11)(b00 + b11)
    add_term(
        {0: 1, 4: 1},  # a00, a11
        {0: 1, 5: 1},  # b00, b11
        {0: 1, 3: 1}   # c00, c11
    )
    
    # M2 = (a10 + a11)b00
    add_term(
        {3: 1, 4: 1},  # a10, a11
        {0: 1},        # b00
        {1: 1, 3: -1}  # c10, c11
    )
    
    # M3 = a00(b01 - b11)
    add_term(
        {0: 1},        # a00
        {1: 1, 5: -1}, # b01, b11
        {2: 1, 3: 1}   # c01, c11
    )
    
    # M4 = a11(b10 - b00)
    add_term(
        {4: 1},        # a11
        {4: 1, 0: -1}, # b10, b00
        {0: 1, 1: 1}   # c00, c10
    )
    
    # M5 = (a00 + a01)b11
    add_term(
        {0: 1, 1: 1},  # a00, a01
        {5: 1},        # b11
        {0: -1, 2: 1}  # c00, c01
    )
    
    # M6 = (a10 - a00)(b00 + b01)
    add_term(
        {3: 1, 0: -1}, # a10, a00
        {0: 1, 1: 1},  # b00, b01
        {3: 1}         # c11
    )
    
    # M7 = (a01 - a11)(b10 + b11)
    add_term(
        {1: 1, 4: -1}, # a01, a11
        {4: 1, 5: 1},  # b10, b11
        {0: 1}         # c00
    )

    # Block 2: C2 = AR * BB2
    # AR: a00, a01, a10, a11
    # BB2: b02, b03, b12, b13
    
    # M'1 = (a00 + a11)(b02 + b13)
    add_term(
        {0: 1, 4: 1},  # a00, a11
        {2: 1, 7: 1},  # b02, b13
        {4: 1, 7: 1}   # c02, c13
    )
    
    # M'2 = (a10 + a11)b02
    add_term(
        {3: 1, 4: 1},  # a10, a11
        {2: 1},        # b02
        {5: 1, 7: -1}  # c12, c13
    )
    
    # M'3 = a00(b03 - b13)
    add_term(
        {0: 1},        # a00
        {3: 1, 7: -1}, # b03, b13
        {6: 1, 7: 1}   # c03, c13
    )
    
    # M'4 = a11(b12 - b02)
    add_term(
        {4: 1},        # a11
        {6: 1, 2: -1}, # b12, b02
        {4: 1, 5: 1}   # c02, c12
    )
    
    # M'5 = (a00 + a01)b13
    add_term(
        {0: 1, 1: 1},  # a00, a01
        {7: 1},        # b13
        {4: -1, 6: 1}  # c02, c03
    )
    
    # M'6 = (a10 - a00)(b02 + b03)
    add_term(
        {3: 1, 0: -1}, # a10, a00
        {2: 1, 3: 1},  # b02, b03
        {7: 1}         # c13
    )
    
    # M'7 = (a01 - a11)(b12 + b13)
    add_term(
        {1: 1, 4: -1}, # a01, a11
        {6: 1, 7: 1},  # b12, b13
        {4: 1}         # c02
    )

    # Block 3: C3 = AL * BT
    # AL: a02, a12
    # BT: b20, b21, b22, b23
    
    # a02 * b20 -> c00
    add_term({2: 1}, {8: 1}, {0: 1})
    # a02 * b21 -> c01
    add_term({2: 1}, {9: 1}, {2: 1})
    # a02 * b22 -> c02
    add_term({2: 1}, {10: 1}, {4: 1})
    # a02 * b23 -> c03
    add_term({2: 1}, {11: 1}, {6: 1})
    
    # a12 * b20 -> c10
    add_term({5: 1}, {8: 1}, {1: 1})
    # a12 * b21 -> c11
    add_term({5: 1}, {9: 1}, {3: 1})
    # a12 * b22 -> c12
    add_term({5: 1}, {10: 1}, {5: 1})
    # a12 * b23 -> c13
    add_term({5: 1}, {11: 1}, {7: 1})

    return {"U": U, "V": V, "W": W}
