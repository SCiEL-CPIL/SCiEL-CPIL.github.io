import time

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 59
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    # u_indices: list of (idx, coeff)
    # v_indices: list of (idx, coeff)
    # w_indices: list of (idx, coeff)
    def add_term(u_indices, v_indices, w_indices):
        nonlocal col
        for idx, coeff in u_indices:
            U[idx][col] += coeff
        for idx, coeff in v_indices:
            V[idx][col] += coeff
        for idx, coeff in w_indices:
            W[idx][col] += coeff
        col += 1

    # Strassen terms for A11 (2x2) * BTL (2x2)
    # A11 indices:
    # a00: 0*5+0 = 0
    # a01: 0*5+1 = 1
    # a10: 1*5+0 = 5
    # a11: 1*5+1 = 6
    
    # BTL indices:
    # b00: 0*6+0 = 0
    # b01: 0*6+1 = 1
    # b10: 1*6+0 = 6
    # b11: 1*6+1 = 7
    
    # CL indices (k in 0,1; i in 0,1):
    # C00: 0*2+0 = 0
    # C01: 1*2+0 = 2
    # C10: 0*2+1 = 1
    # C11: 1*2+1 = 3
    
    # M1 = (a00 + a11)(b00 + b11)
    add_term(
        [(0, 1), (6, 1)],
        [(0, 1), (7, 1)],
        [(0, 1), (3, 1)]
    )
    
    # M2 = (a10 + a11)b00
    add_term(
        [(5, 1), (6, 1)],
        [(0, 1)],
        [(1, 1), (3, -1)]
    )
    
    # M3 = a00(b01 - b11)
    add_term(
        [(0, 1)],
        [(1, 1), (7, -1)],
        [(2, 1)]
    )
    
    # M4 = a11(b10 - b00)
    add_term(
        [(6, 1)],
        [(6, 1), (0, -1)],
        [(0, -1), (1, 1)]
    )
    
    # M5 = (a00 + a01)b11
    add_term(
        [(0, 1), (1, 1)],
        [(7, 1)],
        [(0, 1)]
    )
    
    # M6 = (a10 - a00)(b00 + b01)
    add_term(
        [(5, 1), (0, -1)],
        [(0, 1), (1, 1)],
        [(3, 1)]
    )
    
    # M7 = (a01 - a11)(b10 + b11)
    add_term(
        [(1, 1), (6, -1)],
        [(6, 1), (7, 1)],
        [(0, -1), (2, 1)]
    )
    
    # Schoolbook for A12 (2x3) * BBL (3x2) -> CL (2x2)
    # A12: i in 0..1, j in 2..4
    # BBL: j in 2..4, k in 0..1
    # CL: k in 0..1, i in 0..1
    for i in range(2):
        for j in range(2, 5):
            for k in range(2):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                add_term(
                    [(u_idx, 1)],
                    [(v_idx, 1)],
                    [(w_idx, 1)]
                )
                
    # Schoolbook for A11 (2x2) * BTR (2x4) -> CR (2x4)
    # A11: i in 0..1, j in 0..1
    # BTR: j in 0..1, k in 2..5
    # CR: k in 2..5, i in 0..1
    for i in range(2):
        for j in range(2):
            for k in range(2, 6):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                add_term(
                    [(u_idx, 1)],
                    [(v_idx, 1)],
                    [(w_idx, 1)]
                )
                
    # Schoolbook for A12 (2x3) * BBR (3x4) -> CR (2x4)
    # A12: i in 0..1, j in 2..4
    # BBR: j in 2..4, k in 2..5
    # CR: k in 2..5, i in 0..1
    for i in range(2):
        for j in range(2, 5):
            for k in range(2, 6):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                add_term(
                    [(u_idx, 1)],
                    [(v_idx, 1)],
                    [(w_idx, 1)]
                )
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
