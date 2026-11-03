# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 5
    U_rows = n * m  # 12
    V_rows = m * p  # 20
    W_rows = p * n  # 15
    
    deadline = time.time() + 110  # safety margin
    
    # Helper to get indices
    def a_idx(i, j): return i * m + j
    def b_idx(j, k): return j * p + k
    def c_idx(k, i): return k * n + i

    terms = []
    
    # Function to add a term
    def add_term(u_coeffs, v_coeffs, w_coeffs):
        # u_coeffs: dict {a_idx: coeff}
        # v_coeffs: dict {b_idx: coeff}
        # w_coeffs: dict {c_idx: coeff}
        terms.append((u_coeffs, v_coeffs, w_coeffs))

    # Strassen 2x2x2
    # A indices: a00, a01, a10, a11
    # B indices: b00, b01, b10, b11
    # C indices: c00, c01, c10, c11
    def add_strassen_2x2x2(a00, a01, a10, a11, b00, b01, b10, b11, c00, c01, c10, c11):
        # m1 = (a00 + a11)(b00 + b11) -> c00:+1, c11:+1
        add_term({a00:1, a11:1}, {b00:1, b11:1}, {c00:1, c11:1})
        
        # m2 = (a10 + a11)b00 -> c10:+1, c11:-1
        add_term({a10:1, a11:1}, {b00:1}, {c10:1, c11:-1})
        
        # m3 = a00(b01 - b11) -> c01:+1, c11:+1
        add_term({a00:1}, {b01:1, b11:-1}, {c01:1, c11:1})
        
        # m4 = a11(b10 - b00) -> c00:+1, c10:+1
        add_term({a11:1}, {b10:1, b00:-1}, {c00:1, c10:1})
        
        # m5 = (a00 + a01)b11 -> c00:-1, c01:+1
        add_term({a00:1, a01:1}, {b11:1}, {c00:-1, c01:1})
        
        # m6 = (a10 - a00)(b00 + b01) -> c11:+1
        add_term({a10:1, a00:-1}, {b00:1, b01:1}, {c11:1})
        
        # m7 = (a01 - a11)(b10 + b11) -> c00:+1, c11:+1
        add_term({a01:1, a11:-1}, {b10:1, b11:1}, {c00:1, c11:1})

    # Schoolbook 2x2x2
    def add_schoolbook_2x2x2(a00, a01, a10, a11, b00, b01, b10, b11, c00, c01, c10, c11):
        # c00 = a00b00 + a01b10
        add_term({a00:1}, {b00:1}, {c00:1})
        add_term({a01:1}, {b10:1}, {c00:1})
        # c01 = a00b01 + a01b11
        add_term({a00:1}, {b01:1}, {c01:1})
        add_term({a01:1}, {b11:1}, {c01:1})
        # c10 = a10b00 + a11b10
        add_term({a10:1}, {b00:1}, {c10:1})
        add_term({a11:1}, {b10:1}, {c10:1})
        # c11 = a10b01 + a11b11
        add_term({a10:1}, {b01:1}, {c11:1})
        add_term({a11:1}, {b11:1}, {c11:1})

    # Schoolbook 2x2x1
    # A: 2x2, B: 2x1, C: 2x1
    def add_schoolbook_2x2x1(a00, a01, a10, a11, b0, b1, c0, c1):
        # c0 = a00b0 + a01b1
        add_term({a00:1}, {b0:1}, {c0:1})
        add_term({a01:1}, {b1:1}, {c0:1})
        # c1 = a10b0 + a11b1
        add_term({a10:1}, {b0:1}, {c1:1})
        add_term({a11:1}, {b1:1}, {c1:1})

    # Schoolbook 1x2x2
    # A: 1x2, B: 2x2, C: 1x2
    def add_schoolbook_1x2x2(a0, a1, b00, b01, b10, b11, c0, c1):
        # c0 = a0b00 + a1b10
        add_term({a0:1}, {b00:1}, {c0:1})
        add_term({a1:1}, {b10:1}, {c0:1})
        # c1 = a0b01 + a1b11
        add_term({a0:1}, {b01:1}, {c1:1})
        add_term({a1:1}, {b11:1}, {c1:1})

    # Schoolbook 1x2x1
    # A: 1x2, B: 2x1, C: 1x1
    def add_schoolbook_1x2x1(a0, a1, b0, b1, c0):
        # c0 = a0b0 + a1b1
        add_term({a0:1}, {b0:1}, {c0:1})
        add_term({a1:1}, {b1:1}, {c0:1})

    # Block 1: Top-Left (i=0,1; j=0,1; k=0..4)
    # A: 2x2 (i=0,1; j=0,1)
    # B: 2x5 (j=0,1; k=0..4)
    
    # B split into k=0,1 (2x2), k=2,3 (2x2), k=4 (2x1)
    
    # B_part0: j=0,1; k=0,1
    b00, b01, b10, b11 = b_idx(0,0), b_idx(0,1), b_idx(1,0), b_idx(1,1)
    c00, c01, c10, c11 = c_idx(0,0), c_idx(1,0), c_idx(0,1), c_idx(1,1)
    # Note: c_idx(k,i)
    # c00 corresponds to k=0, i=0
    # c01 corresponds to k=1, i=0
    # c10 corresponds to k=0, i=1
    # c11 corresponds to k=1, i=1
    
    a00, a01, a10, a11 = a_idx(0,0), a_idx(0,1), a_idx(1,0), a_idx(1,1)
    
    add_strassen_2x2x2(a00, a01, a10, a11, b00, b01, b10, b11, c00, c01, c10, c11)
    
    # B_part1: j=0,1; k=2,3
    b00, b01, b10, b11 = b_idx(0,2), b_idx(0,3), b_idx(1,2), b_idx(1,3)
    c00, c01, c10, c11 = c_idx(2,0), c_idx(3,0), c_idx(2,1), c_idx(3,1)
    
    add_strassen_2x2x2(a00, a01, a10, a11, b00, b01, b10, b11, c00, c01, c10, c11)
    
    # B_part2: j=0,1; k=4
    b0, b1 = b_idx(0,4), b_idx(1,4)
    c0, c1 = c_idx(4,0), c_idx(4,1)
    
    add_schoolbook_2x2x1(a00, a01, a10, a11, b0, b1, c0, c1)

    # Block 2: Top-Right (i=0,1; j=2,3; k=0..4)
    # A: 2x2 (i=0,1; j=2,3)
    a00, a01, a10, a11 = a_idx(0,2), a_idx(0,3), a_idx(1,2), a_idx(1,3)
    
    # B_part0: j=2,3; k=0,1
    b00, b01, b10, b11 = b_idx(2,0), b_idx(2,1), b_idx(3,0), b_idx(3,1)
    c00, c01, c10, c11 = c_idx(0,0), c_idx(1,0), c_idx(0,1), c_idx(1,1)
    
    add_strassen_2x2x2(a00, a01, a10, a11, b00, b01, b10, b11, c00, c01, c10, c11)
    
    # B_part1: j=2,3; k=2,3
    b00, b01, b10, b11 = b_idx(2,2), b_idx(2,3), b_idx(3,2), b_idx(3,3)
    c00, c01, c10, c11 = c_idx(2,0), c_idx(3,0), c_idx(2,1), c_idx(3,1)
    
    add_strassen_2x2x2(a00, a01, a10, a11, b00, b01, b10, b11, c00, c01, c10, c11)
    
    # B_part2: j=2,3; k=4
    b0, b1 = b_idx(2,4), b_idx(3,4)
    c0, c1 = c_idx(4,0), c_idx(4,1)
    
    add_schoolbook_2x2x1(a00, a01, a10, a11, b0, b1, c0, c1)

    # Block 3: Bottom-Left (i=2; j=0,1; k=0..4)
    # A: 1x2 (i=2; j=0,1)
    a0, a1 = a_idx(2,0), a_idx(2,1)
    
    # B_part0: j=0,1; k=0,1
    b00, b01, b10, b11 = b_idx(0,0), b_idx(0,1), b_idx(1,0), b_idx(1,1)
    c0, c1 = c_idx(0,2), c_idx(1,2)
    
    add_schoolbook_1x2x2(a0, a1, b00, b01, b10, b11, c0, c1)
    
    # B_part1: j=0,1; k=2,3
    b00, b01, b10, b11 = b_idx(0,2), b_idx(0,3), b_idx(1,2), b_idx(1,3)
    c0, c1 = c_idx(2,2), c_idx(3,2)
    
    add_schoolbook_1x2x2(a0, a1, b00, b01, b10, b11, c0, c1)
    
    # B_part2: j=0,1; k=4
    b0, b1 = b_idx(0,4), b_idx(1,4)
    c0 = c_idx(4,2)
    
    add_schoolbook_1x2x1(a0, a1, b0, b1, c0)

    # Block 4: Bottom-Right (i=2; j=2,3; k=0..4)
    # A: 1x2 (i=2; j=2,3)
    a0, a1 = a_idx(2,2), a_idx(2,3)
    
    # B_part0: j=2,3; k=0,1
    b00, b01, b10, b11 = b_idx(2,0), b_idx(2,1), b_idx(3,0), b_idx(3,1)
    c0, c1 = c_idx(0,2), c_idx(1,2)
    
    add_schoolbook_1x2x2(a0, a1, b00, b01, b10, b11, c0, c1)
    
    # B_part1: j=2,3; k=2,3
    b00, b01, b10, b11 = b_idx(2,2), b_idx(2,3), b_idx(3,2), b_idx(3,3)
    c0, c1 = c_idx(2,2), c_idx(3,2)
    
    add_schoolbook_1x2x2(a0, a1, b00, b01, b10, b11, c0, c1)
    
    # B_part2: j=2,3; k=4
    b0, b1 = b_idx(2,4), b_idx(3,4)
    c0 = c_idx(4,2)
    
    add_schoolbook_1x2x1(a0, a1, b0, b1, c0)

    rank = len(terms)
    
    U = np.zeros((U_rows, rank), dtype=np.float64)
    V = np.zeros((V_rows, rank), dtype=np.float64)
    W = np.zeros((W_rows, rank), dtype=np.float64)
    
    for r, (u_c, v_c, w_c) in enumerate(terms):
        for idx, val in u_c.items():
            U[idx, r] = val
        for idx, val in v_c.items():
            V[idx, r] = val
        for idx, val in w_c.items():
            W[idx, r] = val
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
