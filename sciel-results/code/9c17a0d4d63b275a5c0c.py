"""Strassen-based rank-49 tensor decomposition for 4x4 matrix multiplication."""
import time

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-49 exact tensor decomposition for 4x4 matrix multiplication
    using the 2x2 Strassen algorithm applied to 2x2 blocks.
    """
    n, m, p = 4, 4, 4
    rank = 49
    
    # Initialize factor matrices
    U = [[0] * rank for _ in range(n * m)]  # (16, 49)
    V = [[0] * rank for _ in range(m * p)]  # (16, 49)
    W = [[0] * rank for _ in range(p * n)]  # (16, 49)
    
    # Block contribution coefficients:
    # block_contrib[b][c_block] = coefficient
    # c_block: 0=C00, 1=C01, 2=C10, 3=C11
    block_contrib = [
        [1, 0, 0, 1],   # M0 (M1): C00:+1, C11:+1
        [0, 0, 1, -1],  # M1 (M2): C10:+1, C11:-1
        [0, 1, 0, 1],   # M2 (M3): C01:+1, C11:+1
        [1, 0, 1, 0],   # M3 (M4): C00:+1, C10:+1
        [-1, 1, 0, 0],  # M4 (M5): C00:-1, C01:+1
        [0, 0, 0, 1],   # M5 (M6): C11:+1
        [1, 0, 0, 0],   # M6 (M7): C00:+1
    ]
    
    # Scalar contribution within each M_b:
    # scalar_contrib[s][pos] where pos: 0=[0,0], 1=[0,1], 2=[1,0], 3=[1,1]
    scalar_contrib = [
        [1, 0, 0, 1],   # s=0 (m1): [0,0]:+1, [1,1]:+1
        [0, 0, 1, -1],  # s=1 (m2): [1,0]:+1, [1,1]:-1
        [0, 1, 0, 1],   # s=2 (m3): [0,1]:+1, [1,1]:+1
        [1, 0, 1, 0],   # s=3 (m4): [0,0]:+1, [1,0]:+1
        [-1, 1, 0, 0],  # s=4 (m5): [0,0]:-1, [0,1]:+1
        [0, 0, 0, 1],   # s=5 (m6): [1,1]:+1
        [1, 0, 0, 0],   # s=6 (m7): [0,0]:+1
    ]
    
    # C block to (row_offset, col_offset) mapping
    c_block_offsets = [(0, 0), (0, 2), (2, 0), (2, 2)]
    
    # Build W matrix
    # W[k*n+i, r] where r = b*7 + s
    for b in range(7):
        for s in range(7):
            r = b * 7 + s
            for pos in range(4):
                sc_coeff = scalar_contrib[s][pos]
                if sc_coeff == 0:
                    continue
                if pos == 0:
                    local_r, local_c = 0, 0
                elif pos == 1:
                    local_r, local_c = 0, 1
                elif pos == 2:
                    local_r, local_c = 1, 0
                else:
                    local_r, local_c = 1, 1
                
                for cb in range(4):
                    bc_coeff = block_contrib[b][cb]
                    if bc_coeff == 0:
                        continue
                    row_off, col_off = c_block_offsets[cb]
                    i = row_off + local_r
                    k = col_off + local_c
                    W[k * n + i][r] += bc_coeff * sc_coeff
    
    # Define L_b and R_b as 2x2 matrices of coefficient dicts
    L_defs = [
        # L1 = A00 + A11
        [
            {(0,0): 1, (2,2): 1},
            {(0,1): 1, (2,3): 1},
            {(1,0): 1, (3,2): 1},
            {(1,1): 1, (3,3): 1},
        ],
        # L2 = A10 + A11
        [
            {(2,0): 1, (2,2): 1},
            {(2,1): 1, (2,3): 1},
            {(3,0): 1, (3,2): 1},
            {(3,1): 1, (3,3): 1},
        ],
        # L3 = A00
        [
            {(0,0): 1},
            {(0,1): 1},
            {(1,0): 1},
            {(1,1): 1},
        ],
        # L4 = A11
        [
            {(2,2): 1},
            {(2,3): 1},
            {(3,2): 1},
            {(3,3): 1},
        ],
        # L5 = A00 + A01
        [
            {(0,0): 1, (0,2): 1},
            {(0,1): 1, (0,3): 1},
            {(1,0): 1, (1,2): 1},
            {(1,1): 1, (1,3): 1},
        ],
        # L6 = A10 - A00
        [
            {(2,0): 1, (0,0): -1},
            {(2,1): 1, (0,1): -1},
            {(3,0): 1, (1,0): -1},
            {(3,1): 1, (1,1): -1},
        ],
        # L7 = A01 - A11
        [
            {(0,2): 1, (2,2): -1},
            {(0,3): 1, (2,3): -1},
            {(1,2): 1, (3,2): -1},
            {(1,3): 1, (3,3): -1},
        ],
    ]
    
    R_defs = [
        # R1 = B00 + B11
        [
            {(0,0): 1, (2,2): 1},
            {(0,1): 1, (2,3): 1},
            {(1,0): 1, (3,2): 1},
            {(1,1): 1, (3,3): 1},
        ],
        # R2 = B00
        [
            {(0,0): 1},
            {(0,1): 1},
            {(1,0): 1},
            {(1,1): 1},
        ],
        # R3 = B01 - B11
        [
            {(0,2): 1, (2,2): -1},
            {(0,3): 1, (2,3): -1},
            {(1,2): 1, (3,2): -1},
            {(1,3): 1, (3,3): -1},
        ],
        # R4 = B10 - B00
        [
            {(2,0): 1, (0,0): -1},
            {(2,1): 1, (0,1): -1},
            {(3,0): 1, (1,0): -1},
            {(3,1): 1, (1,1): -1},
        ],
        # R5 = B11
        [
            {(2,2): 1},
            {(2,3): 1},
            {(3,2): 1},
            {(3,3): 1},
        ],
        # R6 = B00 + B01
        [
            {(0,0): 1, (0,2): 1},
            {(0,1): 1, (0,3): 1},
            {(1,0): 1, (1,2): 1},
            {(1,1): 1, (1,3): 1},
        ],
        # R7 = B10 + B11
        [
            {(2,0): 1, (2,2): 1},
            {(2,1): 1, (2,3): 1},
            {(3,0): 1, (3,2): 1},
            {(3,1): 1, (3,3): 1},
        ],
    ]
    
    # Strassen scalar combination definitions
    s_defs = [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)]),  # s=0
        ([(2, 1), (3, 1)], [(0, 1)]),            # s=1
        ([(0, 1)], [(1, 1), (3, -1)]),           # s=2
        ([(3, 1)], [(2, 1), (0, -1)]),           # s=3
        ([(0, 1), (1, 1)], [(3, 1)]),            # s=4
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)]),   # s=5
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)]),   # s=6
    ]
    
    # Build U and V
    for b in range(7):
        for s in range(7):
            r = b * 7 + s
            l_coeffs, r_coeffs = s_defs[s]
            
            left_map = {}
            for pos, coeff in l_coeffs:
                for (ia, ja), c in L_defs[b][pos].items():
                    left_map[(ia, ja)] = left_map.get((ia, ja), 0) + coeff * c
            
            right_map = {}
            for pos, coeff in r_coeffs:
                for (jb, kb), c in R_defs[b][pos].items():
                    right_map[(jb, kb)] = right_map.get((jb, kb), 0) + coeff * c
            
            for (ia, ja), coeff in left_map.items():
                if coeff != 0:
                    U[ia * m + ja][r] = coeff
            
            for (jb, kb), coeff in right_map.items():
                if coeff != 0:
                    V[jb * p + kb][r] = coeff
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
