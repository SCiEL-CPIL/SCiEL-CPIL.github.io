"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 17
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to set a rank-1 term
    def set_term(u_idx, v_idx, w_idx, col, val_u, val_v, val_w):
        U[u_idx][col] = val_u
        V[v_idx][col] = val_v
        W[w_idx][col] = val_w

    # Indices for U (2x3): i*m + j
    # u00=0, u01=1, u02=2, u10=3, u11=4, u12=5
    # Indices for V (3x3): j*p + k
    # v00=0, v01=1, v02=2, v10=3, v11=4, v12=5, v20=6, v21=7, v22=8
    # Indices for W (3x2): k*n + i
    # w00=0, w01=1, w02=2, w10=3, w11=4, w12=5

    # Strassen terms for core 2x2x2 (using cols 0,1 of A and rows 0,1 cols 0,1 of B)
    # S1: (a00+a11)(b00+b11) -> c00, c11
    set_term(0, 0, 0, 0, 1, 1, 1) # a00, b00, c00
    set_term(4, 0, 0, 0, 1, 0, 0) # a11, b00, c00 (part of S1)
    # Wait, S1 is (a00+a11)(b00+b11).
    # U: a00+a11 -> U[0]=1, U[4]=1
    # V: b00+b11 -> V[0]=1, V[4]=1
    # W: c00+c11 -> W[0]=1, W[4]=1
    
    # Let's reset and do it properly
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_term(u_indices, v_indices, w_indices, col, sign=1):
        for u_idx in u_indices:
            U[u_idx][col] += sign
        for v_idx in v_indices:
            V[v_idx][col] += sign
        for w_idx in w_indices:
            W[w_idx][col] += sign

    # S1: (a00+a11)(b00+b11) -> c00, c11
    add_term([0, 4], [0, 4], [0, 4], 0)
    
    # S2: (a10+a11)b00 -> c10, -c11
    add_term([3, 4], [0], [3], 1)
    add_term([], [], [4], 1, -1)
    
    # S3: a00(b10-b11) -> c01, c11
    add_term([0], [3], [1], 2)
    add_term([], [4], [], 2, -1)
    add_term([], [], [4], 2)
    
    # S4: a11(b01-b00) -> -c00, c01
    add_term([4], [1], [1], 3)
    add_term([], [0], [0], 3, -1)
    
    # S5: (a00+a01)b11 -> c00
    add_term([0, 1], [4], [0], 4)
    
    # S6: (a10-a00)(b00+b01) -> c11
    add_term([3], [0, 1], [4], 5)
    add_term([0], [0, 1], [], 5, -1)
    
    # S7: (a01-a11)(b10+b11) -> -c00, c10
    add_term([1], [3, 4], [3], 6)
    add_term([4], [3, 4], [0], 6, -1)
    
    # Extra terms for C_L (col 2 of A, row 2 of B)
    # T8: a02 b20 -> c00
    add_term([2], [6], [0], 7)
    
    # T9: a02 b21 -> c01
    add_term([2], [7], [1], 8)
    
    # T10: a12 b20 -> c10
    add_term([5], [6], [3], 9)
    
    # T11: a12 b21 -> c11
    add_term([5], [7], [4], 10)
    
    # Terms for C_R (col 2 of B)
    # T12: a00 b02 -> c02
    add_term([0], [2], [2], 11)
    
    # T13: a01 b12 -> c02
    add_term([1], [5], [2], 12)
    
    # T14: a02 b22 -> c02
    add_term([2], [8], [2], 13)
    
    # T15: a10 b02 -> c12
    add_term([3], [2], [5], 14)
    
    # T16: a11 b12 -> c12
    add_term([4], [5], [5], 15)
    
    # T17: a12 b22 -> c12
    add_term([5], [8], [5], 16)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
