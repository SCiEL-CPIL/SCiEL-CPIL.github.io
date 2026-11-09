"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Total rank 46
    rank = 46
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen vectors for 2x2x2
    # u, v, w are vectors of length 4
    # Order: 00, 01, 10, 11
    strassen_terms = [
        # m1 = (a+d)(e+h)
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # m2 = (c+d)e
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        # m3 = a(f-h)
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        # m4 = d(g-e)
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]),
        # m5 = (a+b)h
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        # m6 = (c-a)(e+f)
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        # m7 = (b-d)(g+h)
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])
    ]
    
    def add_strassen_block(a_idx, b_idx, c_idx):
        nonlocal col
        for u, v, w in strassen_terms:
            for i in range(4):
                U[a_idx[i]][col] += u[i]
                V[b_idx[i]][col] += v[i]
                W[c_idx[i]][col] += w[i]
            col += 1

    def add_schoolbook_block(a_idx, b_idx, c_idx):
        # a_idx, b_idx, c_idx are lists of tuples (global_idx, local_idx)
        # Or just handle specific cases
        pass

    # Block 1: A(0,1; 0,1) * B(0,1; 0,1)
    # A idx: 0,1,5,6
    # B idx: 0,1,5,6
    # C idx: 0,2,1,3
    add_strassen_block(
        [0, 1, 5, 6],
        [0, 1, 5, 6],
        [0, 2, 1, 3]
    )
    
    # Block 2: A(0,1; 0,1) * B(0,1; 2,3)
    # A idx: 0,1,5,6
    # B idx: 2,3,7,8
    # C idx: 4,6,5,7
    add_strassen_block(
        [0, 1, 5, 6],
        [2, 3, 7, 8],
        [4, 6, 5, 7]
    )
    
    # Block 3: A(0,1; 2,3) * B(2,3; 0,1)
    # A idx: 2,3,7,8
    # B idx: 10,11,15,16
    # C idx: 0,2,1,3
    add_strassen_block(
        [2, 3, 7, 8],
        [10, 11, 15, 16],
        [0, 2, 1, 3]
    )
    
    # Block 4: A(0,1; 2,3) * B(2,3; 2,3)
    # A idx: 2,3,7,8
    # B idx: 12,13,17,18
    # C idx: 4,6,5,7
    add_strassen_block(
        [2, 3, 7, 8],
        [12, 13, 17, 18],
        [4, 6, 5, 7]
    )
    
    # Schoolbook blocks
    # Helper for schoolbook 2x2x1, 2x1x2, 2x1x1
    def add_term(u_idx, v_idx, w_idx, val=1):
        nonlocal col
        U[u_idx][col] = val
        V[v_idx][col] = val
        W[w_idx][col] = val
        col += 1

    # Block 5: A(0,1; 0,1) * B(0,1; 4) -> C(0,1; 4)
    # A idx: 0,1,5,6
    # B idx: 4,9
    # C idx: 8,9
    # Terms:
    # A00*B04 -> C04 (U[0], V[4], W[8])
    # A01*B14 -> C04 (U[1], V[9], W[8])
    # A10*B04 -> C14 (U[5], V[4], W[9])
    # A11*B14 -> C14 (U[6], V[9], W[9])
    add_term(0, 4, 8)
    add_term(1, 9, 8)
    add_term(5, 4, 9)
    add_term(6, 9, 9)
    
    # Block 6: A(0,1; 2,3) * B(2,3; 4) -> C(0,1; 4)
    # A idx: 2,3,7,8
    # B idx: 14,19
    # C idx: 8,9
    add_term(2, 14, 8)
    add_term(3, 19, 8)
    add_term(7, 14, 9)
    add_term(8, 19, 9)
    
    # Block 7: A(0,1; 4) * B(4; 0,1) -> C(0,1; 0,1)
    # A idx: 4,9
    # B idx: 20,21
    # C idx: 0,2,1,3
    # Terms:
    # A04*B40 -> C00 (U[4], V[20], W[0])
    # A04*B41 -> C01 (U[4], V[21], W[2])
    # A14*B40 -> C10 (U[9], V[20], W[1])
    # A14*B41 -> C11 (U[9], V[21], W[3])
    add_term(4, 20, 0)
    add_term(4, 21, 2)
    add_term(9, 20, 1)
    add_term(9, 21, 3)
    
    # Block 8: A(0,1; 4) * B(4; 2,3) -> C(0,1; 2,3)
    # A idx: 4,9
    # B idx: 22,23
    # C idx: 4,6,5,7
    add_term(4, 22, 4)
    add_term(4, 23, 6)
    add_term(9, 22, 5)
    add_term(9, 23, 7)
    
    # Block 9: A(0,1; 4) * B(4; 4) -> C(0,1; 4)
    # A idx: 4,9
    # B idx: 24
    # C idx: 8,9
    add_term(4, 24, 8)
    add_term(9, 24, 9)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
