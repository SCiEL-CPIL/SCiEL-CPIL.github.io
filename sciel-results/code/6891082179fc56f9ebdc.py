"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 100
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0

    def add_term(u_indices, v_indices, w_indices, signs_u=None, signs_v=None, signs_w=None):
        nonlocal col
        if signs_u is None: signs_u = [1] * len(u_indices)
        if signs_v is None: signs_v = [1] * len(v_indices)
        if signs_w is None: signs_w = [1] * len(w_indices)
        
        for idx, sign in zip(u_indices, signs_u):
            U[idx][col] += sign
        for idx, sign in zip(v_indices, signs_v):
            V[idx][col] += sign
        for idx, sign in zip(w_indices, signs_w):
            W[idx][col] += sign
        col += 1

    def get_A(i_b, j_b, i_l, j_l):
        return (2 * i_b + i_l) * m + (2 * j_b + j_l)

    def get_B(j_b, k_start, j_l, k_l):
        return (2 * j_b + j_l) * p + (k_start + k_l)

    def get_C(i_b, k_start, i_l, k_l):
        return (k_start + k_l) * n + (2 * i_b + i_l)

    def add_strassen_block(i_b, j_b, k_start):
        # Adds 7 terms for 2x2x2 multiplication
        # A(2x2) * B(2x2) -> C(2x2)
        # Local indices 0,1
        
        # M1 = (A00+A11)(B00+B11) -> C00+=1, C11+=1
        add_term(
            [get_A(i_b, j_b, 0, 0), get_A(i_b, j_b, 1, 1)],
            [get_B(j_b, k_start, 0, 0), get_B(j_b, k_start, 1, 1)],
            [get_C(i_b, k_start, 0, 0), get_C(i_b, k_start, 1, 1)]
        )
        
        # M2 = (A10+A11)B00 -> C10+=1, C11-=1
        add_term(
            [get_A(i_b, j_b, 1, 0), get_A(i_b, j_b, 1, 1)],
            [get_B(j_b, k_start, 0, 0)],
            [get_C(i_b, k_start, 0, 1), get_C(i_b, k_start, 1, 1)],
            signs_w=[1, -1]
        )
        
        # M3 = A00(B10-B00) -> C01+=1, C11+=1
        add_term(
            [get_A(i_b, j_b, 0, 0)],
            [get_B(j_b, k_start, 1, 0), get_B(j_b, k_start, 0, 0)],
            [get_C(i_b, k_start, 1, 0), get_C(i_b, k_start, 1, 1)],
            signs_v=[1, -1]
        )
        
        # M4 = A11(B11-B00) -> C00+=1, C10+=1
        add_term(
            [get_A(i_b, j_b, 1, 1)],
            [get_B(j_b, k_start, 1, 1), get_B(j_b, k_start, 0, 0)],
            [get_C(i_b, k_start, 0, 0), get_C(i_b, k_start, 0, 1)],
            signs_v=[1, -1]
        )
        
        # M5 = (A00+A01)B11 -> C00-=1, C01+=1
        add_term(
            [get_A(i_b, j_b, 0, 0), get_A(i_b, j_b, 0, 1)],
            [get_B(j_b, k_start, 1, 1)],
            [get_C(i_b, k_start, 0, 0), get_C(i_b, k_start, 1, 0)],
            signs_w=[-1, 1]
        )
        
        # M6 = (A10-A00)(B00+B01) -> C11+=1
        add_term(
            [get_A(i_b, j_b, 1, 0), get_A(i_b, j_b, 0, 0)],
            [get_B(j_b, k_start, 0, 0), get_B(j_b, k_start, 0, 1)],
            [get_C(i_b, k_start, 1, 1)],
            signs_u=[1, -1]
        )
        
        # M7 = (A01-A11)(B10+B11) -> C00+=1
        add_term(
            [get_A(i_b, j_b, 0, 1), get_A(i_b, j_b, 1, 1)],
            [get_B(j_b, k_start, 1, 0), get_B(j_b, k_start, 1, 1)],
            [get_C(i_b, k_start, 0, 0)],
            signs_u=[1, -1]
        )

    def add_schoolbook_221_block(i_b, j_b, k_start):
        # Adds 4 terms for 2x2x1 multiplication
        # A(2x2) * B(2x1) -> C(2x1)
        # B is vector, C is vector
        
        # A00 * B0 -> C0
        add_term(
            [get_A(i_b, j_b, 0, 0)],
            [get_B(j_b, k_start, 0, 0)],
            [get_C(i_b, k_start, 0, 0)]
        )
        
        # A01 * B1 -> C0
        add_term(
            [get_A(i_b, j_b, 0, 1)],
            [get_B(j_b, k_start, 1, 0)],
            [get_C(i_b, k_start, 0, 0)]
        )
        
        # A10 * B0 -> C1
        add_term(
            [get_A(i_b, j_b, 1, 0)],
            [get_B(j_b, k_start, 0, 0)],
            [get_C(i_b, k_start, 1, 0)]
        )
        
        # A11 * B1 -> C1
        add_term(
            [get_A(i_b, j_b, 1, 1)],
            [get_B(j_b, k_start, 1, 0)],
            [get_C(i_b, k_start, 1, 0)]
        )

    for i_b in range(2):
        for j_b in range(2):
            add_strassen_block(i_b, j_b, 0)
            add_strassen_block(i_b, j_b, 2)
            add_strassen_block(i_b, j_b, 4)
            add_schoolbook_221_block(i_b, j_b, 6)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
