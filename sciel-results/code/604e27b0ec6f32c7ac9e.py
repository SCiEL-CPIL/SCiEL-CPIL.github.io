"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 17
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to set values
    def set_term(r, u_indices, v_indices, w_indices):
        for idx in u_indices:
            U[idx[0] * m + idx[1]][r] = idx[2]
        for idx in v_indices:
            V[idx[0] * p + idx[1]][r] = idx[2]
        for idx in w_indices:
            W[idx[0] * n + idx[1]][r] = idx[2]

    # Part 1: Strassen for A[0:2, 0:2] * B[0:2, 0:2]
    # M1 = (A00+A11)(B00+B11) -> C00, C11
    set_term(0, [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)])
    # M2 = (A10+A11)B00 -> C10
    set_term(1, [(1,0,1), (1,1,1)], [(0,0,1)], [(0,1,1)])
    # M3 = A00(B01-B11) -> C01, -C11
    set_term(2, [(0,0,1)], [(0,1,1), (1,1,-1)], [(1,0,1), (1,1,-1)])
    # M4 = A11(B10-B00) -> -C00, C10
    set_term(3, [(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,-1), (0,1,1)])
    # M5 = (A00+A01)B11 -> C01
    set_term(4, [(0,0,1), (0,1,1)], [(1,1,1)], [(1,0,1)])
    # M6 = (A10-A00)(B00+B01) -> C00
    set_term(5, [(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(0,0,1)])
    # M7 = (A01-A11)(B10+B11) -> C11
    set_term(6, [(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(1,1,1)])

    # Part 2: A[:, 2] * B[2, 0:2]
    # T8 = A02 B20 -> C00
    set_term(7, [(0,2,1)], [(2,0,1)], [(0,0,1)])
    # T9 = A02 B21 -> C01
    set_term(8, [(0,2,1)], [(2,1,1)], [(1,0,1)])
    # T10 = A12 B20 -> C10
    set_term(9, [(1,2,1)], [(2,0,1)], [(0,1,1)])
    # T11 = A12 B21 -> C11
    set_term(10, [(1,2,1)], [(2,1,1)], [(1,1,1)])

    # Part 3: A[0:2, 0:2] * B[0:2, 2]
    # T12 = A00 B02 -> C02
    set_term(11, [(0,0,1)], [(0,2,1)], [(2,0,1)])
    # T13 = A01 B12 -> C02
    set_term(12, [(0,1,1)], [(1,2,1)], [(2,0,1)])
    # T14 = A10 B02 -> C12
    set_term(13, [(1,0,1)], [(0,2,1)], [(2,1,1)])
    # T15 = A11 B12 -> C12
    set_term(14, [(1,1,1)], [(1,2,1)], [(2,1,1)])

    # Part 4: A[:, 2] * B[2, 2]
    # T16 = A02 B22 -> C02
    set_term(15, [(0,2,1)], [(2,2,1)], [(2,0,1)])
    # T17 = A12 B22 -> C12
    set_term(16, [(1,2,1)], [(2,2,1)], [(2,1,1)])

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
