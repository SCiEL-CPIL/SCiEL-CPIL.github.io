"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    # Rank 22 decomposition for 3x3 matrix multiplication.
    # Uses Strassen's rank-7 algorithm for the top-left 2x2 block.
    # Uses schoolbook rank-3 algorithms for the remaining 5 output entries.
    n, m, p = DIMENSIONS
    rank = 22
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to set a term
    # u_indices: list of (index, value) for U
    # v_indices: list of (index, value) for V
    # w_indices: list of (index, value) for W
    def add_term(col, u_list, v_list, w_list):
        for idx, val in u_list:
            U[idx][col] = val
        for idx, val in v_list:
            V[idx][col] = val
        for idx, val in w_list:
            W[idx][col] = val

    col = 0
    
    # --- Strassen's 7 terms for C[0:2, 0:2] ---
    # Indices for 2x2 block:
    # A: 0(0,0), 1(0,1), 2(1,0), 3(1,1)
    # B: 0(0,0), 1(0,1), 2(1,0), 3(1,1)
    # C: 0(0,0), 1(0,1), 2(1,0), 3(1,1) -> Global C: 0, 1, 3, 4
    
    # 1. M1 = (A00+A11)(B00+B11) -> C00, C11
    # Global: A:0,3; B:0,3; C:0,4
    add_term(col, [(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (4, 1)])
    col += 1
    
    # 2. M2 = (A10+A11)B00 -> C10, C11
    # Global: A:2,3; B:0; C:3,4 (C10 is global 3, C11 is global 4)
    # C10 gets +M2, C11 gets -M2
    add_term(col, [(2, 1), (3, 1)], [(0, 1)], [(3, 1), (4, -1)])
    col += 1
    
    # 3. M3 = A00(B10-B11) -> C10
    # Global: A:0; B:2,3; C:3
    add_term(col, [(0, 1)], [(2, 1), (3, -1)], [(3, 1)])
    col += 1
    
    # 4. M4 = A11(B01-B00) -> C00, C01
    # Global: A:3; B:0,1; C:0,1 (C00 is global 0, C01 is global 1)
    # C00 gets -M4, C01 gets +M4
    add_term(col, [(3, 1)], [(0, -1), (1, 1)], [(0, -1), (1, 1)])
    col += 1
    
    # 5. M5 = (A00+A01)B11 -> C01
    # Global: A:0,1; B:3; C:1
    add_term(col, [(0, 1), (1, 1)], [(3, 1)], [(1, 1)])
    col += 1
    
    # 6. M6 = (A10-A00)(B00+B01) -> C11
    # Global: A:0,2; B:0,1; C:4
    add_term(col, [(0, -1), (2, 1)], [(0, 1), (1, 1)], [(4, 1)])
    col += 1
    
    # 7. M7 = (A01-A11)(B10+B11) -> C00
    # Global: A:1,3; B:2,3; C:0
    add_term(col, [(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)])
    col += 1

    # --- Schoolbook terms for remaining C entries ---
    
    # C02 (Global 2): A00*B02 + A01*B12 + A02*B22
    # A:0,1,2; B:2,5,8; C:2
    add_term(col, [(0, 1)], [(2, 1)], [(2, 1)])
    col += 1
    add_term(col, [(1, 1)], [(5, 1)], [(2, 1)])
    col += 1
    add_term(col, [(2, 1)], [(8, 1)], [(2, 1)])
    col += 1

    # C12 (Global 5): A10*B02 + A11*B12 + A12*B22
    # A:3,4,5; B:2,5,8; C:5
    add_term(col, [(3, 1)], [(2, 1)], [(5, 1)])
    col += 1
    add_term(col, [(4, 1)], [(5, 1)], [(5, 1)])
    col += 1
    add_term(col, [(5, 1)], [(8, 1)], [(5, 1)])
    col += 1

    # C20 (Global 6): A20*B00 + A21*B10 + A22*B20
    # A:6,7,8; B:0,3,6; C:6
    add_term(col, [(6, 1)], [(0, 1)], [(6, 1)])
    col += 1
    add_term(col, [(7, 1)], [(3, 1)], [(6, 1)])
    col += 1
    add_term(col, [(8, 1)], [(6, 1)], [(6, 1)])
    col += 1

    # C21 (Global 7): A20*B01 + A21*B11 + A22*B21
    # A:6,7,8; B:1,4,7; C:7
    add_term(col, [(6, 1)], [(1, 1)], [(7, 1)])
    col += 1
    add_term(col, [(7, 1)], [(4, 1)], [(7, 1)])
    col += 1
    add_term(col, [(8, 1)], [(7, 1)], [(7, 1)])
    col += 1

    # C22 (Global 8): A20*B02 + A21*B12 + A22*B22
    # A:6,7,8; B:2,5,8; C:8
    add_term(col, [(6, 1)], [(2, 1)], [(8, 1)])
    col += 1
    add_term(col, [(7, 1)], [(5, 1)], [(8, 1)])
    col += 1
    add_term(col, [(8, 1)], [(8, 1)], [(8, 1)])
    col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
