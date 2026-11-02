"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 17
    
    # U shape: (n*m, rank) = (6, 17)
    # V shape: (m*p, rank) = (9, 17)
    # W shape: (p*n, rank) = (6, 17)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to set a term
    # u_indices: list of (row, coeff)
    # v_indices: list of (row, coeff)
    # w_indices: list of (row, coeff)
    def add_term(col, u_idx, v_idx, w_idx):
        for r, c in u_idx:
            U[r][col] += c
        for r, c in v_idx:
            V[r][col] += c
        for r, c in w_idx:
            W[r][col] += c

    # --- Strassen Part (Cols 0-6) ---
    # Handles A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # A indices: i in {0,1}, j in {0,1} -> row i*3+j
    # B indices: j in {0,1}, k in {0,1} -> row j*3+k
    # W indices: k in {0,1}, i in {0,1} -> row k*2+i
    
    # M1 = (A00 + A11)(B00 + B11)
    add_term(0,
             [(0, 1), (4, 1)],       # A00, A11
             [(0, 1), (4, 1)],       # B00, B11
             [(0, 1), (1, 1), (2, 1), (3, 1)]) # C00, C01, C10, C11

    # M2 = (A10 + A11)B00
    add_term(1,
             [(3, 1), (4, 1)],       # A10, A11
             [(0, 1)],               # B00
             [(2, 1), (3, -1)])      # C10, C11

    # M3 = A00(B01 - B11)
    add_term(2,
             [(0, 1)],               # A00
             [(1, 1), (4, -1)],      # B01, B11
             [(1, 1), (3, 1)])       # C01, C11

    # M4 = A11(B10 - B00)
    add_term(3,
             [(4, 1)],               # A11
             [(3, 1), (0, -1)],      # B10, B00
             [(0, 1), (2, 1)])       # C00, C10

    # M5 = (A00 + A10)B11
    add_term(4,
             [(0, 1), (3, 1)],       # A00, A10
             [(4, 1)],               # B11
             [(0, -1), (1, 1)])      # C00, C01

    # M6 = (A11 - A00)(B00 + B01)
    add_term(5,
             [(4, 1), (0, -1)],      # A11, A00
             [(0, 1), (1, 1)],       # B00, B01
             [(3, 1)])               # C11

    # M7 = (A01 - A11)(B10 + B11)
    add_term(6,
             [(1, 1), (4, -1)],      # A01, A11
             [(3, 1), (4, 1)],       # B10, B11
             [(0, 1)])               # C00

    # --- Schoolbook Part 1 (Cols 7-10) ---
    # Handles A[0:2, 0:2] * B[0:2, 2] -> C[0:2, 2]
    # A indices: i in {0,1}, j in {0,1}
    # B indices: j in {0,1}, k=2 -> row j*3+2
    # W indices: k=2, i in {0,1} -> row 2*2+i = 4, 5
    
    # A00 B02 -> C02
    add_term(7,
             [(0, 1)],               # A00
             [(2, 1)],               # B02 (0*3+2)
             [(4, 1)])               # C02 (2*2+0)

    # A01 B12 -> C02
    add_term(8,
             [(1, 1)],               # A01
             [(5, 1)],               # B12 (1*3+2)
             [(4, 1)])               # C02

    # A10 B02 -> C12
    add_term(9,
             [(3, 1)],               # A10
             [(2, 1)],               # B02
             [(5, 1)])               # C12 (2*2+1)

    # A11 B12 -> C12
    add_term(10,
             [(4, 1)],               # A11
             [(5, 1)],               # B12
             [(5, 1)])               # C12

    # --- Schoolbook Part 2 (Cols 11-16) ---
    # Handles A[0:2, 2] * B[2, 0:3] -> C[0:2, 0:3]
    # A indices: i in {0,1}, j=2 -> row i*3+2
    # B indices: j=2, k in {0,1,2} -> row 2*3+k = 6, 7, 8
    # W indices: k in {0,1,2}, i in {0,1} -> row k*2+i
    
    # A02 B20 -> C00
    add_term(11,
             [(2, 1)],               # A02 (0*3+2)
             [(6, 1)],               # B20 (2*3+0)
             [(0, 1)])               # C00 (0*2+0)

    # A02 B21 -> C01
    add_term(12,
             [(2, 1)],               # A02
             [(7, 1)],               # B21 (2*3+1)
             [(1, 1)])               # C01 (1*2+0)

    # A02 B22 -> C02
    add_term(13,
             [(2, 1)],               # A02
             [(8, 1)],               # B22 (2*3+2)
             [(4, 1)])               # C02 (2*2+0)

    # A12 B20 -> C10
    add_term(14,
             [(5, 1)],               # A12 (1*3+2)
             [(6, 1)],               # B20
             [(2, 1)])               # C10 (0*2+1)

    # A12 B21 -> C11
    add_term(15,
             [(5, 1)],               # A12
             [(7, 1)],               # B21
             [(3, 1)])               # C11 (1*2+1)

    # A12 B22 -> C12
    add_term(16,
             [(5, 1)],               # A12
             [(8, 1)],               # B22
             [(5, 1)])               # C12 (2*2+1)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
