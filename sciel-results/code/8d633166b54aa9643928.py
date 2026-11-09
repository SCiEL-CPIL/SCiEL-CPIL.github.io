"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 3
    rank = 17
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_indices, v_indices, w_indices):
        nonlocal col
        for u_idx, val in u_indices.items():
            U[u_idx][col] += val
        for v_idx, val in v_indices.items():
            V[v_idx][col] += val
        for w_idx, val in w_indices.items():
            W[w_idx][col] += val
        col += 1

    # Part 1: Strassen for A_L (2x2) * B_T1 (2x2) -> C_LT1 (2x2)
    # Local i in {0,1}, j in {0,1}, k in {0,1}
    # Global U idx: i*3 + j
    # Global V idx: j*3 + k
    # Global W idx: k*2 + i
    
    # Term 1: M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    add_term(
        {0: 1, 4: 1},  # U[0,0], U[1,1]
        {0: 1, 4: 1},  # V[0,0], V[1,1]
        {0: 1, 3: 1}   # W[0,0], W[1,1] (k=0,i=0 -> 0; k=1,i=1 -> 3)
    )
    
    # Term 2: M2 = (A10+A11)B01 -> C10 += M2, C11 -= M2
    add_term(
        {3: 1, 4: 1},  # U[1,0], U[1,1]
        {1: 1},        # V[0,1]
        {1: 1, 3: -1}  # W[0,1], W[1,1] (k=0,i=1 -> 1; k=1,i=1 -> 3)
    )
    
    # Term 3: M3 = A00(B10-B11) -> C01 += M3, C11 += M3
    add_term(
        {0: 1},        # U[0,0]
        {3: 1, 4: -1}, # V[1,0], V[1,1]
        {2: 1, 3: 1}   # W[1,0], W[1,1] (k=1,i=0 -> 2; k=1,i=1 -> 3)
    )
    
    # Term 4: M4 = A11(B01-B00) -> C00 += M4, C10 += M4
    add_term(
        {4: 1},        # U[1,1]
        {1: 1, 0: -1}, # V[0,1], V[0,0]
        {0: 1, 1: 1}   # W[0,0], W[0,1] (k=0,i=0 -> 0; k=0,i=1 -> 1)
    )
    
    # Term 5: M5 = (A00+A10)B11 -> C00 -= M5, C01 += M5
    add_term(
        {0: 1, 3: 1},  # U[0,0], U[1,0]
        {4: 1},        # V[1,1]
        {0: -1, 2: 1}  # W[0,0], W[1,0] (k=0,i=0 -> 0; k=1,i=0 -> 2)
    )
    
    # Term 6: M6 = (A10-A00)(B00+B10) -> C11 += M6
    add_term(
        {3: 1, 0: -1}, # U[1,0], U[0,0]
        {0: 1, 3: 1},  # V[0,0], V[1,0]
        {3: 1}         # W[1,1] (k=1,i=1 -> 3)
    )
    
    # Term 7: M7 = (A01-A11)(B10+B11) -> C00 += M7
    add_term(
        {1: 1, 4: -1}, # U[0,1], U[1,1]
        {3: 1, 4: 1},  # V[1,0], V[1,1]
        {0: 1}         # W[0,0] (k=0,i=0 -> 0)
    )

    # Part 2: A_L (2x2) * B_T2 (2x1) -> C_LT2 (2x1)
    # Local i in {0,1}, j in {0,1}, k=2
    # Global U idx: i*3 + j
    # Global V idx: j*3 + 2
    # Global W idx: 2*2 + i = 4 + i
    
    # Term 8: A00 * B02 -> C02
    add_term(
        {0: 1},  # U[0,0]
        {2: 1},  # V[0,2]
        {4: 1}   # W[2,0]
    )
    
    # Term 9: A01 * B12 -> C02
    add_term(
        {1: 1},  # U[0,1]
        {5: 1},  # V[1,2]
        {4: 1}   # W[2,0]
    )
    
    # Term 10: A10 * B02 -> C12
    add_term(
        {3: 1},  # U[1,0]
        {2: 1},  # V[0,2]
        {5: 1}   # W[2,1]
    )
    
    # Term 11: A11 * B12 -> C12
    add_term(
        {4: 1},  # U[1,1]
        {5: 1},  # V[1,2]
        {5: 1}   # W[2,1]
    )

    # Part 3: A_R (2x1) * B_B (1x3) -> C_RB (2x3)
    # Local i in {0,1}, j=2, k in {0,1,2}
    # Global U idx: i*3 + 2
    # Global V idx: 2*3 + k = 6 + k
    # Global W idx: k*2 + i
    
    # Term 12: A02 * B20 -> C00
    add_term(
        {2: 1},  # U[0,2]
        {6: 1},  # V[2,0]
        {0: 1}   # W[0,0]
    )
    
    # Term 13: A02 * B21 -> C01
    add_term(
        {2: 1},  # U[0,2]
        {7: 1},  # V[2,1]
        {1: 1}   # W[0,1]
    )
    
    # Term 14: A02 * B22 -> C02
    add_term(
        {2: 1},  # U[0,2]
        {8: 1},  # V[2,2]
        {2: 1}   # W[1,0]
    )
    
    # Term 15: A12 * B20 -> C10
    add_term(
        {5: 1},  # U[1,2]
        {6: 1},  # V[2,0]
        {3: 1}   # W[0,1]
    )
    
    # Term 16: A12 * B21 -> C11
    add_term(
        {5: 1},  # U[1,2]
        {7: 1},  # V[2,1]
        {4: 1}   # W[1,1]
    )
    
    # Term 17: A12 * B22 -> C12
    add_term(
        {5: 1},  # U[1,2]
        {8: 1},  # V[2,2]
        {5: 1}   # W[1,2]
    )

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
