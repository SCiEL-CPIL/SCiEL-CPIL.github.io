"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # n=3, m=3, p=4
    # rank = 28
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_term(col, u_indices, v_indices, w_indices, val=1):
        for idx in u_indices:
            U[idx][col] += val
        for idx in v_indices:
            V[idx][col] += val
        for idx in w_indices:
            W[idx][col] += val

    # Process two blocks of 2 columns for B
    for s in range(2):
        c1 = 2 * s
        c2 = 2 * s + 1
        base_col = 14 * s
        
        # Part 1: Strassen for A[0:2, 0:2] * B[0:2, c1:c2+1]
        # U indices for A[0:2, 0:2]: 0, 1, 3, 4
        # V indices for B[0:2, c1:c2+1]: c1, c2, 4+c1, 4+c2
        # W indices for C[0:2, c1:c2+1]: 3*c1, 3*c2, 3*c1+1, 3*c2+1
        
        # P1: (A00+A11)(B00+B11)
        add_term(base_col + 0, [0, 4], [c1, 4+c2], [3*c1, 3*c2+1], 1)
        
        # P2: (A10+A11)B00
        add_term(base_col + 1, [3, 4], [c1], [3*c1+1, 3*c2+1], 1)
        # Note: P2 contributes +1 to C10 and -1 to C11
        W[3*c2+1][base_col + 1] -= 1
        
        # P3: A00(B01-B11)
        add_term(base_col + 2, [0], [c2, 4+c2], [3*c2, 3*c2+1], 1)
        V[4+c2][base_col + 2] -= 1
        
        # P4: A11(B10-B00)
        add_term(base_col + 3, [4], [4+c1, c1], [3*c1, 3*c1+1], 1)
        V[c1][base_col + 3] -= 1
        
        # P5: (A00+A01)B11
        add_term(base_col + 4, [0, 1], [4+c2], [3*c1, 3*c2], 1)
        W[3*c1][base_col + 4] -= 1
        
        # P6: (A10-A00)(B00+B01)
        add_term(base_col + 5, [3, 0], [c1, c2], [3*c2+1], 1)
        U[0][base_col + 5] -= 1
        
        # P7: (A01-A11)(B10+B11)
        add_term(base_col + 6, [1, 4], [4+c1, 4+c2], [3*c1], 1)
        U[4][base_col + 6] -= 1

        # Part 2: A[0:2, 2] * B[2, c1:c2+1]
        # U: 2, 5
        # V: 8+c1, 8+c2
        # W: 3*c1, 3*c2, 3*c1+1, 3*c2+1
        add_term(base_col + 7, [2, 5], [8+c1, 8+c2], [3*c1, 3*c2, 3*c1+1, 3*c2+1], 1)

        # Part 3: A[2, 0:2] * B[0:2, c1:c2+1]
        # Q1: A20 B0,c1 -> C2,c1
        add_term(base_col + 8, [6], [c1], [3*c1+2], 1)
        # Q2: A21 B1,c1 -> C2,c1
        add_term(base_col + 9, [7], [4+c1], [3*c1+2], 1)
        # Q3: A20 B0,c2 -> C2,c2
        add_term(base_col + 10, [6], [c2], [3*c2+2], 1)
        # Q4: A21 B1,c2 -> C2,c2
        add_term(base_col + 11, [7], [4+c2], [3*c2+2], 1)

        # Part 4: A[2, 2] * B[2, c1:c2+1]
        # R1: A22 B2,c1 -> C2,c1
        add_term(base_col + 12, [8], [8+c1], [3*c1+2], 1)
        # R2: A22 B2,c2 -> C2,c2
        add_term(base_col + 13, [8], [8+c2], [3*c2+2], 1)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
