import time

DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS
    # Rank calculation
    rank_part1 = n * 1 * p  # 20
    num_blocks = p // 2     # 5
    rank_part2 = num_blocks * 7 # 35
    rank = rank_part1 + rank_part2 # 55

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    # Part 1: j=0 term
    # A[i,0] * B[0,k]
    for i in range(n):
        for k in range(p):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1

    # Part 2: j=1,2 terms using Strassen
    # A' is A[:, 1:3] (indices 1, 2 in m)
    # B' is B[1:3, :] (indices 1, 2 in m)
    
    # Strassen definitions for 2x2 matrices
    # A' = [[A01, A02], [A11, A12]]
    # B_q = [[B1_2q, B1_2q+1], [B2_2q, B2_2q+1]]
    
    # We define the coefficients for U, V, W for each of the 7 Strassen products
    # U coeffs: dict { (i, j_global) : coeff }
    # V coeffs: dict { (j_global, k) : coeff }
    # W coeffs: dict { (k, i) : coeff }
    
    # Note: j_global for A' is 1 and 2.
    # Note: j_global for B' is 1 and 2.
    
    # Term 1: (A01 + A12) * (B1_2q + B2_2q+1)
    # U: A01 (i=0, j=1), A12 (i=1, j=2)
    # V: B1_2q (j=1, k=2q), B2_2q+1 (j=2, k=2q+1)
    # W: C0_2q (+1), C1_2q+1 (-1) -> (k=2q, i=0), (k=2q+1, i=1)
    
    # Term 2: (A11 + A12) * B1_2q
    # U: A11 (i=1, j=1), A12 (i=1, j=2)
    # V: B1_2q (j=1, k=2q)
    # W: C1_2q (+1), C1_2q+1 (-1) -> (k=2q, i=1), (k=2q+1, i=1)
    
    # Term 3: A01 * (B1_2q+1 - B2_2q+1)
    # U: A01 (i=0, j=1)
    # V: B1_2q+1 (j=1, k=2q+1), B2_2q+1 (j=2, k=2q+1)
    # W: C1_2q (+1) -> (k=2q, i=1)
    
    # Term 4: A12 * (B2_2q - B1_2q)
    # U: A12 (i=1, j=2)
    # V: B2_2q (j=2, k=2q), B1_2q (j=1, k=2q)
    # W: C0_2q+1 (+1) -> (k=2q+1, i=0)
    
    # Term 5: (A01 + A02) * B2_2q+1
    # U: A01 (i=0, j=1), A02 (i=0, j=2)
    # V: B2_2q+1 (j=2, k=2q+1)
    # W: C0_2q (-1), C0_2q+1 (+1) -> (k=2q, i=0), (k=2q+1, i=0)
    
    # Term 6: (A11 - A01) * (B1_2q + B1_2q+1)
    # U: A11 (i=1, j=1), A01 (i=0, j=1)
    # V: B1_2q (j=1, k=2q), B1_2q+1 (j=1, k=2q+1)
    # W: C1_2q+1 (+1) -> (k=2q+1, i=1)
    
    # Term 7: (A02 - A12) * (B2_2q + B2_2q+1)
    # U: A02 (i=0, j=2), A12 (i=1, j=2)
    # V: B2_2q (j=2, k=2q), B2_2q+1 (j=2, k=2q+1)
    # W: C0_2q (+1) -> (k=2q, i=0)

    # Helper to add a term
    def add_term(u_coeffs, v_coeffs, w_coeffs):
        nonlocal col
        for (i, j), coeff in u_coeffs.items():
            U[i * m + j][col] += coeff
        for (j, k), coeff in v_coeffs.items():
            V[j * p + k][col] += coeff
        for (k, i), coeff in w_coeffs.items():
            W[k * n + i][col] += coeff
        col += 1

    for q in range(num_blocks):
        k0 = 2 * q
        k1 = 2 * q + 1
        
        # Term 1
        add_term(
            {(0, 1): 1, (1, 2): 1},
            {(1, k0): 1, (2, k1): 1},
            {(k0, 0): 1, (k1, 1): -1}
        )
        
        # Term 2
        add_term(
            {(1, 1): 1, (1, 2): 1},
            {(1, k0): 1},
            {(k0, 1): 1, (k1, 1): -1}
        )
        
        # Term 3
        add_term(
            {(0, 1): 1},
            {(1, k1): 1, (2, k1): -1},
            {(k0, 1): 1}
        )
        
        # Term 4
        add_term(
            {(1, 2): 1},
            {(2, k0): 1, (1, k0): -1},
            {(k1, 0): 1}
        )
        
        # Term 5
        add_term(
            {(0, 1): 1, (0, 2): 1},
            {(2, k1): 1},
            {(k0, 0): -1, (k1, 0): 1}
        )
        
        # Term 6
        add_term(
            {(1, 1): 1, (0, 1): -1},
            {(1, k0): 1, (1, k1): 1},
            {(k1, 1): 1}
        )
        
        # Term 7
        add_term(
            {(0, 2): 1, (1, 2): -1},
            {(2, k0): 1, (2, k1): 1},
            {(k0, 0): 1}
        )

    return {"U": U, "V": V, "W": W}
