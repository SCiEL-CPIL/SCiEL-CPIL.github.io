"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
import time

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    # Strassen decomposition for 2x4 * 4x8 -> 2x8
    # We split into 8 independent 2x2 * 2x2 blocks
    # Block b: A rows 0-1, A cols 2*(b%2)..2*(b%2)+1
    #          B rows 2*(b%2)..2*(b%2)+1, B cols 2*(b//2)..2*(b//2)+1
    # Output: C row li (0-1), col 2*(b//2)..2*(b//2)+1
    
    # For each 2x2*2x2 block, use 7 Strassen terms
    # Total rank: 8 * 7 = 56

    rank = 56
    U = [[0] * rank for _ in range(n * m)]  # shape (8, 56)
    V = [[0] * rank for _ in range(m * p)]  # shape (32, 56)
    W = [[0] * rank for _ in range(p * n)]  # shape (16, 56)

    # Strassen products for 2x2 * 2x2:
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B01-B11)
    # P4 = A11(B10-B00)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    #
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6

    strassen_products = [
        # P1
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # P2
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        # P3
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1}),
        # P4
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(1,0):1, (0,0):1}),
        # P5
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        # P6
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        # P7
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1, (0,1):-1}),
    ]

    col = 0
    for block in range(8):
        # Determine which 2x2 block this is
        # b%2 gives intermediate column pair (0 or 1)
        # b//2 gives output column pair (0-3)
        ar = block % 2    # A column pair index: 0 or 1
        qc = block // 2   # output column pair index: 0-3
        
        a_col0 = 2 * ar       # A columns: a_col0, a_col0+1
        b_row0 = 2 * ar       # B rows: b_row0, b_row0+1
        b_col0 = 2 * qc       # B columns: b_col0, b_col0+1

        for u_coeff, v_coeff, w_coeff in strassen_products:
            # Fill U: A indices (li, lj) where li is A row (0-1), lj is A col (0-3)
            for (li, lj), coeff in u_coeff.items():
                gi = li           # A row
                gj = a_col0 + lj  # A column
                U[gi * m + gj][col] += coeff

            # Fill V: B indices (lj, lk) where lj is B row (0-3), lk is B col (0-7)
            for (lj, lk), coeff in v_coeff.items():
                gj = b_row0 + lj  # B row
                gk = b_col0 + lk  # B column
                V[gj * p + gk][col] += coeff

            # Fill W: C indices (lk, li) where lk is C col (0-7), li is C row (0-1)
            # W[k*n + i, t] convention
            for (lk, li), coeff in w_coeff.items():
                gk = b_col0 + lk  # C column
                gi = li           # C row
                W[gk * n + gi][col] += coeff

            col += 1

            # Time check
            if time.time() > deadline:
                # Should not happen for 56 rank, but safety
                break

        if time.time() > deadline:
            break

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
