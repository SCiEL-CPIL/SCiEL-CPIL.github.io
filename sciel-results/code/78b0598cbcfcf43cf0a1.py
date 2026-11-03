"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
import time

def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=5
    # Use Strassen for 2x2 sub-block (j in {0,1}), schoolbook for j=2
    # Strassen 2x2x2: 7 products; extended to 2x2x5 still rank 7
    # Schoolbook 2x1x5: rank 10
    # Total rank: 7 + 10 = 17
    # But we need to verify correctness. Let's use a safer approach:
    # Keep schoolbook for now but try to merge some terms.
    
    # Actually, let's implement the known rank-28 decomposition:
    # Use Strassen on the 2x2 block formed by (i in {0,1}, j in {0,1})
    # and schoolbook for j=2.
    
    # For the 2x2 x 2x5 subproblem (j in {0,1}):
    # A_sub = [[a00, a01], [a10, a11]] (2x2)
    # B_sub = [[b00, b01, b02, b03, b04], [b10, b11, b12, b13, b14]] (2x5)
    # C_sub = A_sub * B_sub (2x5)
    #
    # Strassen decomposition (7 products):
    # P1 = (a00+a11) * (row0_B + row1_B)
    # P2 = (a10+a11) * row0_B
    # P3 = a00 * (row1_B - row0_B)
    # P4 = a11 * (row0_B - row1_B)
    # P5 = (a00+a01) * row1_B
    # P6 = (a10-a00) * (row0_B + row1_B)
    # P7 = (a01-a11) * (row1_B + row0_B)
    #
    # C_sub[0] = P1 - P4 + P7
    # C_sub[1] = P2 + P4
    #
    # But I verified these formulas don't work. Let me use a different approach.
    # Since I can't easily derive the correct Strassen formulas, let me just
    # return the schoolbook decomposition which is known to be correct.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
