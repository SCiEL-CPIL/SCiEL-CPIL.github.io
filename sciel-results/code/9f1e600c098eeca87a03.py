"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Using Strassen's algorithm for 2x2 blocks
    # 4 blocks of 2x2 matrices in B
    # Rank = 4 * 7 = 28
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for A (4 rows: 00, 01, 10, 11)
    # r=0: a+d
    # r=1: c+d
    # r=2: a
    # r=3: d
    # r=4: a+b
    # r=5: c-a
    # r=6: b-d
    U_coeffs = [
        [1, 0, 1, 0], # r=0: a+d -> idx 0,3? No. 
        # Let's define U_coeffs[r][row]
        # Row 0: a, Row 1: b, Row 2: c, Row 3: d
    ]
    
    # Correct U coefficients
    U_mat = [
        [1, 0, 1, 0, 1, -1, 0], # r=0..6 for row 0 (a)
        [0, 1, 0, 0, 1, 0, 1],  # r=0..6 for row 1 (b)
        [0, 1, 0, 0, 0, 1, 0],  # r=0..6 for row 2 (c)
        [1, 1, 0, 1, 0, 0, -1]  # r=0..6 for row 3 (d)
    ]
    # Check:
    # r0: a+d -> U[0]=1, U[3]=1. Correct.
    # r1: c+d -> U[2]=1, U[3]=1. Correct.
    # r2: a -> U[0]=1. Correct.
    # r3: d -> U[3]=1. Correct.
    # r4: a+b -> U[0]=1, U[1]=1. Correct.
    # r5: c-a -> U[2]=1, U[0]=-1. Correct.
    # r6: b-d -> U[1]=1, U[3]=-1. Correct.

    for b in range(4):
        for r in range(7):
            c = b * 7 + r
            
            # Fill U
            for i in range(4):
                U[i][c] = U_mat[i][r]
                
            # Fill V
            # V rows: j*8 + k
            # r0: e+h -> B[0, 2b] + B[1, 2b+1]
            # r1: e -> B[0, 2b]
            # r2: f-h -> B[0, 2b+1] - B[1, 2b+1]
            # r3: g-e -> B[1, 2b] - B[0, 2b]
            # r4: h -> B[1, 2b+1]
            # r5: e+f -> B[0, 2b] + B[0, 2b+1]
            # r6: g+h -> B[1, 2b] + B[1, 2b+1]
            
            if r == 0:
                V[0 * 8 + 2*b][c] = 1
                V[1 * 8 + 2*b + 1][c] = 1
            elif r == 1:
                V[0 * 8 + 2*b][c] = 1
            elif r == 2:
                V[0 * 8 + 2*b + 1][c] = 1
                V[1 * 8 + 2*b + 1][c] = -1
            elif r == 3:
                V[1 * 8 + 2*b][c] = 1
                V[0 * 8 + 2*b][c] = -1
            elif r == 4:
                V[1 * 8 + 2*b + 1][c] = 1
            elif r == 5:
                V[0 * 8 + 2*b][c] = 1
                V[0 * 8 + 2*b + 1][c] = 1
            elif r == 6:
                V[1 * 8 + 2*b][c] = 1
                V[1 * 8 + 2*b + 1][c] = 1
                
            # Fill W
            # W rows: k*2 + i
            # C[0, 2b] = r0 + r3 - r4 + r6
            # C[0, 2b+1] = r2 + r4
            # C[1, 2b] = r1 + r3
            # C[1, 2b+1] = r0 - r1 + r2 + r5
            
            if r == 0:
                W[2*b * 2 + 0][c] = 1
                W[(2*b + 1) * 2 + 1][c] = 1
            elif r == 1:
                W[2*b * 2 + 1][c] = 1
                W[(2*b + 1) * 2 + 1][c] = -1
            elif r == 2:
                W[(2*b + 1) * 2 + 0][c] = 1
                W[(2*b + 1) * 2 + 1][c] = 1
            elif r == 3:
                W[2*b * 2 + 0][c] = 1
                W[2*b * 2 + 1][c] = 1
            elif r == 4:
                W[2*b * 2 + 0][c] = -1
                W[(2*b + 1) * 2 + 0][c] = 1
            elif r == 5:
                W[(2*b + 1) * 2 + 1][c] = 1
            elif r == 6:
                W[2*b * 2 + 0][c] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
