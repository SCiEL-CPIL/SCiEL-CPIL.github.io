DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=14
    # Rank = 7 * 7 = 49
    rank = 49
    U = [[0] * rank for _ in range(n * m)]  # 4 x 49
    V = [[0] * rank for _ in range(m * p)]  # 28 x 49
    W = [[0] * rank for _ in range(p * n)]  # 28 x 49

    # Strassen factors for 2x2 multiplication
    # s=0: M1, s=1: M2, s=2: M3, s=3: M4, s=4: M5, s=5: M6, s=6: M7
    
    for t in range(7):
        for s in range(7):
            r = 7 * t + s
            if s == 0:
                # M1 = (A00 + A11)(B00 + B11)
                U[0][r] = 1  # A00
                U[3][r] = 1  # A11
                V[2*t][r] = 1       # B00 (j=0, k=2t)
                V[14 + 2*t + 1][r] = 1 # B11 (j=1, k=2t+1)
                W[4*t][r] = 1       # C00 (k=2t, i=0)
                W[4*t + 3][r] = 1   # C11 (k=2t+1, i=1)
            elif s == 1:
                # M2 = (A10 + A11)B00
                U[2][r] = 1  # A10
                U[3][r] = 1  # A11
                V[2*t][r] = 1       # B00
                W[4*t + 1][r] = 1   # C10 (k=2t, i=1)
                W[4*t + 3][r] = -1  # C11
            elif s == 2:
                # M3 = A00(B10 - B11)
                U[0][r] = 1  # A00
                V[14 + 2*t][r] = 1       # B10
                V[14 + 2*t + 1][r] = -1  # B11
                W[4*t + 2][r] = 1        # C01 (k=2t+1, i=0)
                W[4*t + 3][r] = 1        # C11
            elif s == 3:
                # M4 = A11(B01 - B00)
                U[3][r] = 1  # A11
                V[2*t + 1][r] = 1     # B01
                V[2*t][r] = -1        # B00
                W[4*t][r] = 1         # C00
                W[4*t + 1][r] = 1     # C10
            elif s == 4:
                # M5 = (A00 + A01)B11
                U[0][r] = 1  # A00
                U[1][r] = 1  # A01
                V[14 + 2*t + 1][r] = 1 # B11
                W[4*t][r] = -1         # C00
                W[4*t + 2][r] = 1      # C01
            elif s == 5:
                # M6 = (A10 - A00)(B00 + B01)
                U[2][r] = 1  # A10
                U[0][r] = -1 # A00
                V[2*t][r] = 1       # B00
                V[2*t + 1][r] = 1   # B01
                W[4*t + 3][r] = 1   # C11
            elif s == 6:
                # M7 = (A01 - A11)(B10 + B11)
                U[1][r] = 1  # A01
                U[3][r] = -1 # A11
                V[14 + 2*t][r] = 1       # B10
                V[14 + 2*t + 1][r] = 1   # B11
                W[4*t][r] = 1            # C00

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
