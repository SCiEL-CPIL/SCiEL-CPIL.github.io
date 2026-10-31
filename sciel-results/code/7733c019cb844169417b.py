"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Outer Strassen definitions
    # X_r and Y_r are 2x2 matrices of linear forms on A and B blocks.
    # We represent each linear form as a list of 16 coefficients (for A or B).
    # A indices: 0-15 (row 0-3, col 0-3)
    # B indices: 0-15 (row 0-3, col 0-3)
    
    def block_coeff(block_row, block_col, sign=1):
        """Returns a list of 16 coefficients for a single block."""
        coeffs = [0] * 16
        for u in range(2):
            for v in range(2):
                idx = (block_row * 2 + u) * 4 + (block_row * 2 + v) # Wait, A is 4x4.
                # A index is i*4 + k.
                # Block (br, bc) contains A[2*br+u, 2*bc+v].
                idx = (2 * block_row + u) * 4 + (2 * block_col + v)
                coeffs[idx] = sign
        return coeffs

    # Define X_r and Y_r for r in 0..6
    # Each is a 2x2 list of lists of 16 coeffs.
    outer_forms = []
    for r in range(7):
        X = [[None for _ in range(2)] for _ in range(2)]
        Y = [[None for _ in range(2)] for _ in range(2)]
        
        if r == 0:
            # X = A00 + A11
            for u in range(2):
                for v in range(2):
                    X[u][v] = [a + b for a, b in zip(block_coeff(0, 0), block_coeff(1, 1))]
            # Y = B00 + B11
            for u in range(2):
                for v in range(2):
                    Y[u][v] = [a + b for a, b in zip(block_coeff(0, 0), block_coeff(1, 1))]
        elif r == 1:
            # X = A10 + A11
            for u in range(2):
                for v in range(2):
                    X[u][v] = [a + b for a, b in zip(block_coeff(1, 0), block_coeff(1, 1))]
            # Y = B00
            for u in range(2):
                for v in range(2):
                    Y[u][v] = block_coeff(0, 0)
        elif r == 2:
            # X = A00
            for u in range(2):
                for v in range(2):
                    X[u][v] = block_coeff(0, 0)
            # Y = B01 - B11
            for u in range(2):
                for v in range(2):
                    Y[u][v] = [a - b for a, b in zip(block_coeff(0, 1), block_coeff(1, 1))]
        elif r == 3:
            # X = A11
            for u in range(2):
                for v in range(2):
                    X[u][v] = block_coeff(1, 1)
            # Y = B10 - B00
            for u in range(2):
                for v in range(2):
                    Y[u][v] = [a - b for a, b in zip(block_coeff(1, 0), block_coeff(0, 0))]
        elif r == 4:
            # X = A00 + A01
            for u in range(2):
                for v in range(2):
                    X[u][v] = [a + b for a, b in zip(block_coeff(0, 0), block_coeff(0, 1))]
            # Y = B11
            for u in range(2):
                for v in range(2):
                    Y[u][v] = block_coeff(1, 1)
        elif r == 5:
            # X = A10 - A00
            for u in range(2):
                for v in range(2):
                    X[u][v] = [a - b for a, b in zip(block_coeff(1, 0), block_coeff(0, 0))]
            # Y = B00 + B01
            for u in range(2):
                for v in range(2):
                    Y[u][v] = [a + b for a, b in zip(block_coeff(0, 0), block_coeff(0, 1))]
        elif r == 6:
            # X = A01 - A11
            for u in range(2):
                for v in range(2):
                    X[u][v] = [a - b for a, b in zip(block_coeff(0, 1), block_coeff(1, 1))]
            # Y = B10 + B11
            for u in range(2):
                for v in range(2):
                    Y[u][v] = [a + b for a, b in zip(block_coeff(1, 0), block_coeff(1, 1))]
        
        outer_forms.append((X, Y))

    # Inner Strassen linear forms for X and Y
    # s=0: x00+x11, y00+y11
    # s=1: x10+x11, y00
    # s=2: x00, y01-y11
    # s=3: x11, y10-y00
    # s=4: x00+x01, y11
    # s=5: x10-x00, y00+y01
    # s=6: x01-x11, y10+y11
    
    def combine(l1, l2, op='+'):
        if op == '+':
            return [a + b for a, b in zip(l1, l2)]
        else:
            return [a - b for a, b in zip(l1, l2)]

    inner_forms = []
    for s in range(7):
        if s == 0:
            lx = lambda X: combine(X[0][0], X[1][1], '+')
            ly = lambda Y: combine(Y[0][0], Y[1][1], '+')
        elif s == 1:
            lx = lambda X: combine(X[1][0], X[1][1], '+')
            ly = lambda Y: Y[0][0]
        elif s == 2:
            lx = lambda X: X[0][0]
            ly = lambda Y: combine(Y[0][1], Y[1][1], '-')
        elif s == 3:
            lx = lambda X: X[1][1]
            ly = lambda Y: combine(Y[1][0], Y[0][0], '-')
        elif s == 4:
            lx = lambda X: combine(X[0][0], X[0][1], '+')
            ly = lambda Y: Y[1][1]
        elif s == 5:
            lx = lambda X: combine(X[1][0], X[0][0], '-')
            ly = lambda Y: combine(Y[0][0], Y[0][1], '+')
        elif s == 6:
            lx = lambda X: combine(X[0][1], X[1][1], '-')
            ly = lambda Y: combine(Y[1][0], Y[1][1], '+')
        inner_forms.append((lx, ly))

    # Inner Result Coefficients K_s[u][v]
    K_coeffs = [
        [[1, 0], [0, 1]], # s=0
        [[0, 0], [1, 1]], # s=1
        [[0, 1], [0, 1]], # s=2
        [[1, 0], [0, 0]], # s=3
        [[-1, 1], [0, 0]], # s=4
        [[0, 0], [1, 1]], # s=5
        [[1, 0], [0, 0]]  # s=6
    ]

    # Outer Result Coefficients W_r[i1][k1]
    W_coeffs = [
        [[1, 0], [0, 1]], # r=0
        [[0, 0], [1, -1]], # r=1
        [[0, 1], [0, 1]], # r=2
        [[1, 0], [0, 0]], # r=3
        [[-1, 1], [0, 0]], # r=4
        [[0, 0], [1, 1]], # r=5
        [[1, 0], [0, 0]]  # r=6
    ]

    for r in range(7):
        X, Y = outer_forms[r]
        for s in range(7):
            col = r * 7 + s
            lx, ly = inner_forms[s]
            L_A = lx(X)
            L_B = ly(Y)
            
            # Update U and V
            for idx in range(16):
                if L_A[idx] != 0:
                    U[idx][col] += L_A[idx]
                if L_B[idx] != 0:
                    V[idx][col] += L_B[idx]
            
            # Update W
            for i1 in range(2):
                for k1 in range(2):
                    w_outer = W_coeffs[r][i1][k1]
                    if w_outer == 0:
                        continue
                    for u in range(2):
                        for v in range(2):
                            k_inner = K_coeffs[s][u][v]
                            if k_inner == 0:
                                continue
                            coeff = w_outer * k_inner
                            if coeff == 0:
                                continue
                            i = i1 * 2 + u
                            k = k1 * 2 + v
                            w_idx = k * 4 + i
                            W[w_idx][col] += coeff

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
