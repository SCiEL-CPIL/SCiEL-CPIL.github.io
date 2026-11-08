import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 49
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)

    # Outer Strassen coefficients for X (A blocks)
    # x[k][a][b] where k is 1..7, a,b are 1..2 (0-indexed 0..1)
    x = [
        {}, # 0
        {(0,0): 1, (1,1): 1},      # 1: A11 + A22
        {(1,0): 1, (1,1): 1},      # 2: A21 + A22
        {(0,0): 1},                # 3: A11
        {(1,1): 1},                # 4: A22
        {(0,0): 1, (0,1): 1},      # 5: A11 + A12
        {(1,0): 1, (0,0): -1},     # 6: A21 - A11
        {(0,1): 1, (1,1): -1}      # 7: A12 - A22
    ]

    # Outer Strassen coefficients for Y (B blocks)
    y = [
        {}, # 0
        {(0,0): 1, (1,1): 1},      # 1: B11 + B22
        {(0,0): 1},                # 2: B11
        {(0,1): 1, (1,1): -1},     # 3: B12 - B22
        {(1,0): 1, (0,0): -1},     # 4: B21 - B11
        {(1,1): 1},                # 5: B22
        {(0,0): 1, (0,1): 1},      # 6: B11 + B12
        {(1,0): 1, (1,1): 1}       # 7: B21 + B22
    ]

    # Outer Strassen coefficients for C blocks
    # nu[ab][k] where ab is block index (0,0)..(1,1), k is 1..7
    nu = [
        [0, 0, 0, 0, 0, 0, 0, 0], # 0
        [0, 1, 0, 1, -1, 0, 1, 0], # 1: C11 = M1 + M4 - M5 + M7
        [0, 0, 0, 1, 1, 0, 0, 0], # 2: C12 = M3 + M5
        [0, 0, 1, 1, 0, 0, 0, 0], # 3: C21 = M2 + M4
        [0, 1, -1, 1, 0, 1, 0, 0]  # 4: C22 = M1 - M2 + M3 + M6
    ]

    # Inner Strassen coefficients for X (entries of X)
    # alpha[l][p][q] where l is 1..7, p,q are 1..2 (0-indexed 0..1)
    alpha = [
        {}, # 0
        {(0,0): 1, (1,1): 1},      # 1: X11 + X22
        {(1,0): 1, (1,1): 1},      # 2: X21 + X22
        {(0,0): 1},                # 3: X11
        {(1,1): 1},                # 4: X22
        {(0,0): 1, (0,1): 1},      # 5: X11 + X12
        {(1,0): 1, (0,0): -1},     # 6: X21 - X11
        {(0,1): 1, (1,1): -1}      # 7: X12 - X22
    ]

    # Inner Strassen coefficients for Y (entries of Y)
    beta = [
        {}, # 0
        {(0,0): 1, (1,1): 1},      # 1: Y11 + Y22
        {(0,0): 1},                # 2: Y11
        {(0,1): 1, (1,1): -1},     # 3: Y12 - Y22
        {(1,0): 1, (0,0): -1},     # 4: Y21 - Y11
        {(1,1): 1},                # 5: Y22
        {(0,0): 1, (0,1): 1},      # 6: Y11 + Y12
        {(1,0): 1, (1,1): 1}       # 7: Y21 + Y22
    ]

    # Inner Strassen coefficients for M entries
    # mu[pq][l] where pq is entry index (0,0)..(1,1), l is 1..7
    mu = [
        [0, 0, 0, 0, 0, 0, 0, 0], # 0
        [0, 1, 0, 1, -1, 0, 1, 0], # 1: M11 = P1 + P4 - P5 + P7
        [0, 0, 1, 0, 1, 0, 0, 0], # 2: M12 = P3 + P5
        [0, 0, 1, 0, 1, 0, 0, 0], # 3: M21 = P2 + P4  (Wait, M21 = P2 + P4? No. M21 = P2 + P4? Let's check)
        # Strassen M21 = P2 + P4?
        # P2 = (A21+A22)B11. P4 = A22(B21-B11).
        # M21 = A21B11 + A22B21.
        # P2 + P4 = A21B11 + A22B11 + A22B21 - A22B11 = A21B11 + A22B21. Correct.
        # So mu[3] corresponds to M21.
        # But my list above for mu was:
        # mu[1] for M11
        # mu[2] for M12
        # mu[3] for M21
        # mu[4] for M22
        # Let's re-verify mu[3] coefficients.
        # M21 = P2 + P4.
        # So mu[3] = [0, 1, 0, 1, 0, 0, 0, 0].
        # My previous definition: [0, 0, 1, 0, 1, 0, 0, 0] was for M12 (P3+P5).
        # So I need to be careful with indices.
        [0, 1, 0, 1, 0, 0, 0, 0], # 3: M21 = P2 + P4
        [0, 1, -1, 1, 0, 1, 0, 0]  # 4: M22 = P1 - P2 + P3 + P6
    ]

    # Iterate over outer steps k (1..7) and inner steps l (1..7)
    for k in range(1, 8):
        for l in range(1, 8):
            r = (k - 1) * 7 + (l - 1)
            
            # Calculate U contributions
            # U[I, r] where I = i*4 + j
            # U[I, r] = sum_{p,q,a,b} alpha[l][p,q] * x[k][a,b] for I=(2a+p)*4 + (2b+q)
            for (p, q), val_a in alpha[l].items():
                for (a, b), val_x in x[k].items():
                    i = 2 * a + p
                    j = 2 * b + q
                    I = i * 4 + j
                    U[I, r] += val_a * val_x
            
            # Calculate V contributions
            # V[J, r] where J = i*4 + j
            # V[J, r] = sum_{r',s',c,d} beta[l][r',s'] * y[k][c,d] for J=(2c+r')*4 + (2d+s')
            for (r_prime, s_prime), val_b in beta[l].items():
                for (c, d), val_y in y[k].items():
                    i = 2 * c + r_prime
                    j = 2 * d + s_prime
                    J = i * 4 + j
                    V[J, r] += val_b * val_y
            
            # Calculate W contributions
            # W[K, r] where K = j*4 + i
            # W[K, r] = sum_{a,b,p,q} nu[ab][k] * mu[pq][l] for K=(2b+q)*4 + (2a+p)
            # Note: nu is indexed by block (a,b) and step k.
            # mu is indexed by entry (p,q) and step l.
            # We sum over all blocks (a,b) and entries (p,q).
            for a in range(2):
                for b in range(2):
                    block_idx = a * 2 + b + 1 # 1..4
                    coeff_nu = nu[block_idx][k]
                    if coeff_nu == 0:
                        continue
                    for p in range(2):
                        for q in range(2):
                            entry_idx = p * 2 + q + 1 # 1..4
                            coeff_mu = mu[entry_idx][l]
                            if coeff_mu == 0:
                                continue
                            i = 2 * a + p
                            j = 2 * b + q
                            K = j * 4 + i
                            W[K, r] += coeff_nu * coeff_mu

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
