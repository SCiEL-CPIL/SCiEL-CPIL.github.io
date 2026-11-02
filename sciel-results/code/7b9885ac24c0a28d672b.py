import numpy as np

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We decompose the 2x4x6 multiplication into 6 independent 2x2x2 multiplications
    # by splitting the middle dimension m=4 into two parts of size 2,
    # and the output dimension p=6 into three parts of size 2.
    # Each 2x2x2 multiplication uses Strassen's algorithm (rank 7).
    # Total rank = 2 * 3 * 7 = 42.
    
    rank = 42
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen coefficients for 2x2x2
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)(e)
    # M3 = (a-b)(g+h)
    # M4 = (a)(h-d)
    # M5 = (c)(e-f)
    # M6 = (b)(g-h)  # Wait, standard Strassen is:
    # P1 = (a+d)(e+h)
    # P2 = (c+d)e
    # P3 = a(f-h)
    # P4 = d(g-e)
    # P5 = (a+b)h
    # P6 = (c+d)(g-h) # No, let's use the standard one from the current code which worked.
    
    # Current code U_coeffs (for A):
    # P1: a+d -> {(0,0):1, (1,1):1}
    # P2: c+d -> {(1,0):1, (1,1):1}
    # P3: a   -> {(0,0):1}
    # P4: d   -> {(1,1):1}
    # P5: a+b -> {(0,0):1, (0,1):1}
    # P6: c-a -> {(1,0):1, (0,0):-1}
    # P7: b-d -> {(0,1):1, (1,1):-1}
    
    # Current code V_coeffs (for B):
    # P1: e+h -> {(0,0):1, (1,1):1}
    # P2: e   -> {(0,0):1}
    # P3: f-h -> {(0,1):1, (1,1):-1}
    # P4: g-e -> {(1,0):1, (0,0):-1}
    # P5: h   -> {(1,1):1}
    # P6: e+f -> {(0,0):1, (0,1):1}
    # P7: g+h -> {(1,0):1, (1,1):1}
    
    # Current code W_contrib (for C):
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    U_coeffs = [
        {(0, 0): 1, (1, 1): 1},
        {(1, 0): 1, (1, 1): 1},
        {(0, 0): 1},
        {(1, 1): 1},
        {(0, 0): 1, (0, 1): 1},
        {(1, 0): 1, (0, 0): -1},
        {(0, 1): 1, (1, 1): -1}
    ]
    
    V_coeffs = [
        {(0, 0): 1, (1, 1): 1},
        {(0, 0): 1},
        {(0, 1): 1, (1, 1): -1},
        {(1, 0): 1, (0, 0): -1},
        {(1, 1): 1},
        {(0, 0): 1, (0, 1): 1},
        {(1, 0): 1, (1, 1): 1}
    ]
    
    W_coeffs = [
        {(0, 0): 1, (1, 1): 1},  # P1
        {(1, 0): 1, (1, 1): -1}, # P2
        {(0, 1): 1, (1, 1): 1},  # P3
        {(0, 0): 1, (1, 0): 1},  # P4
        {(0, 0): -1, (0, 1): 1}, # P5
        {(1, 1): 1},             # P6
        {(0, 0): 1}              # P7
    ]

    col = 0
    for a_block in range(2): # Split m=4 into 2 blocks of 2
        for b_block in range(3): # Split p=6 into 3 blocks of 2
            for s in range(7):
                # Fill U
                for (i_loc, j_loc), coeff in U_coeffs[s].items():
                    i = i_loc
                    j = a_block * 2 + j_loc
                    U[i * m + j, col] += coeff
                
                # Fill V
                for (j_loc, k_loc), coeff in V_coeffs[s].items():
                    j = a_block * 2 + j_loc
                    k = b_block * 2 + k_loc
                    V[j * p + k, col] += coeff
                
                # Fill W
                for (i_loc, k_loc), coeff in W_coeffs[s].items():
                    i = i_loc
                    k = b_block * 2 + k_loc
                    W[k * n + i, col] += coeff
                
                col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
