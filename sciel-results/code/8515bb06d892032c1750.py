"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    # Block Strassen: 2 blocks of A (2x2) x 3 blocks of B (2x2 cols) x 7 Strassen terms = 42
    rank = 42
    U = [[0] * rank for _ in range(n * m)]  # shape (8, 42)
    V = [[0] * rank for _ in range(m * p)]  # shape (24, 42)
    W = [[0] * rank for _ in range(p * n)]  # shape (12, 42)
    
    # Strassen coefficients for U (first matrix M, 2x2 local)
    # Local index (i_loc, j_loc) -> global row i=i_loc, col j
    # P1: M00+M11, P2: M10+M11, P3: M00, P4: M11, P5: M00+M01, P6: M10-M00, P7: M01-M11
    U_coeffs = [
        {(0, 0): 1, (1, 1): 1},   # P1
        {(1, 0): 1, (1, 1): 1},   # P2
        {(0, 0): 1},               # P3
        {(1, 1): 1},               # P4
        {(0, 0): 1, (0, 1): 1},   # P5
        {(1, 0): 1, (0, 0): -1},  # P6
        {(0, 1): 1, (1, 1): -1},  # P7
    ]
    
    # Strassen coefficients for V (second matrix N, 2x2 local)
    # P1: N00+N11, P2: N00, P3: N01-N11, P4: N10-N00, P5: N11, P6: N00+N01, P7: N10+N11
    V_coeffs = [
        {(0, 0): 1, (1, 1): 1},   # P1
        {(0, 0): 1},               # P2
        {(0, 1): 1, (1, 1): -1},  # P3
        {(1, 0): 1, (0, 0): -1},  # P4
        {(1, 1): 1},               # P5
        {(0, 0): 1, (0, 1): 1},   # P6
        {(1, 0): 1, (1, 1): 1},   # P7
    ]
    
    # W contributions: output R (2x2 local), trace convention W[k*n+i]
    # R00 = P1 + P4 - P5 + P7
    # R01 = P3 + P5
    # R10 = P2 + P4
    # R11 = P1 - P2 + P3 + P6
    W_contrib = [
        {(0, 0): 1, (1, 1): 1},    # P1 -> R00(+1), R11(+1)
        {(1, 0): 1, (1, 1): -1},   # P2 -> R10(+1), R11(-1)
        {(0, 1): 1, (1, 1): 1},    # P3 -> R01(+1), R11(+1)
        {(0, 0): 1, (1, 0): 1},    # P4 -> R00(+1), R10(+1)
        {(0, 0): -1, (0, 1): 1},   # P5 -> R00(-1), R01(+1)
        {(1, 1): 1},               # P6 -> R11(+1)
        {(0, 0): 1},               # P7 -> R00(+1)
    ]
    
    col = 0
    for a in range(2):  # A block: j in {0,1} or {2,3}
        for b in range(3):  # B col block: k in {0,1}, {2,3}, {4,5}
            for s in range(7):  # Strassen term
                # Update U: index (i, j) where i=i_loc, j=a*2+j_loc
                for (i_loc, j_loc), coeff in U_coeffs[s].items():
                    i = i_loc
                    j = a * 2 + j_loc
                    u_idx = i * m + j
                    U[u_idx][col] += coeff
                
                # Update V: index (j, k) where j=a*2+j_loc, k=b*2+k_loc
                for (j_loc, k_loc), coeff in V_coeffs[s].items():
                    j = a * 2 + j_loc
                    k = b * 2 + k_loc
                    v_idx = j * p + k
                    V[v_idx][col] += coeff
                
                # Update W: trace convention W[k*n+i], index (i_loc, k_loc) -> w_idx = k*n+i
                for (i_loc, k_loc), coeff in W_contrib[s].items():
                    i = i_loc
                    k = b * 2 + k_loc
                    w_idx = k * n + i
                    W[w_idx][col] += coeff
                
                col += 1
    
    if time.time() - start_time > 140:
        raise TimeoutError()
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
