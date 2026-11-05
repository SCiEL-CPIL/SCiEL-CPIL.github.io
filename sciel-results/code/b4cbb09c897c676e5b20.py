"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    # Rank 17 decomposition using Strassen for the 2x2x2 sub-block
    n, m, p = 2, 3, 3
    rank = 17
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Block 1: C1 = A[:, 0] * B[0, :]
    # i in 0..1, j=0, l in 0..2
    # Rank 6
    for i in range(n):
        for l in range(p):
            # A[i, 0] -> U[i*m + 0]
            # B[0, l] -> V[0*p + l]
            # C[i, l] -> W[l*n + i]
            U[i * m + 0][col] = 1
            V[0 * p + l][col] = 1
            W[l * n + i][col] = 1
            col += 1
            
    # Block 2: C22 = A[:, 1:3] * B[1:3, 2]
    # i in 0..1, j in 1..2, l=2
    # Rank 4
    for i in range(n):
        for j in range(1, 3):
            # A[i, j] -> U[i*m + j]
            # B[j, 2] -> V[j*p + 2]
            # C[i, 2] -> W[2*n + i]
            U[i * m + j][col] = 1
            V[j * p + 2][col] = 1
            W[2 * n + i][col] = 1
            col += 1

    # Block 3: C21 = A[:, 1:3] * B[1:3, 0:2]
    # This is a 2x2x2 multiplication.
    # A' is 2x2: A'[i', j'] = A[i', 1+j']
    # B' is 2x2: B'[j', l'] = B[1+j', l']
    # C' is 2x2: C'[i', l'] = C[i', l']
    # Use Strassen's algorithm.
    
    # Strassen terms:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A10) * B11
    # M6 = (A11 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Helper to add a Strassen term
    def add_strassen_term(u_coeffs, v_coeffs, w_coeffs, col_idx):
        # u_coeffs: list of (global_u_idx, coeff)
        # v_coeffs: list of (global_v_idx, coeff)
        # w_coeffs: list of (global_w_idx, coeff)
        for u_idx, coeff in u_coeffs:
            U[u_idx][col_idx] += coeff
        for v_idx, coeff in v_coeffs:
            V[v_idx][col_idx] += coeff
        for w_idx, coeff in w_coeffs:
            W[w_idx][col_idx] += coeff

    # Indices for A' (i', j') -> global U[i'*m + (1+j')]
    # i' in 0,1; j' in 0,1
    # A'00 -> U[0*3+1] = U[1]
    # A'01 -> U[0*3+2] = U[2]
    # A'10 -> U[1*3+1] = U[4]
    # A'11 -> U[1*3+2] = U[5]
    
    # Indices for B' (j', l') -> global V[(1+j')*p + l']
    # j' in 0,1; l' in 0,1
    # B'00 -> V[1*3+0] = V[3]
    # B'01 -> V[1*3+1] = V[4]
    # B'10 -> V[2*3+0] = V[6]
    # B'11 -> V[2*3+1] = V[7]
    
    # Indices for C' (i', l') -> global W[l'*n + i']
    # i' in 0,1; l' in 0,1
    # C'00 -> W[0*2+0] = W[0]
    # C'01 -> W[1*2+0] = W[2]
    # C'10 -> W[0*2+1] = W[1]
    # C'11 -> W[1*2+1] = W[3]

    # M1
    u1 = [(1, 1), (5, 1)]
    v1 = [(3, 1), (7, 1)]
    w1 = [(0, 1), (3, 1)] # C00 and C11
    add_strassen_term(u1, v1, w1, col)
    col += 1

    # M2
    u2 = [(4, 1), (5, 1)]
    v2 = [(3, 1)]
    w2 = [(1, 1), (3, -1)] # C10 and C11
    add_strassen_term(u2, v2, w2, col)
    col += 1

    # M3
    u3 = [(1, 1)]
    v3 = [(4, 1), (7, -1)]
    w3 = [(2, 1), (3, 1)] # C01 and C11
    add_strassen_term(u3, v3, w3, col)
    col += 1

    # M4
    u4 = [(5, 1)]
    v4 = [(6, 1), (3, -1)]
    w4 = [(0, 1), (1, 1)] # C00 and C10
    add_strassen_term(u4, v4, w4, col)
    col += 1

    # M5
    u5 = [(1, 1), (4, 1)]
    v5 = [(7, 1)]
    w5 = [(0, -1), (2, 1)] # C00 and C01
    add_strassen_term(u5, v5, w5, col)
    col += 1

    # M6
    u6 = [(5, 1), (1, -1)]
    v6 = [(3, 1), (4, 1)]
    w6 = [(3, 1)] # C11
    add_strassen_term(u6, v6, w6, col)
    col += 1

    # M7
    u7 = [(2, 1), (5, -1)]
    v7 = [(6, 1), (7, 1)]
    w7 = [(0, 1)] # C00
    add_strassen_term(u7, v7, w7, col)
    col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
