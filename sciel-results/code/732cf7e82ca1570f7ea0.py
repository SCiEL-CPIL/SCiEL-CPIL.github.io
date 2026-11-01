"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 7]

# EVOLVE-BLOCK-START
def construct():
    # Block-diagonal Strassen (rank 7 per 2-column block) + schoolbook (rank 4) for the final column.
    # Total rank = 3 * 7 + 4 = 25
    n, m, p = DIMENSIONS  # 2, 2, 7
    rank = 25
    # U: (n*m, rank) = (4, 25)
    # V: (m*p, rank) = (14, 25)
    # W: (p*n, rank) = (14, 25)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Strassen blocks for column pairs (0,1), (2,3), (4,5)
    for block in range(3):
        k1 = 2 * block      # first column in block
        k2 = 2 * block + 1  # second column in block
        
        # Strassen formulas for 2x2 * 2x2:
        # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
        # P1 = (a+d)(e+h)
        # P2 = d(g-h) ... wait, let me re-derive
        # Standard Strassen:
        # P1 = (a+d)(e+h)
        # P2 = (c+d)h
        # P3 = a(f-h)
        # P4 = d(g-e)
        # P5 = (a+b)e
        # P6 = (c-a)(h+e)
        # P7 = (b-d)(g+f)
        # C11 = P1 + P4 - P5 - P7
        # C21 = P2 + P4
        # C12 = P3 + P5
        # C22 = P1 - P2 + P3 + P6
        
        # U coefficients for a, b, c, d (indices 0,1,2,3 in U)
        # P1: a+d -> U = [1, 0, 0, 1]
        # P2: c+d -> U = [0, 0, 1, 1]
        # P3: a   -> U = [1, 0, 0, 0]  (a*(f-h), B part is f-h)
        # P4: d   -> U = [0, 0, 0, 1]  (d*(g-e), B part is g-e)
        # P5: a+b -> U = [1, 1, 0, 0]
        # P6: c-a -> U = [1, 0, 1, 0] ... wait, (c-a) means -a+c, so U = [-1, 0, 1, 0]
        # P7: b-d -> U = [0, 1, 0, -1]
        
        U_coeffs = [
            [1, 0, 0, 1],    # P1: a+d
            [0, 0, 1, 1],    # P2: c+d
            [1, 0, 0, 0],    # P3: a
            [0, 0, 0, 1],    # P4: d
            [1, 1, 0, 0],    # P5: a+b
            [-1, 0, 1, 0],   # P6: c-a
            [0, 1, 0, -1],   # P7: b-d
        ]
        
        # V coefficients: for each P, which entries of B (2 rows x 2 cols)
        # B = [[B[0,k1], B[0,k2]], [B[1,k1], B[1,k2]]]
        # V index: j*p + k where j is row (0 or 1), k is column (k1 or k2)
        # P1: e+h -> B[0,k1] + B[1,k2] -> V[j=0,k=k1]=1, V[j=1,k=k2]=1
        # P2: h   -> B[1,k2] -> V[j=1,k=k2]=1
        # P3: f-h -> B[0,k2] - B[1,k2] -> V[j=0,k=k2]=1, V[j=1,k=k2]=-1
        # P4: g-e -> B[1,k1] - B[0,k1] -> V[j=1,k=k1]=1, V[j=0,k=k1]=-1
        # P5: e   -> B[0,k1] -> V[j=0,k=k1]=1
        # P6: h+e -> B[1,k2] + B[0,k1] -> V[j=1,k=k2]=1, V[j=0,k=k1]=1
        # P7: g+f -> B[1,k1] + B[0,k2] -> V[j=1,k=k1]=1, V[j=0,k=k2]=1
        
        V_entries = [
            [(0, k1, 1), (1, k2, 1)],   # P1
            [(1, k2, 1)],                # P2
            [(0, k2, 1), (1, k2, -1)],   # P3
            [(1, k1, 1), (0, k1, -1)],   # P4
            [(0, k1, 1)],                # P5
            [(1, k2, 1), (0, k1, 1)],    # P6
            [(1, k1, 1), (0, k2, 1)],    # P7
        ]
        
        # W coefficients: for each output C[i,k], sum of P_j with coefficients
        # C[0,k1] (index k1*n+0 = 2*k1+0): P1 + P4 - P5 - P7
        # C[1,k1] (index 2*k1+1): P2 + P4
        # C[0,k2] (index 2*k2+0): P3 + P5
        # C[1,k2] (index 2*k2+1): P1 - P2 + P3 + P6
        
        W_coefficients = {
            2*k1+0: [(0, 1), (3, 1), (4, -1), (6, -1)],  # P1 + P4 - P5 - P7
            2*k1+1: [(1, 1), (3, 1)],                     # P2 + P4
            2*k2+0: [(2, 1), (4, 1)],                     # P3 + P5
            2*k2+1: [(0, 1), (1, -1), (2, 1), (5, 1)],   # P1 - P2 + P3 + P6
        }
        
        for p_idx in range(7):
            c = col + p_idx
            # Fill U
            for i in range(4):
                U[i][c] = U_coeffs[p_idx][i]
            # Fill V
            for (j, k, val) in V_entries[p_idx]:
                V[j * p + k][c] = val
            # Fill W
            for (out_idx, coeff) in [(2*k1+0, 0), (2*k1+1, 0), (2*k2+0, 0), (2*k2+1, 0)]:
                pass  # handled below
        
        # Fill W properly
        for out_idx, coeffs in W_coefficients.items():
            for (p_idx, val) in coeffs:
                W[out_idx][col + p_idx] += val
        
        col += 7
    
    # Schoolbook for last column (k=6)
    # C[0,6] = a*B[0,6] + b*B[1,6]
    # C[1,6] = c*B[0,6] + d*B[1,6]
    # 4 rank-1 terms
    sb_U = [
        [1, 0, 0, 0],  # a
        [0, 1, 0, 0],  # b
        [0, 0, 1, 0],  # c
        [0, 0, 0, 1],  # d
    ]
    # V: for each term, which B entry
    # Term 0: a * B[0,6] -> V[0*7+6] = V[6]
    # Term 1: b * B[1,6] -> V[1*7+6] = V[13]
    # Term 2: c * B[0,6] -> V[6]
    # Term 3: d * B[1,6] -> V[13]
    for t in range(4):
        c = col + t
        for i in range(4):
            U[i][c] = sb_U[t][i]
        if t == 0 or t == 2:
            V[0 * p + 6][c] = 1  # B[0,6]
        else:
            V[1 * p + 6][c] = 1  # B[1,6]
    
    # W for last column: C[0,6] = term0 + term1, C[1,6] = term2 + term3
    W[2*6+0][col + 0] += 1  # C[0,6] += a*B[0,6]
    W[2*6+0][col + 1] += 1  # C[0,6] += b*B[1,6]
    W[2*6+1][col + 2] += 1  # C[1,6] += c*B[0,6]
    W[2*6+1][col + 3] += 1  # C[1,6] += d*B[1,6]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
