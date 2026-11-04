"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for U (A matrix)
    # Indices in local 2x2 block:
    # 0: x00, 1: x01, 2: x10, 3: x11
    # But we map to global U indices.
    
    # Strassen coefficients for V (B matrix)
    # Local 2x2 block indices:
    # 0: y00, 1: y01, 2: y10, 3: y11
    
    # Strassen W coefficients
    # w[i, c', s] where i in {0,1}, c' in {0,1}, s in {0..6}
    W_coeffs = [
        # s=0 (P1)
        [(0, 0, 1), (1, 1, 1)],
        # s=1 (P2)
        [(1, 0, 1), (1, 1, -1)],
        # s=2 (P3)
        [(0, 1, 1), (1, 1, 1)],
        # s=3 (P4)
        [(0, 0, 1), (1, 0, 1)],
        # s=4 (P5)
        [(0, 0, -1), (0, 1, 1)],
        # s=5 (P6)
        [(1, 1, 1)],
        # s=6 (P7)
        [(0, 0, 1)]
    ]
    
    # U coefficients for each s
    # List of (local_i, local_j, coeff)
    U_coeffs = [
        [(0, 0, 1)],                         # P1: x00
        [(0, 0, 1), (1, 0, 1)],              # P2: x00 + x10
        [(1, 1, 1), (0, 1, -1)],             # P3: x11 - x01
        [(1, 1, 1)],                         # P4: x11
        [(0, 0, 1), (0, 1, 1)],              # P5: x00 + x01
        [(1, 0, 1), (0, 0, -1)],             # P6: x10 - x00
        [(0, 1, 1), (1, 1, -1)]              # P7: x01 - x11
    ]
    
    # V coefficients for each s
    # List of (local_i, local_j, coeff)
    V_coeffs = [
        [(0, 0, 1), (1, 1, 1)],              # P1: y00 + y11
        [(0, 0, 1)],                         # P2: y00
        [(0, 1, 1), (1, 1, 1)],              # P3: y01 + y11
        [(0, 0, 1), (0, 1, -1)],             # P4: y00 - y01
        [(1, 1, 1)],                         # P5: y11
        [(0, 0, 1), (0, 1, 1)],              # P6: y00 + y01
        [(1, 0, 1), (1, 1, 1)]               # P7: y10 + y11
    ]
    
    # Loop over A-blocks (b=0 for A[0:2, :], b=1 for A[2:4, :])
    # Actually A is 2x4. Split into two 2x2 blocks along columns.
    # b=0: cols 0,1. b=1: cols 2,3.
    for b in range(2):
        # U row offset for this block
        # U index = i * m + j
        # For b=0, j in {0,1}. For b=1, j in {2,3}.
        # local_j 0 -> global j = b*2. local_j 1 -> global j = b*2+1.
        
        # V row offset for this block
        # V index = j * p + k
        # For b=0, j in {0,1}. For b=1, j in {2,3}.
        # local_i 0 -> global j = b*2. local_i 1 -> global j = b*2+1.
        
        for k in range(4): # B-blocks (columns of C)
            # Rank base for this block
            base_r = b * 28 + k * 7
            
            for s in range(7):
                r = base_r + s
                
                # Fill U
                for li, lj, coeff in U_coeffs[s]:
                    # local i, j
                    gi = li
                    gj = b * 2 + lj
                    u_idx = gi * m + gj
                    U[u_idx][r] += coeff
                
                # Fill V
                for li, lj, coeff in V_coeffs[s]:
                    # local i (row of B), j (col of B)
                    gj = b * 2 + li
                    gk = k * 2 + lj
                    v_idx = gj * p + gk
                    V[v_idx][r] += coeff
                
                # Fill W
                for gi, gc, coeff in W_coeffs[s]:
                    # gi is row of C (0 or 1)
                    # gc is local col of C block (0 or 1)
                    gk = k * 2 + gc
                    w_idx = gk * n + gi
                    W[w_idx][r] += coeff
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
