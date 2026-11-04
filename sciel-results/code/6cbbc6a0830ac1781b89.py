"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 42 decomposition using Strassen's algorithm for 2x2x2 blocks
    # Split A (2x4) into two 2x2 blocks: A1, A2
    # Split B (4x6) into two 2x6 blocks: B1, B2
    # C = A1 B1 + A2 B2
    # Each 2x2 * 2x6 is split into three 2x2 * 2x2 multiplications
    # Total 6 multiplications of rank 7 each -> Rank 42
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 matrices
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = g(b-a)
    # M5 = (a+b)(h-g)
    # M6 = (c-d)(g+h)
    # M7 = (a+c)(e-f)
    
    # Coefficients for A (indices 0,1 for rows; 0,1 for cols)
    # M1: a=1, d=1 -> (0,0)=1, (1,1)=1
    # M2: c=1, d=1 -> (1,0)=1, (1,1)=1
    # M3: a=1 -> (0,0)=1
    # M4: b=1, a=-1 -> (0,1)=1, (0,0)=-1
    # M5: a=1, b=1 -> (0,0)=1, (0,1)=1
    # M6: c=1, d=-1 -> (1,0)=1, (1,1)=-1
    # M7: a=1, c=1 -> (0,0)=1, (1,0)=1
    
    A_coeffs = [
        {(0,0): 1, (1,1): 1},       # M1
        {(1,0): 1, (1,1): 1},       # M2
        {(0,0): 1},                 # M3
        {(0,1): 1, (0,0): -1},      # M4
        {(0,0): 1, (0,1): 1},       # M5
        {(1,0): 1, (1,1): -1},      # M6
        {(0,0): 1, (1,0): 1}        # M7
    ]
    
    # Coefficients for B
    # M1: e=1, h=1 -> (0,0)=1, (1,1)=1
    # M2: e=1 -> (0,0)=1
    # M3: f=1, h=-1 -> (0,1)=1, (1,1)=-1
    # M4: g=1 -> (1,0)=1
    # M5: h=1, g=-1 -> (1,1)=1, (1,0)=-1
    # M6: g=1, h=1 -> (1,0)=1, (1,1)=1
    # M7: e=1, f=-1 -> (0,0)=1, (0,1)=-1
    
    B_coeffs = [
        {(0,0): 1, (1,1): 1},       # M1
        {(0,0): 1},                 # M2
        {(0,1): 1, (1,1): -1},      # M3
        {(1,0): 1},                 # M4
        {(1,1): 1, (1,0): -1},      # M5
        {(1,0): 1, (1,1): 1},       # M6
        {(0,0): 1, (0,1): -1}       # M7
    ]
    
    # Coefficients for C (output 2x2)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We need C_coeffs[r][(i,k)] for r=0..6, (i,k) in {(0,0), (0,1), (1,0), (1,1)}
    C_coeffs = [
        {(0,0): 1, (1,1): 1},       # M1 contributes to C00 (+1) and C11 (+1)
        {(1,0): 1, (1,1): -1},      # M2 contributes to C10 (+1) and C11 (-1)
        {(0,1): 1, (1,1): 1},       # M3 contributes to C01 (+1) and C11 (+1)
        {(0,0): 1, (1,0): 1},       # M4 contributes to C00 (+1) and C10 (+1)
        {(0,0): -1, (0,1): 1},      # M5 contributes to C00 (-1) and C01 (+1)
        {(1,1): 1},                 # M6 contributes to C11 (+1)
        {(0,0): 1}                  # M7 contributes to C00 (+1)
    ]
    
    # t: 0 for A1/B1, 1 for A2/B2
    # s: 0,1,2 for the three 2x2 blocks of B_t
    # r: 0..6 for Strassen terms
    
    for t in range(2):
        for s in range(3):
            for r in range(7):
                col = t * 21 + s * 7 + r
                
                # Fill U
                for (i, j), val in A_coeffs[r].items():
                    # Global A index: row i, col t*2 + j
                    u_idx = i * m + (t * 2 + j)
                    U[u_idx][col] = val
                
                # Fill V
                for (j, k), val in B_coeffs[r].items():
                    # Global B index: row t*2 + j, col s*2 + k
                    v_idx = (t * 2 + j) * p + (s * 2 + k)
                    V[v_idx][col] = val
                
                # Fill W
                for (i, k), val in C_coeffs[r].items():
                    # Global C index: row i, col s*2 + k
                    # W index: k_global * n + i
                    w_idx = (s * 2 + k) * n + i
                    W[w_idx][col] = val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
