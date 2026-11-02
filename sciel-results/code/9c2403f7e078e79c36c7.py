"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank calculation: 4 blocks of Strassen (7 each) + 16 schoolbook terms
    rank = 4 * 7 + 16
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen coefficients for 2x2x2
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    
    # A indices: (0,0)=a, (0,1)=b, (1,0)=c, (1,1)=d
    # B indices: (0,0)=e, (0,1)=f, (1,0)=g, (1,1)=h
    # C indices: (0,0), (0,1), (1,0), (1,1)
    
    # U coeffs (A), V coeffs (B), W coeffs (C)
    # Term 1: m1
    U_coeffs_1 = {(0,0): 1, (1,1): 1}
    V_coeffs_1 = {(0,0): 1, (1,1): 1}
    W_coeffs_1 = {(0,0): 1, (0,1): 1, (1,0): 1, (1,1): -1} # C00=1, C01=1, C10=1, C11=-1? 
    # C00 = m1 - m4 + m6 - m7
    # C01 = m3 + m4 - m5 + m7
    # C10 = m2 + m4 + m5 - m6
    # C11 = m1 - m2 + m3 + m6
    
    # Let's define W for each m_r based on the output equations.
    # m1: C00:+1, C01:0, C10:0, C11:+1
    # m2: C00:0, C01:0, C10:+1, C11:-1
    # m3: C00:0, C01:+1, C10:0, C11:+1
    # m4: C00:-1, C01:+1, C10:+1, C11:0
    # m5: C00:0, C01:-1, C10:+1, C11:0
    # m6: C00:+1, C01:0, C10:-1, C11:+1
    # m7: C00:-1, C01:+1, C10:0, C11:0
    
    U_strassen = [
        {(0,0): 1, (1,1): 1}, # m1
        {(1,0): 1, (1,1): 1}, # m2
        {(0,0): 1},           # m3
        {(1,1): 1},           # m4
        {(0,0): 1, (0,1): 1}, # m5
        {(1,0): 1, (0,0): -1},# m6
        {(0,1): 1, (1,1): -1} # m7
    ]
    
    V_strassen = [
        {(0,0): 1, (1,1): 1}, # m1
        {(0,0): 1},           # m2
        {(0,1): 1, (1,1): -1},# m3
        {(1,0): 1, (0,0): -1},# m4
        {(1,1): 1},           # m5
        {(0,0): 1, (0,1): 1}, # m6
        {(1,0): 1, (1,1): 1}  # m7
    ]
    
    W_strassen = [
        {(0,0): 1, (1,1): 1},  # m1
        {(1,0): 1, (1,1): -1}, # m2
        {(0,1): 1, (1,1): 1},  # m3
        {(0,0): -1, (0,1): 1, (1,0): 1}, # m4
        {(0,1): -1, (1,0): 1}, # m5
        {(0,0): 1, (1,0): -1, (1,1): 1}, # m6
        {(0,0): -1, (0,1): 1}  # m7
    ]
    
    # Apply Strassen to 4 blocks
    for b in range(4):
        for r in range(7):
            # U
            for (i, j), val in U_strassen[r].items():
                U[i * m + j][col] = val
            # V
            for (j, k), val in V_strassen[r].items():
                V[j * p + (2 * b + k)][col] = val
            # W
            for (k, i), val in W_strassen[r].items():
                W[(2 * b + k) * n + i][col] = val
            col += 1
            
    # Apply Schoolbook to remaining terms (j=2)
    for i in range(n):
        for k in range(p):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
