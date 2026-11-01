"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 28  # 4 blocks * 7 Strassen terms
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for the 7 terms
    # Term 0: (a+d)(e+h) -> U: a,d; V: e,h; W: c11, c22
    # Term 1: (c+d)e     -> U: c,d; V: e;    W: c21, c22(-)
    # Term 2: a(f-h)     -> U: a;    V: f,h(-); W: c12, c22
    # Term 3: d(g-e)     -> U: d;    V: g,e(-); W: c11, c21
    # Term 4: (a+b)h     -> U: a,b;  V: h;    W: c11(-), c12
    # Term 5: (c-a)(e+f) -> U: c,a(-); V: e,f; W: c22
    # Term 6: (b-d)(g+h) -> U: b,d(-); V: g,h; W: c11
    
    # Indices in U (n*m = 4): a=0, b=1, c=2, d=3
    # Indices in V (m*p = 16): For block k, e=2k, f=2k+1, g=8+2k, h=9+2k
    # Indices in W (p*n = 16): For block k, c11=4k, c12=4k+1, c21=4k+2, c22=4k+3
    
    u_defs = [
        [1, 0, 0, 1], # T0: a, d
        [0, 0, 1, 1], # T1: c, d
        [1, 0, 0, 0], # T2: a
        [0, 0, 0, 1], # T3: d
        [1, 1, 0, 0], # T4: a, b
        [-1, 0, 1, 0],# T5: a(-), c
        [0, 1, 0, -1] # T6: b, d(-)
    ]
    
    # V and W definitions depend on block k
    # We will iterate through blocks k=0..3 and terms t=0..6
    
    for k in range(4):
        base_v = 2 * k
        e_idx = base_v
        f_idx = base_v + 1
        g_idx = 8 + base_v
        h_idx = 9 + base_v
        
        base_w = 4 * k
        c11_idx = base_w
        c21_idx = base_w + 1
        c12_idx = base_w + 2
        c22_idx = base_w + 3
        
        for t in range(7):
            col = k * 7 + t
            
            # Fill U
            for u_idx, val in enumerate(u_defs[t]):
                if val != 0:
                    U[u_idx][col] = val
            
            # Fill V
            if t == 0:
                V[e_idx][col] = 1
                V[h_idx][col] = 1
            elif t == 1:
                V[e_idx][col] = 1
            elif t == 2:
                V[f_idx][col] = 1
                V[h_idx][col] = -1
            elif t == 3:
                V[g_idx][col] = 1
                V[e_idx][col] = -1
            elif t == 4:
                V[h_idx][col] = 1
            elif t == 5:
                V[e_idx][col] = 1
                V[f_idx][col] = 1
            elif t == 6:
                V[g_idx][col] = 1
                V[h_idx][col] = 1
            
            # Fill W
            if t == 0:
                W[c11_idx][col] = 1
                W[c22_idx][col] = 1
            elif t == 1:
                W[c21_idx][col] = 1
                W[c22_idx][col] = -1
            elif t == 2:
                W[c12_idx][col] = 1
                W[c22_idx][col] = 1
            elif t == 3:
                W[c11_idx][col] = 1
                W[c21_idx][col] = 1
            elif t == 4:
                W[c11_idx][col] = -1
                W[c12_idx][col] = 1
            elif t == 5:
                W[c22_idx][col] = 1
            elif t == 6:
                W[c11_idx][col] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
