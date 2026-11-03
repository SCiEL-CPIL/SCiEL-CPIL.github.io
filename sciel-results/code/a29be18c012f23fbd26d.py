import time

DIMENSIONS = [2, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=8
    # U: (6, R), V: (24, R), W: (16, R)
    
    # Rank calculation:
    # Part 1: 2x1x8 -> 16 ranks
    # Part 2: 2x2x8 -> 4 * 7 = 28 ranks
    # Total: 44
    
    R = 44
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    # --- Part 1: A2 * B2 ---
    # A2 is col 2 of A. B2 is row 2 of B.
    # A[i, 2] * B[2, k] -> C[i, k]
    # U index: i*3 + 2
    # V index: 2*8 + k = 16 + k
    # W index: k*2 + i
    
    r_idx = 0
    for i in range(n):
        for k in range(p):
            u_idx = i * m + 2
            v_idx = 2 * p + k
            w_idx = k * n + i
            
            U[u_idx][r_idx] = 1
            V[v_idx][r_idx] = 1
            W[w_idx][r_idx] = 1
            r_idx += 1
            
    # --- Part 2: A1 * B1 ---
    # A1 is cols 0,1 of A. B1 is rows 0,1 of B.
    # Split B1 into 4 blocks of 2x2.
    # Block q uses cols 2q, 2q+1 of B1.
    
    # Strassen coefficients for 2x2
    # a=A00, b=A01, c=A10, d=A11
    # e=B00, f=B01, g=B10, h=B11 (local to block)
    
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # A indices in U:
    # a: 0*3+0 = 0
    # b: 0*3+1 = 1
    # c: 1*3+0 = 3
    # d: 1*3+1 = 4
    
    for q in range(4):
        # V indices for block q
        # e: B[0, 2q] -> 0*8 + 2q = 2q
        # f: B[0, 2q+1] -> 0*8 + 2q+1 = 2q+1
        # g: B[1, 2q] -> 1*8 + 2q = 8+2q
        # h: B[1, 2q+1] -> 1*8 + 2q+1 = 9+2q
        
        # W indices for block q
        # C[0, 2q] -> 2q*2 + 0 = 4q
        # C[0, 2q+1] -> (2q+1)*2 + 0 = 4q+2
        # C[1, 2q] -> 2q*2 + 1 = 4q+1
        # C[1, 2q+1] -> (2q+1)*2 + 1 = 4q+3
        
        # Define the 7 steps
        # Each step: (U_coeffs, V_coeffs, W_coeffs)
        # U_coeffs: list of (u_idx, coeff)
        # V_coeffs: list of (v_idx, coeff)
        # W_coeffs: list of (w_idx, coeff)
        
        steps = [
            # M1
            ([(0, 1), (4, 1)], [(2*q, 1), (9+2*q, 1)], [(4*q, 1), (4*q+3, 1)]),
            # M2
            ([(3, 1), (4, 1)], [(2*q, 1)], [(4*q+1, 1)]),
            # M3
            ([(0, 1)], [(2*q+1, 1), (9+2*q, -1)], [(4*q+2, 1)]),
            # M4
            ([(4, 1)], [(8+2*q, 1), (2*q, -1)], [(4*q, 1), (4*q+1, 1)]),
            # M5
            ([(0, 1), (1, 1)], [(9+2*q, 1)], [(4*q, -1), (4*q+2, 1)]),
            # M6
            ([(3, 1), (0, -1)], [(2*q, 1), (2*q+1, 1)], [(4*q+3, 1)]),
            # M7
            ([(1, 1), (4, -1)], [(8+2*q, 1), (9+2*q, 1)], [(4*q, 1)]),
        ]
        
        for u_coeffs, v_coeffs, w_coeffs in steps:
            for u_idx, coeff in u_coeffs:
                U[u_idx][r_idx] += coeff
            for v_idx, coeff in v_coeffs:
                V[v_idx][r_idx] += coeff
            for w_idx, coeff in w_coeffs:
                W[w_idx][r_idx] += coeff
            r_idx += 1
            
    return {"U": U, "V": V, "W": W}
