import numpy as np

def construct():
    n, m, p = 2, 5, 6
    # We expect rank 54
    # 6 blocks * 7 (Strassen) + 3 blocks * 4 (Outer) = 42 + 12 = 54
    
    # Pre-allocate with a slightly larger size just in case, or calculate exactly
    # But we know it's 54.
    # However, to be safe and simple, we can build lists and then convert.
    # Or just initialize with 54 columns.
    
    rank = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_idx = 0
    
    # Strassen Coefficients
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # P1 = (a+d)(e+h)
    # P2 = (c+d)e
    # P3 = a(f-h)
    # P4 = d(g-e)
    # P5 = (a+b)h
    # P6 = (c-a)(e+f)
    # P7 = (b-d)(g+h)
    
    # U coeffs for A (i, j_loc)
    # V coeffs for B (j_loc, k_loc)
    # W coeffs for C (k_loc, i) -> stored at W[k_global * n + i_global]
    
    # Local indices:
    # i: 0, 1
    # j_loc: 0, 1
    # k_loc: 0, 1
    
    # P1: U(0,0)=1, U(1,1)=1. V(0,0)=1, V(1,1)=1. W(0,0)=1, W(1,1)=1.
    # Note: W index is k_loc, i.
    # C_11 (i=0, k=0) gets +1. C_22 (i=1, k=1) gets +1.
    # So W[0,0] += 1, W[1,1] += 1.
    
    strassen_U = [
        { (0,0): 1, (1,1): 1 }, # P1
        { (1,0): 1, (1,1): 1 }, # P2 (c,d) -> (1,0) is c? No.
        # A = [[a, b], [c, d]]
        # a=(0,0), b=(0,1), c=(1,0), d=(1,1)
        # P2 = (c+d)e -> c=(1,0), d=(1,1).
        { (0,0): 1 },           # P3 (a)
        { (1,1): 1 },           # P4 (d)
        { (0,0): 1, (0,1): 1 }, # P5 (a,b)
        { (1,0): 1, (0,0): -1 },# P6 (c-a)
        { (0,1): 1, (1,1): -1 } # P7 (b-d)
    ]
    
    strassen_V = [
        { (0,0): 1, (1,1): 1 }, # P1 (e,h)
        { (0,0): 1 },           # P2 (e)
        { (0,1): 1, (1,1): -1 },# P3 (f-h)
        { (1,0): 1, (0,0): -1 },# P4 (g-e)
        { (1,1): 1 },           # P5 (h)
        { (0,0): 1, (0,1): 1 }, # P6 (e+f)
        { (1,0): 1, (1,1): 1 }  # P7 (g+h)
    ]
    
    # C = [[C11, C12], [C21, C22]]
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6
    
    # W coeffs for (k_loc, i)
    # i=0 is row 1, i=1 is row 2.
    # k_loc=0 is col 1, k_loc=1 is col 2.
    # W index in local block: (k_loc, i)
    
    strassen_W = [
        { (0,0): 1, (1,1): 1 }, # P1: C11(0,0)+1, C22(1,1)+1
        { (0,1): 1, (1,1): -1 },# P2: C21(0,1)+1, C22(1,1)-1
        { (0,0): 1, (1,1): 1 }, # P3: C12(0,0)+1? No.
        # C12 is i=0, k=1. So (k_loc=1, i=0).
        # C22 is i=1, k=1. So (k_loc=1, i=1).
        # P3 contributes to C12 and C22.
        # So (1,0): 1, (1,1): 1.
        # Let's re-verify.
        # C12 = P3 + P5. i=0, k=1.
        # C22 = P1 - P2 + P3 + P6. i=1, k=1.
        # So P3 adds to (1,0) and (1,1).
        { (1,0): 1, (1,1): 1 },
        { (0,0): 1, (0,1): 1 }, # P4: C11(0,0)+1, C21(0,1)+1
        { (0,0): -1, (0,1): 1 },# P5: C11(0,0)-1, C12(0,1)+1
        { (1,1): 1 },           # P6: C22(1,1)+1
        { (0,0): 1 }            # P7: C11(0,0)+1
    ]
    
    # Correcting Strassen W based on formulas:
    # C11 (i=0, k=0): P1 + P4 - P5 + P7
    # C12 (i=0, k=1): P3 + P5
    # C21 (i=1, k=0): P2 + P4
    # C22 (i=1, k=1): P1 - P2 + P3 + P6
    
    # P1: C11(0,0)+1, C22(1,1)+1. -> {(0,0):1, (1,1):1}
    # P2: C21(0,1)+1, C22(1,1)-1. -> {(0,1):1, (1,1):-1}
    # P3: C12(1,0)+1, C22(1,1)+1. -> {(1,0):1, (1,1):1}
    # P4: C11(0,0)+1, C21(0,1)+1. -> {(0,0):1, (0,1):1}
    # P5: C11(0,0)-1, C12(1,0)+1. -> {(0,0):-1, (1,0):1}
    # P6: C22(1,1)+1. -> {(1,1):1}
    # P7: C11(0,0)+1. -> {(0,0):1}
    
    # Update strassen_W list
    strassen_W = [
        { (0,0): 1, (1,1): 1 }, # P1
        { (0,1): 1, (1,1): -1 },# P2
        { (1,0): 1, (1,1): 1 }, # P3
        { (0,0): 1, (0,1): 1 }, # P4
        { (0,0): -1, (1,0): 1 },# P5
        { (1,1): 1 },           # P6
        { (0,0): 1 }            # P7
    ]

    # Iterate Strassen Blocks
    for J in range(2):
        for K in range(3):
            for prod in range(7):
                r = col_idx
                col_idx += 1
                
                # Fill U
                for (i, j_loc), val in strassen_U[prod].items():
                    j_global = J * 2 + j_loc
                    U[i * m + j_global][r] += val
                
                # Fill V
                for (j_loc, k_loc), val in strassen_V[prod].items():
                    j_global = J * 2 + j_loc
                    k_global = K * 2 + k_loc
                    V[j_global * p + k_global][r] += val
                
                # Fill W
                for (k_loc, i), val in strassen_W[prod].items():
                    k_global = K * 2 + k_loc
                    W[k_global * n + i][r] += val

    # Iterate Outer Product Blocks (J=3, i.e., j_global=4)
    j_global = 4
    for K in range(3):
        # 4 terms
        # 1. i=0, k_loc=0
        for i in range(2):
            for k_loc in range(2):
                r = col_idx
                col_idx += 1
                k_global = K * 2 + k_loc
                
                U[i * m + j_global][r] = 1
                V[j_global * p + k_global][r] = 1
                W[k_global * n + i][r] = 1

    return {"U": U, "V": V, "W": W}
