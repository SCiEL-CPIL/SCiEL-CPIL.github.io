import numpy as np
import time

DIMENSIONS = [2, 2, 13]

def construct():
    n, m, p = DIMENSIONS
    # Rank is 6 * 7 + 4 = 46
    rank = 46
    
    # U: (n*m, rank) = (4, 46)
    # V: (m*p, rank) = (26, 46)
    # W: (p*n, rank) = (26, 46)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a rank-1 term
    # u: dict { (i,j): val }
    # v: dict { (j,k): val }
    # w: dict { (k,i): val }
    def add_term(u_coeffs, v_coeffs, w_coeffs):
        nonlocal col
        for (i, j), val in u_coeffs.items():
            if val != 0:
                U[i * m + j][col] = val
        for (j, k), val in v_coeffs.items():
            if val != 0:
                V[j * p + k][col] = val
        for (k, i), val in w_coeffs.items():
            if val != 0:
                W[k * n + i][col] = val
        col += 1

    # Strassen for 6 pairs of columns
    # Pairs: (0,1), (2,3), (4,5), (6,7), (8,9), (10,11)
    for t in range(6):
        k1 = 2 * t
        k2 = 2 * t + 1
        
        # Strassen M1: (A00+A11)(B0_k1 + B1_k2)
        # Note: Standard Strassen M1 uses B00 and B11.
        # Here B00 = B[0, k1], B11 = B[1, k2]
        # Wait, let's stick to the standard definition:
        # A = [[A00, A01], [A10, A11]]
        # B = [[B00, B01], [B10, B11]]
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A11)(B01+B11)
        # M7 = (A01-A10)(B00+B10)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # M1
        add_term(
            {(0,0): 1, (1,1): 1},
            {(0, k1): 1, (1, k2): 1},
            {(k1, 0): 1, (k2, 1): 1}
        )
        
        # M2
        add_term(
            {(1,0): 1, (1,1): 1},
            {(0, k1): 1},
            {(k1, 1): 1}
        )
        
        # M3
        add_term(
            {(0,0): 1},
            {(0, k2): 1, (1, k2): -1},
            {(k2, 0): 1}
        )
        
        # M4
        add_term(
            {(1,1): 1},
            {(1, k1): 1, (0, k1): -1},
            {(k1, 0): 1, (k1, 1): 1}
        )
        
        # M5
        add_term(
            {(0,0): 1, (0,1): 1},
            {(1, k2): 1},
            {(k1, 0): -1, (k2, 0): 1}
        )
        
        # M6
        add_term(
            {(1,0): 1, (1,1): -1},
            {(0, k2): 1, (1, k2): 1},
            {(k2, 1): 1}
        )
        
        # M7
        add_term(
            {(0,1): 1, (1,0): -1},
            {(0, k1): 1, (1, k1): 1},
            {(k1, 0): 1}
        )
        
    # Schoolbook for column 12
    k = 12
    # C0_k = A00 B0_k + A01 B1_k
    # C1_k = A10 B0_k + A11 B1_k
    
    # Term 1: A00 B0_k C0_k
    add_term(
        {(0,0): 1},
        {(0, k): 1},
        {(k, 0): 1}
    )
    
    # Term 2: A01 B1_k C0_k
    add_term(
        {(0,1): 1},
        {(1, k): 1},
        {(k, 0): 1}
    )
    
    # Term 3: A10 B0_k C1_k
    add_term(
        {(1,0): 1},
        {(0, k): 1},
        {(k, 1): 1}
    )
    
    # Term 4: A11 B1_k C1_k
    add_term(
        {(1,1): 1},
        {(1, k): 1},
        {(k, 1): 1}
    )
    
    return {"U": U, "V": V, "W": W}
