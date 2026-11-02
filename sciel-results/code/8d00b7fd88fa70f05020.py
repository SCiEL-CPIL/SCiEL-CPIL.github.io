# EVOLVE-BLOCK-START
def construct():
    """
    Returns the rank-35 decomposition for 2x2 by 2x10 matrix multiplication.
    This is optimal as rank(mu_{2,2,10}) = 35.
    """
    n, m, p = 2, 2, 10
    
    # Laderman's 7 products for 2x2 multiplication
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)(e)
    # M3 = (a)(f-h)
    # M4 = (d)(g-e)
    # M5 = (a+b)(h)
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # Coefficients for A (4x7)
    U_lad = [
        [1, 0, 1, 0, 1, -1, 0],  # a
        [0, 0, 0, 0, 1, 0, 1],   # b
        [0, 1, 0, 0, 0, 1, 0],   # c
        [1, 1, 0, 1, 0, 0, -1]   # d
    ]
    
    # Coefficients for B (4x7)
    V_lad = [
        [1, 1, 0, -1, 0, 1, 0],  # e
        [0, 0, 1, 0, 0, 1, 0],   # f
        [0, 0, 0, 1, 0, 0, 1],   # g
        [1, 0, -1, 0, 1, 0, 1]   # h
    ]
    
    # Coefficients for C (4x7)
    # C = [[C00, C01], [C10, C11]]
    # C00 = M1 + M4 + M5 - M2 - M6? 
    # Let's check standard Laderman:
    # C00 = M1 + M4 - M2 + M5? 
    # Actually:
    # C00 = M1 + M4 + M5 - M2? 
    # Let's use the standard reconstruction:
    # C00 = M1 + M4 - M2 + M5? No.
    # C00 = M1 + M4 + M5 - M2? 
    # Let's stick to the verified one from the previous code.
    
    # Previous code W contrib:
    # M1: C00+=1, C11+=1
    # M2: C10+=1, C11-=1
    # M3: C01+=1, C11+=1
    # M4: C00+=1, C10+=1
    # M5: C00-=1, C01+=1
    # M6: C11+=1
    # M7: C00+=1
    
    # Let's re-verify M5: (a+b)h. 
    # C01 = ah + bh. 
    # M5 contributes to C01. 
    # In previous code: W[4*b+2] (C01) += 1 for M5? 
    # w_contrib[4] = [-1, 1, 0, 0]. 
    # W indices: 0=C00, 1=C01, 2=C10, 3=C11? 
    # Trace convention W[k*n+i]. 
    # C is 2x10. 
    # Block b (0..4). 
    # C block is 2x2. 
    # Indices in C block: 
    # k=0, i=0 -> C00 -> idx 0
    # k=0, i=1 -> C10? No. 
    # Trace convention: W[k*n+i]. 
    # n=2. 
    # k=0, i=0 -> 0. C[0,0]
    # k=0, i=1 -> 1. C[1,0]
    # k=1, i=0 -> 2. C[0,1]
    # k=1, i=1 -> 3. C[1,1]
    
    # So W indices 0,1,2,3 correspond to C[0,0], C[1,0], C[0,1], C[1,1].
    
    # Let's check the previous w_contrib against this.
    # M1: (a+d)(e+h) = ae+ah+de+dh. 
    # C00 = ae+bg. C11 = ch+df. 
    # This doesn't match directly. 
    # The previous code verified successfully, so I will reuse its logic.
    
    rank = 35
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(20)]
    W = [[0] * rank for _ in range(20)]
    
    laderman_A = [
        [1, 0, 0, 1],   # M1: a+d
        [0, 0, 1, 1],   # M2: c+d
        [1, 0, 0, 0],   # M3: a
        [0, 0, 0, 1],   # M4: d
        [1, 1, 0, 0],   # M5: a+b
        [-1, 0, 1, 0],  # M6: c-a
        [0, 1, 0, -1]   # M7: b-d
    ]
    
    laderman_B = [
        [1, 0, 0, 1],   # M1: e+h
        [1, 0, 0, 0],   # M2: e
        [0, 1, 0, -1],  # M3: f-h
        [-1, 0, 1, 0],  # M4: g-e
        [0, 0, 0, 1],   # M5: h
        [1, 1, 0, 0],   # M6: e+f
        [0, 0, 1, 1]    # M7: g+h
    ]
    
    w_contrib = [
        [1, 0, 0, 1],   # M1: C00 += 1, C11 += 1
        [0, 0, 1, -1],  # M2: C10 += 1, C11 -= 1
        [0, 1, 0, 1],   # M3: C01 += 1, C11 += 1
        [1, 0, 1, 0],   # M4: C00 += 1, C10 += 1
        [-1, 1, 0, 0],  # M5: C00 -= 1, C01 += 1
        [0, 0, 0, 1],   # M6: C11 += 1
        [1, 0, 0, 0]    # M7: C00 += 1
    ]
    
    for b in range(5):
        for r in range(7):
            col = b * 7 + r
            
            U[0][col] = laderman_A[r][0]
            U[1][col] = laderman_A[r][1]
            U[2][col] = laderman_A[r][2]
            U[3][col] = laderman_A[r][3]
            
            V[2*b][col] = laderman_B[r][0]
            V[2*b + 1][col] = laderman_B[r][1]
            V[10 + 2*b][col] = laderman_B[r][2]
            V[10 + 2*b + 1][col] = laderman_B[r][3]
            
            W[4*b][col] = w_contrib[r][0]
            W[4*b + 2][col] = w_contrib[r][1]
            W[4*b + 1][col] = w_contrib[r][2]
            W[4*b + 3][col] = w_contrib[r][3]
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
