import numpy as np

DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We partition the p=8 dimension into 4 blocks of size 2.
    # Each block is a 2x2x2 multiplication.
    # Rank of 2x2x2 is 7 (Strassen).
    # Total rank = 4 * 7 = 28.
    
    rank = 28
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # A = [[a00, a01], [a10, a11]]
    # B = [[b00, b01], [b10, b11]]
    # C = [[c00, c01], [c10, c11]]
    
    # 7 terms
    # Term 0: (a00+a11)(b00+b11) -> c00+=1, c01+=1, c10+=1, c11+=1? No.
    # Standard Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Define U, V, W for each term
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    
    # For a block l, k indices are 2l, 2l+1.
    # Let k0 = 2l, k1 = 2l+1.
    
    # Term 0: M1
    # A part: a00=1, a11=1
    # B part: b00=1, b11=1
    # C part: c00=1, c01=1, c10=1, c11=1? No, M1 contributes to all?
    # C00 += M1, C11 += M1.
    # C01, C10 do NOT get M1 directly in the standard formula?
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # So:
    # M1: C00 += 1, C11 += 1
    # M2: C10 += 1, C11 -= 1
    # M3: C01 += 1, C11 += 1
    # M4: C00 += 1, C10 += 1
    # M5: C00 -= 1, C01 += 1
    # M6: C11 += 1
    # M7: C00 += 1
    
    # Let's list (U_coeffs, V_coeffs, W_coeffs) for each term
    # U is 2x2, V is 2x2 (for the block), W is 2x2 (for the block)
    
    terms = [
        # (U_00, U_01, U_10, U_11, V_00, V_01, V_10, V_11, W_00, W_01, W_10, W_11)
        # Note: V index is j*p+k. For block l, j=0,1 and k=0,1 (local).
        # W index is k*n+i. For block l, k=0,1 (local) and i=0,1.
        
        # Term 0: M1
        (1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1), # C00+=1, C11+=1. W_00=1, W_11=1.
        # Term 1: M2
        (0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, -1), # C10+=1, C11-=1. W_10=1, W_11=-1.
        # Term 2: M3
        (1, 0, 0, 0, 0, 1, 0, -1, 0, 1, 0, 1), # C01+=1, C11+=1. W_01=1, W_11=1.
        # Term 3: M4
        (0, 0, 0, 1, 0, 0, 1, -1, 1, 0, 1, 0), # C00+=1, C10+=1. W_00=1, W_10=1.
        # Term 4: M5
        (1, 1, 0, 0, 0, 0, 0, 1, -1, 1, 0, 0), # C00-=1, C01+=1. W_00=-1, W_01=1.
        # Term 5: M6
        (0, 0, 1, -1, 1, 1, 0, 0, 0, 0, 0, 1), # C11+=1. W_11=1.
        # Term 6: M7
        (0, 1, 0, -1, 0, 0, 1, 1, 1, 0, 0, 0), # C00+=1. W_00=1.
    ]
    
    for l in range(4):
        for s in range(7):
            r = 7 * l + s
            u, v, w = terms[s]
            
            # U indices: i*m + j
            # u = (U00, U01, U10, U11)
            U[0*2+0][r] = u[0]
            U[0*2+1][r] = u[1]
            U[1*2+0][r] = u[2]
            U[1*2+1][r] = u[3]
            
            # V indices: j*p + k
            # v = (V00, V01, V10, V11) corresponding to j=0,k=0; j=0,k=1; j=1,k=0; j=1,k=1
            # Local k=0 is global 2l, local k=1 is global 2l+1
            V[0*8 + (2*l)    ][r] = v[0]
            V[0*8 + (2*l + 1)][r] = v[1]
            V[1*8 + (2*l)    ][r] = v[2]
            V[1*8 + (2*l + 1)][r] = v[3]
            
            # W indices: k*n + i
            # w = (W00, W01, W10, W11) corresponding to k=0,i=0; k=0,i=1; k=1,i=0; k=1,i=1
            # Local k=0 is global 2l, local k=1 is global 2l+1
            W[(2*l)    *2 + 0][r] = w[0]
            W[(2*l)    *2 + 1][r] = w[1]
            W[(2*l + 1)*2 + 0][r] = w[2]
            W[(2*l + 1)*2 + 1][r] = w[3]
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
