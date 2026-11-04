import numpy as np

def construct():
    n, m, p = 4, 4, 8
    
    # Rank 49 decomposition for 4x4x4
    # Based on recursive Strassen
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    # Strassen factors for 2x2x2 (Rank 7)
    # u_c, v_c, w_c are vectors of length 4
    # Indices: i*2+j
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0], # c=0..6 for i=0,j=0
        [0, 0, 0, 0, 1, 0, 1],  # i=0,j=1
        [0, 1, 0, 0, 0, 1, 0],  # i=1,j=0
        [1, 1, 0, 1, 0, 0, -1]  # i=1,j=1
    ])
    # Wait, my previous manual derivation:
    # u0 = [1, 0, 0, 1] -> col 0: [1, 0, 0, 1]
    # u1 = [0, 0, 1, 1] -> col 1: [0, 0, 1, 1]
    # u2 = [1, 0, 0, 0] -> col 2: [1, 0, 0, 0]
    # u3 = [0, 0, 0, 1] -> col 3: [0, 0, 0, 1]
    # u4 = [1, 1, 0, 0] -> col 4: [1, 1, 0, 0]
    # u5 = [-1, 0, 1, 0] -> col 5: [-1, 0, 1, 0]
    # u6 = [0, 1, 0, -1] -> col 6: [0, 1, 0, -1]
    
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ])
    
    V2 = np.array([
        [1, 1, 0, 0, 0, 1, 0], # j=0,k=0
        [0, 0, 1, 0, 0, 1, 0], # j=0,k=1
        [0, 0, 0, 1, 0, 0, 1], # j=1,k=0
        [1, 0, -1, -1, 1, 0, 1] # j=1,k=1
    ])
    # v0 = [1, 0, 0, 1]
    # v1 = [1, 0, 0, 0]
    # v2 = [0, 1, 0, -1]
    # v3 = [0, 0, 1, -1]
    # v4 = [0, 0, 0, 1]
    # v5 = [1, 1, 0, 0]
    # v6 = [0, 0, 1, 1]
    
    V2 = np.array([
        [1, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, -1, 1, 0, 1]
    ])
    
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1], # k=0,i=0
        [0, 0, 1, 0, 1, 0, 0],  # k=0,i=1
        [0, 1, 0, 1, 0, 0, 0],  # k=1,i=0
        [1, -1, 1, 0, 0, 1, 0]  # k=1,i=1
    ])
    # w0 = [1, 0, 0, 1]
    # w1 = [0, 0, 1, -1]
    # w2 = [0, 1, 0, 1]
    # w3 = [1, 0, 1, 0]
    # w4 = [-1, 1, 0, 0]
    # w5 = [0, 0, 0, 1]
    # w6 = [1, 0, 0, 0]
    
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ])

    # Strassen combination for 4x4
    # t=0: M1 = (A11+A22)(B11+B22)
    # A11 is block (0,0), A22 is block (1,1)
    # X00 = A00, X11 = A11 (in 2x2 block space)
    # Actually, X is 2x2 matrix of blocks.
    # X[0,0] = A[0,0] + A[1,1]
    # X[1,1] = A[1,1] ? No.
    # M1 = (A11+A22)(B11+B22).
    # A11 is block (0,0), A22 is block (1,1).
    # So X[0,0] = A[0,0], X[1,1] = A[1,1].
    # Wait, X is a 2x2 matrix.
    # X[0,0] = A[0,0] + A[1,1]? No, A[0,0] and A[1,1] are 2x2 blocks.
    # The sum A[0,0]+A[1,1] is a 2x2 matrix.
    # So X[0,0] = A[0,0]+A[1,1].
    # X[1,1] = 0? No, M1 is a single product.
    # The product is of two 2x2 matrices.
    # Let X = [[A00+A11, 0], [0, 0]]? No.
    # Strassen computes 7 products of 2x2 matrices.
    # M1 = (A00+A11)(B00+B11).
    # Here A00+A11 is a 2x2 matrix.
    # So the left factor of the 2x2 multiplication is X = A00+A11.
    # The right factor is Y = B00+B11.
    # So X is a 2x2 matrix.
    # X[0,0] = A[0,0][0,0] + A[1,1][0,0]
    # X[0,1] = A[0,0][0,1] + A[1,1][0,1]
    # X[1,0] = A[0,0][1,0] + A[1,1][1,0]
    # X[1,1] = A[0,0][1,1] + A[1,1][1,1]
    
    # So for t=0:
    # alpha[0,0, i'', j''] = 1 if (i'', j'') is in block (0,0) or (1,1).
    # Specifically, X[i', j'] = sum over blocks.
    # X[0,0] = A[0,0] + A[1,1].
    # So for i'=0, j'=0:
    # X[0,0][i'', j''] = A[0,0][i'', j''] + A[1,1][i'', j''].
    # A[0,0] corresponds to global i=0,1; j=0,1.
    # A[1,1] corresponds to global i=2,3; j=2,3.
    
    # We can define the coefficients alpha, beta for each t.
    
    # This is getting complex to hardcode.
    # I will write a helper to generate the coefficients.
    
    def get_coeffs(t):
        # Returns alpha (2x2x2x2), beta (2x2x2x2)
        # alpha[i', j', i'', j'']
        alpha = np.zeros((2, 2, 2, 2))
        beta = np.zeros((2, 2, 2, 2))
        
        if t == 0: # M1
            # X = A00 + A11
            alpha[0,0,0,0] = 1; alpha[0,0,0,1] = 1; alpha[0,0,1,0] = 1; alpha[0,0,1,1] = 1
            alpha[1,1,0,0] = 1; alpha[1,1,0,1] = 1; alpha[1,1,1,0] = 1; alpha[1,1,1,1] = 1
            # Y = B00 + B11
            beta[0,0,0,0] = 1; beta[0,0,0,1] = 1; beta[0,0,1,0] = 1; beta[0,0,1,1] = 1
            beta[1,1,0,0] = 1; beta[1,1,0,1] = 1; beta[1,1,1,0] = 1; beta[1,1,1,1] = 1
        elif t == 1: # M2
            # X = A10 + A11
            alpha[1,0,0,0] = 1; alpha[1,0,0,1] = 1; alpha[1,0,1,0] = 1; alpha[1,0,1,1] = 1
            alpha[1,1,0,0] = 1; alpha[1,1,0,1] = 1; alpha[1,1,1,0] = 1; alpha[1,1,1,1] = 1
            # Y = B00
            beta[0,0,0,0] = 1; beta[0,0,0,1] = 1; beta[0,0,1,0] = 1; beta[0,0,1,1] = 1
        elif t == 2: # M3
            # X = A00
            alpha[0,0,0,0] = 1; alpha[0,0,0,1] = 1; alpha[0,0,1,0] = 1; alpha[0,0,1,1] = 1
            # Y = B01 - B11
            beta[0,1,0,0] = 1; beta[0,1,0,1] = 1; beta[0,1,1,0] = 1; beta[0,1,1,1] = 1
            beta[1,1,0,0] = -1; beta[1,1,0,1] = -1; beta[1,1,1,0] = -1; beta[1,1,1,1] = -1
        elif t == 3: # M4
            # X = A11
            alpha[1,1,0,0] = 1; alpha[1,1,0,1] = 1; alpha[1,1,1,0] = 1; alpha[1,1,1,1] = 1
            # Y = B10 - B00
            beta[1,0,0,0] = 1; beta[1,0,0,1] = 1; beta[1,0,1,0] = 1; beta[1,0,1,1] = 1
            beta[0,0,0,0] = -1; beta[0,0,0,1] = -1; beta[0,0,1,0] = -1; beta[0,0,1,1] = -1
        elif t == 4: # M5
            # X = A00 + A01
            alpha[0,0,0,0] = 1; alpha[0,0,0,1] = 1; alpha[0,0,1,0] = 1; alpha[0,0,1,1] = 1
            alpha[0,1,0,0] = 1; alpha[0,1,0,1] = 1; alpha[0,1,1,0] = 1; alpha[0,1,1,1] = 1
            # Y = B11
            beta[1,1,0,0] = 1; beta[1,1,0,1] = 1; beta[1,1,1,0] = 1; beta[1,1,1,1] = 1
        elif t == 5: # M6
            # X = A10 - A00
            alpha[1,0,0,0] = 1; alpha[1,0,0,1] = 1; alpha[1,0,1,0] = 1; alpha[1,0,1,1] = 1
            alpha[0,0,0,0] = -1; alpha[0,0,0,1] = -1; alpha[0,0,1,0] = -1; alpha[0,0,1,1] = -1
            # Y = B00 + B01
            beta[0,0,0,0] = 1; beta[0,0,0,1] = 1; beta[0,0,1,0] = 1; beta[0,0,1,1] = 1
            beta[0,1,0,0] = 1; beta[0,1,0,1] = 1; beta[0,1,1,0] = 1; beta[0,1,1,1] = 1
        elif t == 6: # M7
            # X = A01 - A11
            alpha[0,1,0,0] = 1; alpha[0,1,0,1] = 1; alpha[0,1,1,0] = 1; alpha[0,1,1,1] = 1
            alpha[1,1,0,0] = -1; alpha[1,1,0,1] = -1; alpha[1,1,1,0] = -1; alpha[1,1,1,1] = -1
            # Y = B10 + B11
            beta[1,0,0,0] = 1; beta[1,0,0,1] = 1; beta[1,0,1,0] = 1; beta[1,0,1,1] = 1
            beta[1,1,0,0] = 1; beta[1,1,0,1] = 1; beta[1,1,1,0] = 1; beta[1,1,1,1] = 1
            
        return alpha, beta

    for t in range(7):
        alpha, beta = get_coeffs(t)
        for c in range(7):
            r = t * 7 + c
            for i in range(4):
                for j in range(4):
                    ip = i // 2
                    iqq = i % 2
                    jp = j // 2
                    jqq = j % 2
                    U4[i*4+j, r] = U2[ip*2+jp, c] * alpha[ip, jp, iqq, jqq]
                    
                    for k in range(4):
                        kp = k // 2
                        kqq = k % 2
                        V4[j*4+k, r] = V2[jp*2+kp, c] * beta[jp, kp, jqq, kqq]
                        
                        W4[k*4+i, r] = W2[kp*2+ip, c]
                        
    # Now construct 4x4x8
    U8 = np.zeros((16, 98))
    V8 = np.zeros((32, 98))
    W8 = np.zeros((32, 98))
    
    # First 49 columns
    U8[:, :49] = U4
    V8[:16, :49] = V4
    V8[16:, :49] = 0
    W8[:16, :49] = W4
    W8[16:, :49] = 0
    
    # Next 49 columns
    U8[:, 49:] = U4
    V8[:16, 49:] = 0
    V8[16:, 49:] = V4
    W8[:16, 49:] = 0
    W8[16:, 49:] = W4
    
    return {"U": U8.tolist(), "V": V8.tolist(), "W": W8.tolist()}

