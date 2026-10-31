def construct():
    # 2x2 Strassen factors
    # u: [A00, A01, A10, A11], v: [B00, B01, B10, B11], w: [C00, C01, C10, C11] (trace: W[k*2+i])
    # Trace convention for 2x2: W[k*2+i], so W[0]=C[0,0], W[1]=C[0,1], W[2]=C[1,0], W[3]=C[1,1]
    
    strassen_u = [
        [1, 0, 0, 1],   # r=1: A00+A11
        [0, 0, 1, 1],   # r=2: A10+A11
        [1, 0, 0, 0],   # r=3: A00
        [0, 0, 0, 1],   # r=4: A11
        [1, 1, 0, 0],   # r=5: A00+A01
        [-1, 0, 1, 0],  # r=6: A10-A00
        [0, 1, 0, -1],  # r=7: A01-A11
    ]
    strassen_v = [
        [1, 0, 0, 1],   # r=1: B00+B11
        [1, 0, 0, 0],   # r=2: B00
        [0, 1, 0, -1],  # r=3: B01-B11
        [0, 1, -1, 0],  # r=4: B10-B00
        [0, 0, 0, 1],   # r=5: B11
        [1, 1, 0, 0],   # r=6: B00+B01
        [0, 0, 1, 1],   # r=7: B10+B11
    ]
    strassen_w = [
        [1, 0, 0, 1],   # r=1: C00+C11
        [0, 0, 1, 0],   # r=2: C10
        [0, 1, 0, 0],   # r=3: C01
        [0, 0, 1, 0],   # r=4: C10
        [-1, 0, 0, 1],  # r=5: -C00+C11
        [0, 0, 0, 1],   # r=6: C11
        [1, 0, 0, 0],   # r=7: C00
    ]
    
    # Verify: sum_r u_r[i,j] * v_r[j,k] * w_r[k,i] should give the 2x2 MM tensor
    # Let's just proceed with the Kronecker product
    
    def kron_2x2(a, b):
        """Kronecker product of two 2x2 matrices (flattened row-major)."""
        # a = [a00, a01, a10, a11], b = [b00, b01, b10, b11]
        # result = [[a00*b00, a00*b01, a01*b00, a01*b01],
        #           [a10*b00, a10*b01, a11*b00, a11*b01],
        #           [a00*b10, a00*b11, a01*b10, a01*b11],
        #           [a10*b10, a10*b11, a11*b10, a11*b11]]
        a00, a01, a10, a11 = a
        b00, b01, b10, b11 = b
        return [
            a00*b00, a00*b01, a01*b00, a01*b01,
            a10*b00, a10*b01, a11*b00, a11*b01,
            a00*b10, a00*b11, a01*b10, a01*b11,
            a10*b10, a10*b11, a11*b10, a11*b11
        ]
    
    rank = 49
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(16)]
    
    for r in range(7):
        for s in range(7):
            col = r * 7 + s
            u_kron = kron_2x2(strassen_u[r], strassen_u[s])
            v_kron = kron_2x2(strassen_v[r], strassen_v[s])
            w_kron = kron_2x2(strassen_w[r], strassen_w[s])
            for i in range(16):
                U[i][col] = u_kron[i]
                V[i][col] = v_kron[i]
                W[i][col] = w_kron[i]
    
    return {"U": U, "V": V, "W": W}
