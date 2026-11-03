def construct():
    n, m, p = 2, 4, 8
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen Coeffs
    # L: 7 x 4
    L = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 1, 0],
        [-1, 0, 0, 1],
        [0, 1, 0, -1]
    ]
    # K: 7 x 4
    K = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 0, 1, -1],
        [1, -1, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    # W: 7 x 4
    W_coeff = [
        [1, 0, 0, 1],  # Z00, Z01, Z10, Z11 for M1
        [0, 0, 1, -1], # for M2
        [0, 1, 0, 1],  # for M3
        [1, 0, 1, 0],  # for M4
        [-1, 1, 0, 1], # for M5
        [0, 0, 0, 1],  # for M6
        [1, 0, 0, 0]   # for M7
    ]
    # Check W_coeff against formulas:
    # Z00: M1(1), M4(1), M5(-1), M7(1) -> [1,0,0,1] for M1? No.
    # W_coeff[s] is the vector for M_s.
    # M1: Z00+=1, Z11+=1. -> [1, 0, 0, 1]
    # M2: Z10+=1, Z11+=-1. -> [0, 0, 1, -1]
    # M3: Z01+=1, Z11+=1. -> [0, 1, 0, 1]
    # M4: Z00+=1, Z10+=1. -> [1, 0, 1, 0]
    # M5: Z00+=-1, Z01+=1. -> [-1, 1, 0, 0]  <-- Wait.
    # Z00 = M1 + M4 - M5 + M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 + M6
    
    # M5: Z00=-1, Z01=1. -> [-1, 1, 0, 0]
    # M6: Z11=1. -> [0, 0, 0, 1]
    # M7: Z00=1. -> [1, 0, 0, 0]
    
    # Let's re-verify M5.
    # M5 = (X00+X10)Y11.
    # Z00 = M1 + M4 - M5 + M7. Yes.
    # Z01 = M3 + M5. Yes.
    # So M5 contributes -1 to Z00 and 1 to Z01.
    # Vector: [-1, 1, 0, 0].
    
    # My previous W_coeff for M5 was [-1, 1, 0, 1]. That was wrong.
    # Let's list them carefully.
    
    W_coeff = [
        [1, 0, 0, 1],  # M1
        [0, 0, 1, -1], # M2
        [0, 1, 0, 1],  # M3
        [1, 0, 1, 0],  # M4
        [-1, 1, 0, 0], # M5
        [0, 0, 0, 1],  # M6
        [1, 0, 0, 0]   # M7
    ]

    for b in range(2):
        for a in range(2):
            for c in range(2):
                for s in range(7):
                    r = (b * 4 + a * 2 + c) * 7 + s
                    
                    # U
                    for i in range(2):
                        for j_prime in range(2):
                            idx_A = i * 4 + (2 * a + j_prime)
                            U[idx_A][r] += L[s][i * 2 + j_prime]
                            
                    # V
                    for i in range(2):
                        for j_prime in range(2):
                            idx_B = (2 * a + i) * 8 + (4 * b + 2 * c + j_prime)
                            V[idx_B][r] += K[s][i * 2 + j_prime]
                            
                    # W
                    for i in range(2):
                        for j_prime in range(2):
                            idx_C = (4 * b + 2 * c + j_prime) * 2 + i
                            W[idx_C][r] += W_coeff[s][i * 2 + j_prime]

    return {"U": U, "V": V, "W": W}
