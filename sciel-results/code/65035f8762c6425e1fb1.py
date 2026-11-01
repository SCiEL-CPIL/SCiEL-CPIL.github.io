import numpy as np

DIMENSIONS = [2, 2, 8]

def construct():
    n, m, p = DIMENSIONS
    # Rank is 4 * 7 = 28
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 multiplication
    # U_part: coefficients for A
    # V_part: coefficients for B
    # W_part: coefficients for C
    # Indexing:
    # A: i*2+j
    # B: j*8+k
    # C: k*2+i
    
    # Strassen products for one block
    # r=0..6
    # U_r: [u00, u01, u10, u11]
    # V_r: [v00, v01, v10, v11] (relative to block)
    # W_r: [w00, w01, w10, w11] (relative to block)
    
    # Using the standard Strassen algorithm
    # P1 = (a00+a11)(b00+b11)
    # P2 = (a10+a11)b00
    # P3 = a00(b10-b11)
    # P4 = a11(b01-b00)
    # P5 = (a00+a01)b11
    # P6 = (a10-a00)(b00+b01)
    # P7 = (a01-a11)(b10+b11)
    
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # Let's verify C01 again.
    # P3 = a00(b10-b11)
    # P5 = (a00+a01)b11
    # P3+P5 = a00b10 - a00b11 + a00b11 + a01b11 = a00b10 + a01b11
    # But C01 should be a00b01 + a01b11.
    # So P3 must be a00(b01-b11)?
    # If P3 = a00(b01-b11), then P3+P5 = a00b01 - a00b11 + a00b11 + a01b11 = a00b01 + a01b11. Correct.
    # So P3 = a00(b01-b11).
    
    # Let's check C10.
    # C10 = P2 + P4
    # P2 = (a10+a11)b00
    # P4 = a11(b01-b00)
    # P2+P4 = a10b00 + a11b00 + a11b01 - a11b00 = a10b00 + a11b01. Correct.
    
    # Let's check C11.
    # C11 = P1 - P2 + P3 + P6
    # P1 = a00b00 + a00b11 + a11b00 + a11b11
    # -P2 = -a10b00 - a11b00
    # P3 = a00b01 - a00b11
    # P6 = (a10-a00)(b00+b01) = a10b00 + a10b01 - a00b00 - a00b01
    # Sum:
    # a00b00: 1 - 1 = 0
    # a00b11: 1 - 1 = 0
    # a11b00: 1 - 1 = 0
    # a11b11: 1
    # a10b00: -1 + 1 = 0
    # a00b01: -1 - 1 = -2? No.
    # P3 has -a00b11. P6 has -a00b01.
    # a00b01: from P3 (0) + P6 (-1) = -1?
    # Wait. P3 = a00(b01-b11). So a00b01 is +1.
    # P6 = a10b00 + a10b01 - a00b00 - a00b01.
    # a00b01: P3(+1) + P6(-1) = 0.
    # a10b01: P6(+1) = 1.
    # a11b11: 1.
    # Result: a11b11 + a10b01. Correct.
    
    # So the correct P3 is a00(b01-b11).
    
    strassen_U = [
        [1, 0, 1, 0], # P1: a00+a11 -> [1, 0, 0, 1] ? No.
        # a00 is index 0, a01 is 1, a10 is 2, a11 is 3.
        # P1: a00+a11 -> [1, 0, 0, 1]
        # P2: a10+a11 -> [0, 0, 1, 1]
        # P3: a00 -> [1, 0, 0, 0]
        # P4: a11 -> [0, 0, 0, 1]
        # P5: a00+a01 -> [1, 1, 0, 0]
        # P6: a10-a00 -> [-1, 0, 1, 0]
        # P7: a01-a11 -> [0, 1, 0, -1]
    ]
    strassen_U = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    
    # V coefficients for B (relative to block: b00, b01, b10, b11)
    # P1: b00+b11 -> [1, 0, 0, 1]
    # P2: b00 -> [1, 0, 0, 0]
    # P3: b01-b11 -> [0, 1, 0, -1]
    # P4: b01-b00 -> [-1, 1, 0, 0]
    # P5: b11 -> [0, 0, 0, 1]
    # P6: b00+b01 -> [1, 1, 0, 0]
    # P7: b10+b11 -> [0, 0, 1, 1]
    strassen_V = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    
    # W coefficients for C (relative to block: c00, c01, c10, c11)
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # W_r is the contribution of P_r to C.
    # W_0 (P1): c00+=1, c11+=1 -> [1, 0, 0, 1]
    # W_1 (P2): c10+=1, c11-=1 -> [0, 0, 1, -1]
    # W_2 (P3): c01+=1, c11+=1 -> [0, 1, 0, 1]
    # W_3 (P4): c00+=1, c10+=1 -> [1, 0, 1, 0]
    # W_4 (P5): c00-=1, c01+=1 -> [-1, 1, 0, 0]
    # W_5 (P6): c11+=1 -> [0, 0, 0, 1]
    # W_6 (P7): c00+=1 -> [1, 0, 0, 0]
    
    strassen_W = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]
    
    # Fill U, V, W
    for q in range(4):
        for r in range(7):
            t = q * 7 + r
            
            # U
            for i in range(2):
                for j in range(2):
                    idx = i * m + j
                    U[idx][t] = strassen_U[r][i * 2 + j]
            
            # V
            for j in range(2):
                for k in range(2):
                    # B index: j * p + (2q + k)
                    idx = j * p + (2 * q + k)
                    V[idx][t] = strassen_V[r][j * 2 + k]
            
            # W
            for k in range(2):
                for i in range(2):
                    # C index: k * n + i
                    # Block q corresponds to k' in {0,1} relative to block, so k = 2q + k'
                    idx = (2 * q + k) * n + i
                    W[idx][t] = strassen_W[r][k * 2 + i]
    
    return {"U": U, "V": V, "W": W}
