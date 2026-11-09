import time
import sys

def construct():
    # Strassen coefficients for 2x2
    # m0 = a00+a11, m1 = a10+a11, m2 = a00, m3 = a11, m4 = a00-a10, m5 = a01+a11, m6 = a11-a01
    # U (A side)
    # 0: a00+a11 -> [1, 0, 0, 1]
    # 1: a10+a11 -> [0, 0, 1, 1]
    # 2: a00     -> [1, 0, 0, 0]
    # 3: a11     -> [0, 0, 0, 1]
    # 4: a00-a10 -> [1, -1, 0, 0] (Wait, standard Strassen: m4 = a00 - a10? No, m4 = a00 - a10 is not standard. 
    # Standard: 
    # m0 = (a00+a11)(b00+b11)
    # m1 = (a10+a11)b00
    # m2 = a00(b10-b11)
    # m3 = a11(b01-b00)
    # m4 = (a00+a01)b11
    # m5 = (a10-a00)(b00+b10)
    # m6 = (a01-a11)(b01+b11)
    
    # Let's stick to the forms in the previous code which worked:
    # X_forms (A coefficients for 4 elements a00, a01, a10, a11)
    # m0: [1, 0, 0, 1]  (a00 + a11)
    # m1: [0, 0, 1, 1]  (a10 + a11) -> Wait, previous code had [0,0,1,1] for m1.
    # m2: [1, 0, 0, 0]  (a00)
    # m3: [0, 0, 0, 1]  (a11)
    # m4: [1, 1, 0, 0]  (a00 + a01)
    # m5: [-1, 0, 1, 0] (a10 - a00)
    # m6: [0, 1, 0, -1] (a01 - a11)
    
    # Y_forms (B coefficients for 4 elements b00, b01, b10, b11)
    # m0: [1, 0, 0, 1]  (b00 + b11)
    # m1: [1, 0, 0, 0]  (b00)
    # m2: [0, 1, 0, -1] (b01 - b11)
    # m3: [-1, 0, 1, 0] (b10 - b00)
    # m4: [0, 0, 0, 1]  (b11)
    # m5: [1, 1, 0, 0]  (b00 + b10)
    # m6: [0, 0, 1, 1]  (b01 + b11)
    
    # Z_combos (C coefficients for 4 elements c00, c01, c10, c11)
    # c00 = m0 - m3 - m5 + m6
    # c01 = m4 + m5
    # c10 = m1 + m3
    # c11 = m0 - m2 + m4 - m1 ? No.
    # c11 = m0 - m2 + m4 - m1 is not right.
    # c11 = m0 - m2 + m4 - m1? 
    # Let's check the previous Z_combos:
    # z00: [1, 0, 0, 1, -1, 0, 1] -> m0 + m3 - m5 + m6? 
    # Previous: [1, 0, 0, 1, -1, 0, 1] corresponds to m0, m1, m2, m3, m4, m5, m6.
    # So c00 = m0 + m3 - m5 + m6.
    # Standard Strassen:
    # c00 = m0 - m3 - m5 + m6? 
    # Let's re-verify standard Strassen formulas.
    # c00 = m0 - m3 - m5 + m6
    # c01 = m4 + m5
    # c10 = m1 + m3
    # c11 = m0 - m2 + m4 - m1
    # 
    # Let's check the previous code's Z_combos again.
    # Z_combos[0] (z00) = [1, 0, 0, 1, -1, 0, 1] -> m0 + m3 - m5 + m6.
    # Is c00 = m0 + m3 - m5 + m6?
    # m0 = (a00+a11)(b00+b11)
    # m3 = a11(b01-b00)
    # m5 = (a10-a00)(b00+b10)
    # m6 = (a01-a11)(b01+b11)
    # This combination seems specific to the coefficient sets used.
    # Since the previous code passed verification (score 1.1429), the coefficients are correct for that specific X/Y set.
    # I will reuse the exact same coefficient arrays to ensure correctness.
    
    X_forms = [
        [1, 0, 0, 1],   
        [0, 0, 1, 1],   
        [1, 0, 0, 0],   
        [0, 0, 0, 1],   
        [1, 1, 0, 0],   
        [-1, 0, 1, 0],  
        [0, 1, 0, -1]   
    ]
    
    Y_forms = [
        [1, 0, 0, 1],   
        [1, 0, 0, 0],   
        [0, 1, 0, -1],  
        [-1, 0, 1, 0],  
        [0, 0, 0, 1],   
        [1, 1, 0, 0],   
        [0, 0, 1, 1]   
    ]
    
    Z_combos = [
        [1, 0, 0, 1, -1, 0, 1],   
        [0, 0, 1, 0, 1, 0, 0],    
        [0, 1, 0, 1, 0, 0, 0],    
        [1, -1, 1, 0, 0, 1, 0]    
    ]
    
    n, m, p = 2, 4, 8
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Distinct construction loop
    r = 0
    for g in range(4): # 4 groups of 2 cols in C
        for b in range(2): # 2 blocks of 2 cols in A
            for s in range(7): # 7 Strassen terms
                # U: A[i, b*2+jp]
                for i in range(2):
                    for jp in range(2):
                        a_idx = i * 4 + b * 2 + jp
                        U[a_idx][r] = X_forms[s][i * 2 + jp]
                
                # V: B[b*2+jp, 2*g+kp]
                for jp in range(2):
                    for kp in range(2):
                        b_idx = (b * 2 + jp) * 8 + (2 * g + kp)
                        V[b_idx][r] = Y_forms[s][jp * 2 + kp]
                
                # W: C[2*g+kp, i]
                for i in range(2):
                    for kp in range(2):
                        c_idx = (2 * g + kp) * 2 + i
                        W[c_idx][r] = Z_combos[i * 2 + kp][s]
                
                r += 1
                
    return {"U": U, "V": V, "W": W}

if __name__ == '__main__':
    res = construct()
    U, V, W = res['U'], res['V'], res['W']
    print(f"Rank: {len(U[0])}")
