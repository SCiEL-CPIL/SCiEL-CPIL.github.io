import numpy as np

def construct():
    n, m, p = 4, 4, 4
    rank = 49
    
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    # 2x2 Strassen coefficients
    # U2[i][k] is coeff of A element k for term i
    # V2[i][k] is coeff of B element k for term i
    # W2[i][k] is coeff of C element k for term i
    # Element indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    # Term 0: M1 = (A00+A11)(B00+B11) -> C00+=, C11+=
    # Term 1: M2 = (A10+A11)B00 -> C10+=, C11-=
    # Term 2: M3 = A00(B01-B11) -> C01+=, C11+=
    # Term 3: M4 = A11(B10-B00) -> C00+=, C10+=
    # Term 4: M5 = (A00+A01)B11 -> C00-=, C01+=
    # Term 5: M6 = (A10-A00)(B00+B01) -> C11+=
    # Term 6: M7 = (A01-A11)(B10+B11) -> C00+=
    
    U2 = np.zeros((7, 4))
    V2 = np.zeros((7, 4))
    W2 = np.zeros((7, 4))
    
    # T0
    U2[0, 0] = 1; U2[0, 3] = 1
    V2[0, 0] = 1; V2[0, 3] = 1
    W2[0, 0] = 1; W2[0, 3] = 1
    
    # T1
    U2[1, 2] = 1; U2[1, 3] = 1
    V2[1, 0] = 1
    W2[1, 2] = 1; W2[1, 3] = -1
    
    # T2
    U2[2, 0] = 1
    V2[2, 1] = 1; V2[2, 3] = -1
    W2[2, 1] = 1; W2[2, 3] = 1
    
    # T3
    U2[3, 3] = 1
    V2[3, 2] = 1; V2[3, 0] = -1
    W2[3, 0] = 1; W2[3, 2] = 1
    
    # T4
    U2[4, 0] = 1; U2[4, 1] = 1
    V2[4, 3] = 1
    W2[4, 0] = -1; W2[4, 1] = 1
    
    # T5
    U2[5, 2] = 1; U2[5, 0] = -1
    V2[5, 0] = 1; V2[5, 1] = 1
    W2[5, 3] = 1
    
    # T6
    U2[6, 1] = 1; U2[6, 3] = -1
    V2[6, 2] = 1; V2[6, 3] = 1
    W2[6, 0] = 1

    for ti in range(7):
        for tj in range(7):
            rank_idx = ti * 7 + tj
            for k1 in range(4):
                for k2 in range(4):
                    coeff_u = U2[ti, k1] * U2[tj, k2]
                    coeff_v = V2[ti, k1] * V2[tj, k2]
                    coeff_w = W2[ti, k1] * W2[tj, k2]
                    
                    if coeff_u == 0 and coeff_v == 0 and coeff_w == 0:
                        continue
                        
                    # Map k1, k2 to 4x4 indices
                    # k1 = br * 2 + bc
                    # k2 = ir * 2 + ic
                    br = k1 // 2
                    bc = k1 % 2
                    ir = k2 // 2
                    ic = k2 % 2
                    
                    # 4x4 Row = br * 2 + ir
                    # 4x4 Col = bc * 2 + ic
                    r = br * 2 + ir
                    c = bc * 2 + ic
                    
                    # U, V are row-major: idx = r * 4 + c
                    idx_uv = r * 4 + c
                    U[idx_uv, rank_idx] += coeff_u
                    V[idx_uv, rank_idx] += coeff_v
                    
                    # W is column-major (trace convention): idx = c * 4 + r
                    idx_w = c * 4 + r
                    W[idx_w, rank_idx] += coeff_w

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
