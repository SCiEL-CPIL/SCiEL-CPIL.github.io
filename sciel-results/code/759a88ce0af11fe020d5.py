import time

def construct():
    n, m, p = 2, 3, 3
    rank = 17
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a term
    # u_idx: index in U (i*m + j)
    # v_idx: index in V (j*p + k)
    # w_idx: index in W (k*n + i)
    # coeff: coefficient for this term in the decomposition
    def add_term(u_idx, v_idx, w_idx, coeff, col):
        U[u_idx][col] += coeff
        V[v_idx][col] += coeff
        W[w_idx][col] += coeff

    # Strassen for 2x2 core (indices 0,1 for i,j,k)
    # A indices in U: 0(0,0), 1(0,1), 3(1,0), 4(1,1)
    # B indices in V: 0(0,0), 1(0,1), 3(1,0), 4(1,1)
    # C indices in W: 0(0,0), 1(0,1), 2(1,0), 3(1,1)
    
    # M1 = (A00 + A11)(B00 + B11)
    # U: 0, 4; V: 0, 4; W: 0, 1, 2, 3
    # C00 += M1, C01 += M1, C10 += M1, C11 += M1
    add_term(0, 0, 0, 1, 0) # A00 B00
    add_term(0, 4, 0, 1, 0) # A00 B11
    add_term(4, 0, 0, 1, 0) # A11 B00
    add_term(4, 4, 0, 1, 0) # A11 B11
    # Wait, Strassen is 1 product per term.
    # My add_term adds a single rank-1 term.
    # Strassen M1 is ONE product: (A00+A11)*(B00+B11).
    # This expands to 4 terms.
    # But in the tensor decomposition, M1 is a single column in U, V, W.
    # U[:, 0] should be the vector for (A00+A11).
    # V[:, 0] should be the vector for (B00+B11).
    # W[:, 0] should be the vector for how M1 contributes to C.
    
    # Let's re-initialize and fill columns directly.
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def set_col(col, u_vec, v_vec, w_vec):
        for i in range(len(u_vec)):
            U[i][col] = u_vec[i]
        for i in range(len(v_vec)):
            V[i][col] = v_vec[i]
        for i in range(len(w_vec)):
            W[i][col] = w_vec[i]

    # Strassen M1: (A00+A11)(B00+B11)
    # u: [1,0,0,1,0,0] (A00+A11)
    # v: [1,0,0,1,0,0,0,0,0] (B00+B11)
    # w: M1 contributes to C00, C01, C10, C11?
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    # So M1 contributes +1 to C00, +1 to C11.
    # w: [1, 0, 0, 1, 0, 0] (C00, C01, C10, C11, C02, C12)
    set_col(0, [1,0,0,1,0,0], [1,0,0,1,0,0,0,0,0], [1,0,0,1,0,0])
    
    # Strassen M2: (A10+A11)B00
    # u: [0,0,0,1,1,0]
    # v: [1,0,0,0,0,0,0,0,0]
    # w: M2 contributes -1 to C11, +1 to C10.
    # C10 = M2 + M4 -> +1
    # C11 = M1 - M2 + ... -> -1
    # w: [0, 0, 1, -1, 0, 0]
    set_col(1, [0,0,0,1,1,0], [1,0,0,0,0,0,0,0,0], [0,0,1,-1,0,0])
    
    # Strassen M3: A00(B01-B11)
    # u: [1,0,0,0,0,0]
    # v: [0,1,0,0,-1,0,0,0,0]
    # w: M3 contributes +1 to C01, +1 to C11.
    # C01 = M3 + M5 -> +1
    # C11 = ... + M3 + ... -> +1
    # w: [0, 1, 0, 1, 0, 0]
    set_col(2, [1,0,0,0,0,0], [0,1,0,0,-1,0,0,0,0], [0,1,0,1,0,0])
    
    # Strassen M4: A11(B10-B00)
    # u: [0,0,0,0,1,0]
    # v: [0,0,0,1,-1,0,0,0,0]
    # w: M4 contributes +1 to C00, +1 to C10.
    # C00 = M1 + M4 ... -> +1
    # C10 = M2 + M4 -> +1
    # w: [1, 0, 1, 0, 0, 0]
    set_col(3, [0,0,0,0,1,0], [0,0,0,1,-1,0,0,0,0], [1,0,1,0,0,0])
    
    # Strassen M5: (A00+A01)B11
    # u: [1,1,0,0,0,0]
    # v: [0,0,0,0,1,0,0,0,0]
    # w: M5 contributes -1 to C00, +1 to C01.
    # C00 = ... - M5 ... -> -1
    # C01 = ... + M5 -> +1
    # w: [-1, 1, 0, 0, 0, 0]
    set_col(4, [1,1,0,0,0,0], [0,0,0,0,1,0,0,0,0], [-1,1,0,0,0,0])
    
    # Strassen M6: (A10-A00)(B00+B01)
    # u: [0,0,0,1,-1,0]
    # v: [1,1,0,0,0,0,0,0,0]
    # w: M6 contributes +1 to C11.
    # C11 = ... + M6 -> +1
    # w: [0, 0, 0, 1, 0, 0]
    set_col(5, [0,0,0,1,-1,0], [1,1,0,0,0,0,0,0,0], [0,0,0,1,0,0])
    
    # Strassen M7: (A01-A11)(B10+B11)
    # u: [0,1,0,0,-1,0]
    # v: [0,0,0,1,1,0,0,0,0]
    # w: M7 contributes +1 to C00.
    # C00 = ... + M7 -> +1
    # w: [1, 0, 0, 0, 0, 0]
    set_col(6, [0,1,0,0,-1,0], [0,0,0,1,1,0,0,0,0], [1,0,0,0,0,0])
    
    # Remaining 10 terms
    # G2: a_{i2}b_{2k}
    # 7: a02 b20 -> C00. U[2], V[6], W[0]
    set_col(7, [0,0,1,0,0,0], [0,0,0,0,0,0,1,0,0], [1,0,0,0,0,0])
    # 8: a02 b21 -> C01. U[2], V[7], W[1]
    set_col(8, [0,0,1,0,0,0], [0,0,0,0,0,0,0,1,0], [0,1,0,0,0,0])
    # 9: a12 b20 -> C10. U[5], V[6], W[2]
    set_col(9, [0,0,0,0,0,1], [0,0,0,0,0,0,1,0,0], [0,0,1,0,0,0])
    # 10: a12 b21 -> C11. U[5], V[7], W[3]
    set_col(10, [0,0,0,0,0,1], [0,0,0,0,0,0,0,1,0], [0,0,0,1,0,0])
    
    # G3: a_{ij}b_{j2}
    # 11: a00 b02 -> C02. U[0], V[2], W[4]
    set_col(11, [1,0,0,0,0,0], [0,0,1,0,0,0,0,0,0], [0,0,0,0,1,0])
    # 12: a01 b12 -> C02. U[1], V[5], W[4]
    set_col(12, [0,1,0,0,0,0], [0,0,0,0,0,1,0,0,0], [0,0,0,0,1,0])
    # 13: a10 b02 -> C12. U[3], V[2], W[5]
    set_col(13, [0,0,0,1,0,0], [0,0,1,0,0,0,0,0,0], [0,0,0,0,0,1])
    # 14: a11 b12 -> C12. U[4], V[5], W[5]
    set_col(14, [0,0,0,0,1,0], [0,0,0,0,0,1,0,0,0], [0,0,0,0,0,1])
    
    # G4: a_{i2}b_{22}
    # 15: a02 b22 -> C02. U[2], V[8], W[4]
    set_col(15, [0,0,1,0,0,0], [0,0,0,0,0,0,0,0,1], [0,0,0,0,1,0])
    # 16: a12 b22 -> C12. U[5], V[8], W[5]
    set_col(16, [0,0,0,0,0,1], [0,0,0,0,0,0,0,0,1], [0,0,0,0,0,1])
    
    return {"U": U, "V": V, "W": W}
