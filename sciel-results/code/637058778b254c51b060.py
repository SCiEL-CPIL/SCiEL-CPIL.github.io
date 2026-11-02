import numpy as np

def construct():
    n, m, p = 3, 3, 8
    nm, mp, pn = n*m, m*p, p*n
    
    U = [[0.0] * 68 for _ in range(nm)]
    V = [[0.0] * 68 for _ in range(mp)]
    W = [[0.0] * 68 for _ in range(pn)]
    
    col = 0
    
    def add_strassen_224(a_indices, b_indices, c_indices):
        """
        Adds a Strassen-like decomposition for 2x2 A, 2x4 B -> 2x4 C.
        a_indices: [a00, a01, a10, a11]
        b_indices: [b00, b01, b02, b03, b10, b11, b12, b13]
        c_indices: [c00, c01, c02, c03, c10, c11, c12, c13]
        """
        nonlocal col
        a00, a01, a10, a11 = a_indices
        b00, b01, b02, b03, b10, b11, b12, b13 = b_indices
        c00, c01, c02, c03, c10, c11, c12, c13 = c_indices
        
        # Split B into B_L (cols 0-1) and B_R (cols 2-3)
        # B_L indices: b00, b01, b10, b11
        # B_R indices: b02, b03, b12, b13
        
        # Strassen for B_L
        # m1 = (a00+a11)(b00+b11)
        U[a00][col] += 1; U[a11][col] += 1
        V[b00][col] += 1; V[b11][col] += 1
        W[c00][col] += 1; W[c11][col] += 1
        W[c01][col] -= 1; W[c10][col] += 1 # Wait, Strassen formulas
        # Let's use standard Strassen coefficients
        # C00 = m1 + m3 - m5 + m7
        # C01 = m4 + m5
        # C10 = m2 + m7
        # C11 = m1 - m2 + m4 + m6
        
        # m1
        col += 1
        # m2 = (a10+a11)b00
        U[a10][col] += 1; U[a11][col] += 1
        V[b00][col] += 1
        W[c10][col] += 1; W[c11][col] -= 1
        col += 1
        # m3 = a00(b01-b11)
        U[a00][col] += 1
        V[b01][col] += 1; V[b11][col] -= 1
        W[c00][col] += 1; W[c01][col] += 1
        col += 1
        # m4 = a11(b10-b00)
        U[a11][col] += 1
        V[b10][col] += 1; V[b00][col] -= 1
        W[c01][col] += 1; W[c11][col] += 1
        col += 1
        # m5 = (a00+a01)b11
        U[a00][col] += 1; U[a01][col] += 1
        V[b11][col] += 1
        W[c00][col] -= 1; W[c01][col] += 1
        col += 1
        # m6 = (a10-a00)(b00+b01)
        U[a10][col] += 1; U[a00][col] -= 1
        V[b00][col] += 1; V[b01][col] += 1
        W[c11][col] += 1
        col += 1
        # m7 = (a01-a11)(b10+b11)
        U[a01][col] += 1; U[a11][col] -= 1
        V[b10][col] += 1; V[b11][col] += 1
        W[c00][col] += 1; W[c10][col] += 1
        col += 1
        
        # Strassen for B_R
        # m1' = (a00+a11)(b02+b12)
        U[a00][col] += 1; U[a11][col] += 1
        V[b02][col] += 1; V[b12][col] += 1
        W[c02][col] += 1; W[c13][col] += 1
        W[c03][col] -= 1; W[c12][col] += 1
        col += 1
        # m2' = (a10+a11)b02
        U[a10][col] += 1; U[a11][col] += 1
        V[b02][col] += 1
        W[c12][col] += 1; W[c13][col] -= 1
        col += 1
        # m3' = a00(b03-b12)
        U[a00][col] += 1
        V[b03][col] += 1; V[b12][col] -= 1
        W[c02][col] += 1; W[c03][col] += 1
        col += 1
        # m4' = a11(b12-b02)
        U[a11][col] += 1
        V[b12][col] += 1; V[b02][col] -= 1
        W[c03][col] += 1; W[c13][col] += 1
        col += 1
        # m5' = (a00+a01)b12
        U[a00][col] += 1; U[a01][col] += 1
        V[b12][col] += 1
        W[c02][col] -= 1; W[c03][col] += 1
        col += 1
        # m6' = (a10-a00)(b02+b03)
        U[a10][col] += 1; U[a00][col] -= 1
        V[b02][col] += 1; V[b03][col] += 1
        W[c13][col] += 1
        col += 1
        # m7' = (a01-a11)(b12+b13)
        U[a01][col] += 1; U[a11][col] -= 1
        V[b12][col] += 1; V[b13][col] += 1
        W[c02][col] += 1; W[c12][col] += 1
        col += 1

    def add_simple_124(a_indices, b_indices, c_indices):
        """
        Adds decomposition for 1x2 A, 2x4 B -> 1x4 C.
        a_indices: [a0, a1]
        b_indices: [b00, b01, b02, b03, b10, b11, b12, b13]
        c_indices: [c0, c1, c2, c3]
        """
        nonlocal col
        a0, a1 = a_indices
        b00, b01, b02, b03, b10, b11, b12, b13 = b_indices
        c0, c1, c2, c3 = c_indices
        
        # c0 = a0*b00 + a1*b10
        U[a0][col] += 1; V[b00][col] += 1; W[c0][col] += 1
        col += 1
        U[a1][col] += 1; V[b10][col] += 1; W[c0][col] += 1
        col += 1
        
        # c1 = a0*b01 + a1*b11
        U[a0][col] += 1; V[b01][col] += 1; W[c1][col] += 1
        col += 1
        U[a1][col] += 1; V[b11][col] += 1; W[c1][col] += 1
        col += 1
        
        # c2 = a0*b02 + a1*b12
        U[a0][col] += 1; V[b02][col] += 1; W[c2][col] += 1
        col += 1
        U[a1][col] += 1; V[b12][col] += 1; W[c2][col] += 1
        col += 1
        
        # c3 = a0*b03 + a1*b13
        U[a0][col] += 1; V[b03][col] += 1; W[c3][col] += 1
        col += 1
        U[a1][col] += 1; V[b13][col] += 1; W[c3][col] += 1
        col += 1

    def add_simple_214(a_indices, b_indices, c_indices):
        """
        Adds decomposition for 2x1 A, 1x4 B -> 2x4 C.
        a_indices: [a0, a1]
        b_indices: [b0, b1, b2, b3]
        c_indices: [c00, c01, c02, c03, c10, c11, c12, c13]
        """
        nonlocal col
        a0, a1 = a_indices
        b0, b1, b2, b3 = b_indices
        c00, c01, c02, c03, c10, c11, c12, c13 = c_indices
        
        # Row 0 of C: a0 * [b0, b1, b2, b3]
        for k in range(4):
            U[a0][col] += 1
            V[b_indices[k]][col] += 1
            W[c_indices[k]][col] += 1
            col += 1
            
        # Row 1 of C: a1 * [b0, b1, b2, b3]
        for k in range(4):
            U[a1][col] += 1
            V[b_indices[k]][col] += 1
            W[c_indices[4+k]][col] += 1
            col += 1

    def add_simple_114(a_indices, b_indices, c_indices):
        """
        Adds decomposition for 1x1 A, 1x4 B -> 1x4 C.
        a_indices: [a0]
        b_indices: [b0, b1, b2, b3]
        c_indices: [c0, c1, c2, c3]
        """
        nonlocal col
        a0 = a_indices[0]
        for k in range(4):
            U[a0][col] += 1
            V[b_indices[k]][col] += 1
            W[c_indices[k]][col] += 1
            col += 1

    # Indices
    # A: 3x3
    # A11: 0,1,3,4
    # A12: 2,5
    # A21: 6,7
    # A22: 8
    
    # B: 3x8
    # B11: 0-3, 8-11
    # B12: 4-7, 12-15
    # B21: 16-19
    # B22: 20-23
    
    # C: 3x8 (c = 3k + i)
    # C11 (i=0,1; k=0-3): 0,1,3,4,6,7,9,10
    # C12 (i=0,1; k=4-7): 12,13,15,16,18,19,21,22
    # C21 (i=2; k=0-3): 2,5,8,11
    # C22 (i=2; k=4-7): 14,17,20,23
    
    # 1. A11 B11 -> C11
    add_strassen_224(
        [0, 1, 3, 4],
        [0, 1, 2, 3, 8, 9, 10, 11],
        [0, 1, 3, 4, 6, 7, 9, 10]
    )
    
    # 2. A11 B12 -> C12
    add_strassen_224(
        [0, 1, 3, 4],
        [4, 5, 6, 7, 12, 13, 14, 15],
        [12, 13, 15, 16, 18, 19, 21, 22]
    )
    
    # 3. A21 B11 -> C21
    add_simple_124(
        [6, 7],
        [0, 1, 2, 3, 8, 9, 10, 11],
        [2, 5, 8, 11]
    )
    
    # 4. A21 B12 -> C22
    add_simple_124(
        [6, 7],
        [4, 5, 6, 7, 12, 13, 14, 15],
        [14, 17, 20, 23]
    )
    
    # 5. A12 B21 -> C11
    add_simple_214(
        [2, 5],
        [16, 17, 18, 19],
        [0, 1, 3, 4, 6, 7, 9, 10]
    )
    
    # 6. A12 B22 -> C12
    add_simple_214(
        [2, 5],
        [20, 21, 22, 23],
        [12, 13, 15, 16, 18, 19, 21, 22]
    )
    
    # 7. A22 B21 -> C21
    add_simple_114(
        [8],
        [16, 17, 18, 19],
        [2, 5, 8, 11]
    )
    
    # 8. A22 B22 -> C22
    add_simple_114(
        [8],
        [20, 21, 22, 23],
        [14, 17, 20, 23]
    )
    
    return {"U": U, "V": V, "W": W}
