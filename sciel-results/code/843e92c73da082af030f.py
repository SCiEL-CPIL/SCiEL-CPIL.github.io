import numpy as np

DIMENSIONS = [2, 2, 11]

def construct():
    n, m, p = DIMENSIONS
    # We will use 5 Strassen blocks (2x2x2) and 1 Schoolbook block (2x2x1)
    # Total rank: 5 * 7 + 1 * 4 = 39
    
    rank = 39
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen coefficients for 2x2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # A indices: 0:00, 1:01, 2:10, 3:11
    # B indices: 0:00, 1:01, 2:10, 3:11 (local)
    # C indices: 0:00, 1:01, 2:10, 3:11 (local)
    
    strassen_U = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [0, 0, 1, -1], # M6
        [0, 1, 0, -1]  # M7
    ]
    
    strassen_V = [
        [1, 1, 0, 0], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1], # M3
        [0, 0, 1, -1], # M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    
    strassen_W = [
        [1, 0, 0, 1], # C00: M1 + M4 - M5 + M7
        [0, 1, 0, 0], # C01: M3 + M5
        [0, 0, 1, 0], # C10: M2 + M4
        [1, 0, 0, 1]  # C11: M1 - M2 + M3 + M6
    ]
    # Wait, W is for the output C.
    # The tensor decomposition sums to the multiplication tensor.
    # The W vectors correspond to the output basis.
    # For Strassen, the output is linear combination of M's.
    # So W_r should be the coefficient of M_r in C_i?
    # No, the decomposition is sum u_r v_r w_r.
    # The term w_r is the vector in C space.
    # If C = sum M_r * c_r, then w_r is the vector c_r?
    # No.
    # The tensor is T.
    # T = sum u_r \otimes v_r \otimes w_r.
    # This means T(i,j,k) = sum u_r(i,j) v_r(j,k) w_r(k,i).
    # For Strassen, we have 7 multiplications M_r.
    # C_i = sum_r coeff(i,r) M_r.
    # So w_r(k,i) should be coeff(i,r) if k matches?
    # Actually, the output index is (k,i).
    # The coefficient of M_r in C_i is coeff(i,r).
    # So w_r(k,i) = coeff(i,r) for the specific k?
    # No, M_r is a scalar.
    # C_i = sum_r coeff(i,r) M_r.
    # So the contribution to C_i from M_r is coeff(i,r).
    # So w_r(k,i) = coeff(i,r) if k is the column?
    # In the local 2x2x2, k is 0 or 1.
    # C_0 = M1 + M4 - M5 + M7
    # C_1 = M3 + M5
    # So for r=1 (M1), w_1(0,0)=1, w_1(0,1)=0, w_1(1,0)=0, w_1(1,1)=1?
    # No, C_00 uses M1. C_11 uses M1.
    # So w_1(0,0)=1, w_1(1,1)=1.
    # My strassen_W above is actually the matrix of coefficients for C rows?
    # Let's check.
    # C_00 = M1 + M4 - M5 + M7.
    # So w_1(0,0)=1, w_4(0,0)=1, w_5(0,0)=-1, w_7(0,0)=1.
    # C_01 = M3 + M5.
    # w_3(0,1)=1, w_5(0,1)=1.
    # C_10 = M2 + M4.
    # w_2(1,0)=1, w_4(1,0)=1.
    # C_11 = M1 - M2 + M3 + M6.
    # w_1(1,1)=1, w_2(1,1)=-1, w_3(1,1)=1, w_6(1,1)=1.
    
    # So w_r is a vector of length 4 (flattened C).
    # Index k*n + i.
    # 0:00, 1:01, 2:10, 3:11.
    
    w1 = [1, 0, 0, 1]
    w2 = [0, 0, 1, -1]
    w3 = [0, 1, 1, 1]
    w4 = [1, 0, 1, 0]
    w5 = [-1, 1, 0, 0]
    w6 = [0, 0, 0, 1]
    w7 = [1, 0, 0, 0]
    
    strassen_W = [w1, w2, w3, w4, w5, w6, w7]

    # Process 5 Strassen blocks
    for b in range(5):
        k1 = b * 2
        k2 = b * 2 + 1
        for r in range(7):
            u = strassen_U[r]
            v = strassen_V[r]
            w = strassen_W[r]
            
            # Fill U
            for idx, val in enumerate(u):
                if val != 0:
                    U[idx][col] = val
            
            # Fill V
            # Local v indices: 0:00, 1:01, 2:10, 3:11
            # Global indices: j*11 + k
            # 0 -> 0*11 + k1
            # 1 -> 0*11 + k2
            # 2 -> 1*11 + k1
            # 3 -> 1*11 + k2
            v_indices = [
                0 * p + k1,
                0 * p + k2,
                1 * p + k1,
                1 * p + k2
            ]
            for idx, val in enumerate(v):
                if val != 0:
                    V[v_indices[idx]][col] = val
            
            # Fill W
            # Local w indices: 0:00, 1:01, 2:10, 3:11
            # Global indices: k*n + i
            # 0 -> k1*2 + 0
            # 1 -> k2*2 + 0
            # 2 -> k1*2 + 1
            # 3 -> k2*2 + 1
            w_indices = [
                k1 * n + 0,
                k2 * n + 0,
                k1 * n + 1,
                k2 * n + 1
            ]
            for idx, val in enumerate(w):
                if val != 0:
                    W[w_indices[idx]][col] = val
            
            col += 1
            
    # Process 1 Schoolbook block (last column)
    k1 = 10
    # Schoolbook for 2x2x1
    # C0 = A00 B0 + A01 B1
    # C1 = A10 B0 + A11 B1
    # 4 terms
    sb_U = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]
    sb_V = [
        [1, 0, 0, 0], # B0
        [0, 1, 0, 0], # B1
        [1, 0, 0, 0], # B0
        [0, 1, 0, 0]  # B1
    ]
    sb_W = [
        [1, 0, 0, 0], # C0
        [1, 0, 0, 0], # C0
        [0, 1, 0, 0], # C1
        [0, 1, 0, 0]  # C1
    ]
    # Note: W index is k*n+i.
    # k=10.
    # C0 is (10, 0) -> 10*2 + 0 = 20
    # C1 is (10, 1) -> 10*2 + 1 = 21
    
    for r in range(4):
        u = sb_U[r]
        v = sb_V[r]
        w = sb_W[r]
        
        for idx, val in enumerate(u):
            if val != 0:
                U[idx][col] = val
        
        # V indices for k1
        # Local 0 -> 0*11 + 10
        # Local 1 -> 1*11 + 10
        v_indices = [
            0 * p + k1,
            1 * p + k1
        ]
        # sb_V has 4 entries, but only 2 are used?
        # sb_V[0] is B0. sb_V[1] is B1.
        # The vector v is length 4.
        # v[0] is coeff of B00? No, B is 2x1.
        # B indices: 0:0, 1:1.
        # So v[0] is B0, v[1] is B1.
        # v[2], v[3] are 0.
        for idx, val in enumerate(v):
            if val != 0 and idx < 2:
                V[v_indices[idx]][col] = val
        
        # W indices
        # w[0] is C0. w[1] is C1.
        # w[2], w[3] are 0.
        w_indices = [
            k1 * n + 0,
            k1 * n + 1
        ]
        for idx, val in enumerate(w):
            if val != 0 and idx < 2:
                W[w_indices[idx]][col] = val
        
        col += 1

    return {"U": U, "V": V, "W": W}
