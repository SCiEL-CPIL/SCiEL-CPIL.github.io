"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We will build a rank-minimal decomposition using Kronecker/block structure.
    # Split m=5 into m1=2 and m2=3.
    # Split p=6 into p1=2, p2=2, p3=2.
    #
    # For n=2, m=2, p=2: Strassen gives rank 7.
    # For n=2, m=1, p=2: rank 4 (schoolbook, since R(2,1,2)=4).
    #
    # We need to handle:
    # 1. i in [0,1], j in [0,1], k in [0,1] -> 2x2x2 -> rank 7
    # 2. i in [0,1], j in [0,1], k in [2,3] -> 2x2x2 -> rank 7
    # 3. i in [0,1], j in [0,1], k in [4,5] -> 2x2x2 -> rank 7
    # 4. i in [0,1], j in [2,4], k in [0,1] -> 2x3x2 -> need to split j into 1+2
    #    4a. i in [0,1], j=2, k in [0,1] -> 2x1x2 -> rank 4
    #    4b. i in [0,1], j in [3,4], k in [0,1] -> 2x2x2 -> rank 7
    # 5. i in [0,1], j in [2,4], k in [2,3] -> 2x3x2
    #    5a. i in [0,1], j=2, k in [2,3] -> 2x1x2 -> rank 4
    #    5b. i in [0,1], j in [3,4], k in [2,3] -> 2x2x2 -> rank 7
    # 6. i in [0,1], j in [2,4], k in [4,5] -> 2x3x2
    #    6a. i in [0,1], j=2, k in [4,5] -> 2x1x2 -> rank 4
    #    6b. i in [0,1], j in [3,4], k in [4,5] -> 2x2x2 -> rank 7
    #
    # Total rank = 7*3 + 4 + 7 + 4 + 7 + 4 + 7 = 21 + 4 + 7 + 4 + 7 + 4 + 7 = 54.
    
    rank = 54
    
    U = [[0] * rank for _ in range(n * m)]  # shape (10, 54)
    V = [[0] * rank for _ in range(m * p)]  # shape (30, 54)
    W = [[0] * rank for _ in range(p * n)]  # shape (12, 54)
    
    col = 0
    
    def add_term(u_map, v_map, w_map):
        nonlocal col
        for idx, coeff in u_map.items():
            U[idx][col] += coeff
        for idx, coeff in v_map.items():
            V[idx][col] += coeff
        for idx, coeff in w_map.items():
            W[idx][col] += coeff
        col += 1
    
    # Strassen decomposition for 2x2x2 block
    # i_off: offset for i (0 or 1)
    # j_off: offset for j (0, 1, 2, 3, or 4)
    # k_off: offset for k (0, 2, or 4)
    def strassen_2x2x2(i_off, j_off, k_off):
        # Local indices: i in [0,1], j in [0,1], k in [0,1]
        # U index: (i_off + i) * m + (j_off + j)
        # V index: (j_off + j) * p + (k_off + k)
        # W index: (k_off + k) * n + (i_off + i)
        
        # M1 = (A00 + A11)(B00 + B11)
        # U: A00: (i_off)*m + (j_off), A11: (i_off+1)*m + (j_off+1)
        # V: B00: j_off*p + k_off, B11: (j_off+1)*p + (k_off+1)
        # W: C00: k_off*n + i_off, C11: (k_off+1)*n + (i_off+1)
        # C00 = M1 + M4 - M5 + M7
        # C11 = M1 - M2 + M3 + M6
        
        u1 = {(i_off)*m + (j_off): 1, (i_off+1)*m + (j_off+1): 1}
        v1 = {(j_off)*p + (k_off): 1, (j_off+1)*p + (k_off+1): 1}
        w1 = {(k_off)*n + (i_off): 1, (k_off+1)*n + (i_off+1): 1}
        add_term(u1, v1, w1)
        
        # M2 = (A10 + A11)B00
        # U: A10: (i_off+1)*m + j_off, A11: (i_off+1)*m + (j_off+1)
        # V: B00: j_off*p + k_off
        # W: C10: k_off*n + (i_off+1), C11: (k_off+1)*n + (i_off+1)
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        u2 = {(i_off+1)*m + (j_off): 1, (i_off+1)*m + (j_off+1): 1}
        v2 = {(j_off)*p + (k_off): 1}
        w2 = {(k_off)*n + (i_off+1): 1, (k_off+1)*n + (i_off+1): -1}
        add_term(u2, v2, w2)
        
        # M3 = A00(B10 - B11)
        # U: A00: i_off*m + j_off
        # V: B10: (j_off+1)*p + k_off, B11: (j_off+1)*p + (k_off+1)
        # W: C01: (k_off+1)*n + i_off, C11: (k_off+1)*n + (i_off+1)
        # C01 = M3 + M5
        # C11 = M1 - M2 + M3 + M6
        
        u3 = {(i_off)*m + (j_off): 1}
        v3 = {(j_off+1)*p + (k_off): 1, (j_off+1)*p + (k_off+1): -1}
        w3 = {(k_off+1)*n + (i_off): 1, (k_off+1)*n + (i_off+1): 1}
        add_term(u3, v3, w3)
        
        # M4 = A11(B01 - B00)
        # U: A11: (i_off+1)*m + (j_off+1)
        # V: B01: j_off*p + (k_off+1), B00: j_off*p + k_off
        # W: C00: k_off*n + i_off, C10: k_off*n + (i_off+1)
        # C00 = M1 + M4 - M5 + M7
        # C10 = M2 + M4
        
        u4 = {(i_off+1)*m + (j_off+1): 1}
        v4 = {(j_off)*p + (k_off+1): 1, (j_off)*p + (k_off): -1}
        w4 = {(k_off)*n + (i_off): 1, (k_off)*n + (i_off+1): 1}
        add_term(u4, v4, w4)
        
        # M5 = (A00 + A01)B11
        # U: A00: i_off*m + j_off, A01: i_off*m + (j_off+1)
        # V: B11: (j_off+1)*p + (k_off+1)
        # W: C00: k_off*n + i_off, C01: (k_off+1)*n + i_off
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        
        u5 = {(i_off)*m + (j_off): 1, (i_off)*m + (j_off+1): 1}
        v5 = {(j_off+1)*p + (k_off+1): 1}
        w5 = {(k_off)*n + (i_off): -1, (k_off+1)*n + (i_off): 1}
        add_term(u5, v5, w5)
        
        # M6 = (A10 - A00)(B00 + B01)
        # U: A10: (i_off+1)*m + j_off, A00: i_off*m + j_off
        # V: B00: j_off*p + k_off, B01: j_off*p + (k_off+1)
        # W: C11: (k_off+1)*n + (i_off+1)
        # C11 = M1 - M2 + M3 + M6
        
        u6 = {(i_off+1)*m + (j_off): 1, (i_off)*m + (j_off): -1}
        v6 = {(j_off)*p + (k_off): 1, (j_off)*p + (k_off+1): 1}
        w6 = {(k_off+1)*n + (i_off+1): 1}
        add_term(u6, v6, w6)
        
        # M7 = (A01 - A11)(B10 + B11)
        # U: A01: i_off*m + (j_off+1), A11: (i_off+1)*m + (j_off+1)
        # V: B10: (j_off+1)*p + k_off, B11: (j_off+1)*p + (k_off+1)
        # W: C00: k_off*n + i_off
        # C00 = M1 + M4 - M5 + M7
        
        u7 = {(i_off)*m + (j_off+1): 1, (i_off+1)*m + (j_off+1): -1}
        v7 = {(j_off+1)*p + (k_off): 1, (j_off+1)*p + (k_off+1): 1}
        w7 = {(k_off)*n + (i_off): 1}
        add_term(u7, v7, w7)
    
    # Rank 4 decomposition for 2x1x2 block
    # i in [0,1], j fixed, k in [0,1]
    def direct_2x1x2(i_off, j_fixed, k_off):
        # C_ik = A_i0 B_0k
        # 4 terms: (i=0,k=0), (i=0,k=1), (i=1,k=0), (i=1,k=1)
        
        # Term 1: i=0, k=0
        u = {(i_off)*m + j_fixed: 1}
        v = {j_fixed*p + (k_off): 1}
        w = {(k_off)*n + (i_off): 1}
        add_term(u, v, w)
        
        # Term 2: i=0, k=1
        u = {(i_off)*m + j_fixed: 1}
        v = {j_fixed*p + (k_off+1): 1}
        w = {(k_off+1)*n + (i_off): 1}
        add_term(u, v, w)
        
        # Term 3: i=1, k=0
        u = {(i_off+1)*m + j_fixed: 1}
        v = {j_fixed*p + (k_off): 1}
        w = {(k_off)*n + (i_off+1): 1}
        add_term(u, v, w)
        
        # Term 4: i=1, k=1
        u = {(i_off+1)*m + j_fixed: 1}
        v = {j_fixed*p + (k_off+1): 1}
        w = {(k_off+1)*n + (i_off+1): 1}
        add_term(u, v, w)
    
    # Apply decompositions:
    
    # Block 1: i in [0,1], j in [0,1], k in [0,1]
    strassen_2x2x2(0, 0, 0)
    
    # Block 2: i in [0,1], j in [0,1], k in [2,3]
    strassen_2x2x2(0, 0, 2)
    
    # Block 3: i in [0,1], j in [0,1], k in [4,5]
    strassen_2x2x2(0, 0, 4)
    
    # Block 4a: i in [0,1], j=2, k in [0,1]
    direct_2x1x2(0, 2, 0)
    
    # Block 4b: i in [0,1], j in [3,4], k in [0,1]
    strassen_2x2x2(0, 3, 0)
    
    # Block 5a: i in [0,1], j=2, k in [2,3]
    direct_2x1x2(0, 2, 2)
    
    # Block 5b: i in [0,1], j in [3,4], k in [2,3]
    strassen_2x2x2(0, 3, 2)
    
    # Block 6a: i in [0,1], j=2, k in [4,5]
    direct_2x1x2(0, 2, 4)
    
    # Block 6b: i in [0,1], j in [3,4], k in [4,5]
    strassen_2x2x2(0, 3, 4)
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
