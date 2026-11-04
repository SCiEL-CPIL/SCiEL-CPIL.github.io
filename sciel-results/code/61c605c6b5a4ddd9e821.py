import numpy as np

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    # We will build U, V, W as lists of lists or numpy arrays
    # Rank is 39
    rank = 39
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_offset = 0
    
    # Helper to add a sub-block decomposition
    # sub_n, sub_m, sub_p
    # row_offset_U, row_offset_V, row_offset_W
    # factors: list of (U_sub, V_sub, W_sub)
    
    def add_block(sub_n, sub_m, sub_p, r_off_u, r_off_v, r_off_w, factors):
        nonlocal col_offset
        for U_s, V_s, W_s in factors:
            # U_s is sub_n * sub_m
            for i in range(sub_n):
                for j in range(sub_m):
                    U[r_off_u + i * sub_m + j][col_offset] += U_s[i][j]
            # V_s is sub_m * sub_p
            for j in range(sub_m):
                for k in range(sub_p):
                    V[r_off_v + j * sub_p + k][col_offset] += V_s[j][k]
            # W_s is sub_p * sub_n
            for k in range(sub_p):
                for i in range(sub_n):
                    W[r_off_w + k * sub_n + i][col_offset] += W_s[k][i]
            col_offset += 1

    # Generate factors for 2x2 by 2x2 (Strassen)
    def get_strassen_2x2():
        # Returns list of 7 (U, V, W) for 2x2 by 2x2
        # U, V, W are 4x1 matrices effectively (vectors)
        factors = []
        # 1. (A00+A11)(B00+B11) -> C00+C11
        U1 = [[1, 0], [0, 1]] # A00, A11
        V1 = [[1, 0], [0, 1]] # B00, B11
        W1 = [[1, 0], [0, 1]] # C00, C11 (W index k*2+i)
        # W[0,0]=1 (k=0,i=0), W[1,1]=1 (k=1,i=1)
        # My W representation in add_block expects W_s[k][i]
        # So W1[0][0]=1, W1[1][1]=1.
        factors.append((U1, V1, W1))
        
        # 2. (A10+A11)B00 -> C10
        U2 = [[0, 0], [1, 1]]
        V2 = [[1, 0], [0, 0]]
        W2 = [[0, 1], [0, 0]] # C10 is k=0, i=1
        factors.append((U2, V2, W2))
        
        # 3. A00(B01-B11) -> C01
        U3 = [[1, 0], [0, 0]]
        V3 = [[0, 1], [-1, 0]]
        W3 = [[0, 0], [1, 0]] # C01 is k=1, i=0
        factors.append((U3, V3, W3))
        
        # 4. A11(B10-B00) -> C10
        U4 = [[0, 0], [0, 1]]
        V4 = [[-1, 0], [1, 0]]
        W4 = [[0, 1], [0, 0]]
        factors.append((U4, V4, W4))
        
        # 5. (A00+A01)B11 -> C01
        U5 = [[1, 1], [0, 0]]
        V5 = [[0, 0], [0, 1]]
        W5 = [[0, 0], [1, 0]]
        factors.append((U5, V5, W5))
        
        # 6. (A10-A00)(B00+B01) -> C11
        U6 = [[-1, 0], [1, 0]]
        V6 = [[1, 1], [0, 0]]
        W6 = [[0, 0], [0, 1]] # C11 is k=1, i=1
        factors.append((U6, V6, W6))
        
        # 7. (A01-A11)(B10+B11) -> C00
        U7 = [[0, 1], [-1, 0]] # A01=1, A11=-1? No, A01 - A11.
        # U[i][j] is coeff of A[i][j].
        # A01 is i=0, j=1. A11 is i=1, j=1.
        U7 = [[0, 1], [0, -1]]
        V7 = [[0, 0], [1, 1]]
        W7 = [[1, 0], [0, 0]] # C00
        factors.append((U7, V7, W7))
        
        return factors

    # Generate factors for 2x2 by 2x1 (Rank 4)
    def get_2x2_2x1():
        factors = []
        # C00 = A00 B00 + A01 B10
        # C10 = A10 B00 + A11 B10
        # 1. A00 B00 -> C00
        U1 = [[1, 0], [0, 0]]
        V1 = [[1, 0], [0, 0]]
        W1 = [[1, 0], [0, 0]] # C00
        factors.append((U1, V1, W1))
        
        # 2. A01 B10 -> C00
        U2 = [[0, 1], [0, 0]]
        V2 = [[0, 0], [1, 0]]
        W2 = [[1, 0], [0, 0]]
        factors.append((U2, V2, W2))
        
        # 3. A10 B00 -> C10
        U3 = [[0, 0], [1, 0]]
        V3 = [[1, 0], [0, 0]]
        W3 = [[0, 1], [0, 0]] # C10
        factors.append((U3, V3, W3))
        
        # 4. A11 B10 -> C10
        U4 = [[0, 0], [0, 1]]
        V4 = [[0, 0], [1, 0]]
        W4 = [[0, 1], [0, 0]]
        factors.append((U4, V4, W4))
        
        return factors

    # Generate factors for 2x1 by 1x7 (Rank 14)
    def get_2x1_1x7():
        factors = []
        # C0k = A00 B0k
        # C1k = A10 B0k
        for k in range(7):
            # A00 B0k -> C0k
            U1 = [[1], [0]]
            V1 = [[0]*7]
            V1[0][k] = 1
            W1 = [[0]*2 for _ in range(7)]
            W1[k][0] = 1
            factors.append((U1, V1, W1))
            
            # A10 B0k -> C1k
            U2 = [[0], [1]]
            V2 = [[0]*7]
            V2[0][k] = 1
            W2 = [[0]*2 for _ in range(7)]
            W2[k][1] = 1
            factors.append((U2, V2, W2))
        return factors

    # Block 1: 2x2 by 2x7
    # Split 2x7 into 2x2, 2x2, 2x3
    # 2x3 into 2x2, 2x1
    
    # Sub-block 1.1: 2x2 by 2x2 (cols 0,1)
    f11 = get_strassen_2x2()
    add_block(2, 2, 2, 0, 0, 0, f11)
    
    # Sub-block 1.2: 2x2 by 2x2 (cols 2,3)
    f12 = get_strassen_2x2()
    add_block(2, 2, 2, 0, 2*2, 2*2, f12) # V offset 2*2=4, W offset 2*2=4
    
    # Sub-block 1.3: 2x2 by 2x2 (cols 4,5)
    f13 = get_strassen_2x2()
    add_block(2, 2, 2, 0, 4*2, 4*2, f13) # V offset 8, W offset 8
    
    # Sub-block 1.4: 2x2 by 2x1 (col 6)
    f14 = get_2x2_2x1()
    add_block(2, 2, 1, 0, 6*2, 6*2, f14) # V offset 12, W offset 12
    
    # Block 2: 2x1 by 1x7
    # A col 2, B row 2
    f2 = get_2x1_1x7()
    add_block(2, 1, 7, 2*2, 2*7, 0, f2) # U offset 4, V offset 14, W offset 0
    
    return {"U": U, "V": V, "W": W}
