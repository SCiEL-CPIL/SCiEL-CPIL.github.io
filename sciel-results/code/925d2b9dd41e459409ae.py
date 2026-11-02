import numpy as np

DIMENSIONS = [2, 2, 11]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Base case k=1: Trivial
    # A (2x2) * B (2x1) = C (2x1)
    # C00 = A00 B00 + A01 B10
    # C10 = A10 B00 + A11 B10
    # Terms:
    # 1: A00 B00 -> C00
    # 2: A01 B10 -> C00
    # 3: A10 B00 -> C10
    # 4: A11 B10 -> C10
    
    def get_base_1():
        # U: (n*m, rank) = (4, 4)
        # V: (m*p, rank) = (2, 4)  (p=1)
        # W: (p*n, rank) = (2, 4)  (p=1)
        U = [[1,0,0,0],
             [0,1,0,0],
             [0,0,1,0],
             [0,0,0,1]]
        # V rows: B00 (idx 0), B10 (idx 1)
        V = [[1,0,1,0],
             [0,1,0,1]]
        # W rows: C00 (idx 0), C10 (idx 1)
        W = [[1,1,0,0],
             [0,0,1,1]]
        return U, V, W

    # Base case k=2: Strassen
    # A (2x2) * B (2x2) = C (2x2)
    # Strassen 7 multiplications
    def get_base_2():
        # U: (4, 7)
        # V: (4, 7)  (p=2, m=2 -> 4 rows)
        # W: (4, 7)  (p=2, n=2 -> 4 rows)
        
        # m1 = (A00+A11)(B00+B11)
        # m2 = (A10+A11)B00
        # m3 = A00(B01-B11)
        # m4 = A11(B10-B00)
        # m5 = (A00+A01)B11
        # m6 = (A10-A00)(B00+B01)
        # m7 = (A01-A11)(B10+B11)
        
        # U columns (coefficients for A)
        # A indices: 00=0, 01=1, 10=2, 11=3
        u1 = [1, 0, 0, 1]
        u2 = [0, 0, 1, 1]
        u3 = [1, 0, 0, 0]
        u4 = [0, 0, 0, 1]
        u5 = [1, 1, 0, 0]
        u6 = [-1, 0, 1, 0]
        u7 = [0, 1, 0, -1]
        
        U = [u1, u2, u3, u4, u5, u6, u7]
        
        # V columns (coefficients for B)
        # B indices: 00=0, 01=1, 10=2, 11=3
        v1 = [1, 0, 0, 1]
        v2 = [1, 0, 0, 0]
        v3 = [0, 1, 0, -1]
        v4 = [0, 0, 1, -1]
        v5 = [0, 0, 0, 1]
        v6 = [1, 1, 0, 0]
        v7 = [0, 0, 1, 1]
        
        V = [v1, v2, v3, v4, v5, v6, v7]
        
        # W columns (coefficients for C)
        # C indices: 00=0, 01=1, 10=2, 11=3
        # C00 = m1 + m4 - m5 + m7
        # C01 = m3 + m5
        # C10 = m2 + m4
        # C11 = m1 - m2 + m3 + m6
        
        w1 = [1, 0, 0, 1]
        w2 = [0, 0, 1, -1]
        w3 = [0, 1, 0, 1]
        w4 = [1, 0, 1, 0]
        w5 = [-1, 1, 0, 0]
        w6 = [0, 0, 0, 1]
        w7 = [1, 0, 0, 0]
        
        W = [w1, w2, w3, w4, w5, w6, w7]
        
        return U, V, W

    def decompose(k):
        if k == 1:
            return get_base_1()
        if k == 2:
            return get_base_2()
        
        # Split k into k1, k2
        # Greedy: try to balance or use known optimal splits
        # For now, use k1 = k // 2, k2 = k - k1
        # But we want to minimize rank.
        # R(3) = 11 (2+1)
        # R(4) = 14 (2+2)
        # R(5) = 18 (2+3)
        # R(6) = 21 (2+4)
        # R(7) = 25 (3+4) or 2+5=25?
        # Let's just use k1 = k // 2
        
        k1 = k // 2
        k2 = k - k1
        
        U1, V1, W1 = decompose(k1)
        U2, V2, W2 = decompose(k2)
        
        # Concatenate
        # U: same A, so just stack columns
        U = U1 + U2
        
        # V: shift V2 rows by 2*k1
        # V1 has shape (2*k1, r1)
        # V2 has shape (2*k2, r2)
        # We need to create a new V2_shifted
        r1 = len(V1[0])
        r2 = len(V2[0])
        
        V2_shifted = [[0]*r2 for _ in range(2*k2)]
        for i in range(2*k2):
            for j in range(r2):
                V2_shifted[i][j] = V2[i][j]
        
        # Actually, we just need to place V2 in the correct rows of the final V
        # Final V has 2*k rows.
        # Rows 0..2*k1-1 come from V1.
        # Rows 2*k1..2*k-1 come from V2.
        
        # But we are concatenating columns.
        # The final V will have r1+r2 columns.
        # For columns 0..r1-1, V1 provides values for rows 0..2*k1-1, and 0 for others.
        # For columns r1..r1+r2-1, V2 provides values for rows 2*k1..2*k-1, and 0 for others.
        
        # It is easier to construct the final matrices directly.
        
        rank = r1 + r2
        U_final = [[0]*rank for _ in range(4)]
        V_final = [[0]*rank for _ in range(2*k)]
        W_final = [[0]*rank for _ in range(2*k)]
        
        # Fill from Part 1
        for i in range(4):
            for j in range(r1):
                U_final[i][j] = U1[i][j]
        
        for i in range(2*k1):
            for j in range(r1):
                V_final[i][j] = V1[i][j]
                
        for i in range(2*k1):
            for j in range(r1):
                W_final[i][j] = W1[i][j]
                
        # Fill from Part 2
        offset = r1
        for i in range(4):
            for j in range(r2):
                U_final[i][offset+j] = U2[i][j]
                
        for i in range(2*k2):
            for j in range(r2):
                V_final[2*k1 + i][offset+j] = V2[i][j]
                
        for i in range(2*k2):
            for j in range(r2):
                W_final[2*k1 + i][offset+j] = W2[i][j]
                
        return U_final, V_final, W_final

    U, V, W = decompose(p)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
