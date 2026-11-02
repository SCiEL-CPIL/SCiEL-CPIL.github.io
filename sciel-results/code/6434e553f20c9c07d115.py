import numpy as np

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We will construct a decomposition of rank 40.
    # Split p=7 into p1=2, p2=2, p3=3.
    # C1 = A B1 (2x2), C2 = A B2 (2x2), C3 = A B3 (2x3).
    # A is 2x3, B1 is 3x2, B2 is 3x2, B3 is 3x3.
    
    # For C1 (2x2):
    # A = [A1 (2x2), A2 (2x1)]
    # B1 = [B11 (2x2), B12 (1x2)]^T  (actually B1 is 3x2, so B11 is 2x2, B12 is 1x2)
    # C1 = A1 B11 + A2 B12
    # A1 B11 is 2x2x2 -> Rank 7 (Strassen)
    # A2 B12 is 2x1x2 -> Rank 4 (Schoolbook)
    # Total 11.
    
    # For C2 (2x2): Same as C1. Rank 11.
    
    # For C3 (2x3):
    # A = [A1 (2x2), A2 (2x1)]
    # B3 = [B31 (2x3), B32 (1x3)]^T
    # C3 = A1 B31 + A2 B32
    # A1 B31 is 2x2x3 -> Rank 12 (Schoolbook, since 2x2x3 rank is not 7)
    # A2 B32 is 2x1x3 -> Rank 6 (Schoolbook)
    # Total 18.
    
    # Total rank = 11 + 11 + 18 = 40.
    
    rank = 40
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen factors for 2x2x2
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # Indices for A (2x2): 0,1,2,3 (row-major: 00,01,10,11)
    # Indices for B (2x2): 0,1,2,3
    # Indices for C (2x2): 0,1,2,3
    
    # We need to map these to the global indices.
    # Global A index: i*m + j. n=2, m=3.
    # A1 is A[:, 0:2]. Indices: (0,0)->0, (0,1)->1, (1,0)->3, (1,1)->4.
    # A2 is A[:, 2]. Indices: (0,2)->2, (1,2)->5.
    
    # Global B index: j*p + k. m=3, p=7.
    # B1 is B[0:2, 0:2]. Indices: (0,0)->0, (0,1)->1, (1,0)->7, (1,1)->8.
    # B12 is B[2, 0:2]. Indices: (2,0)->14, (2,1)->15.
    
    # B2 is B[0:2, 2:4]. Indices: (0,2)->2, (0,3)->3, (1,2)->9, (1,3)->10.
    # B22 is B[2, 2:4]. Indices: (2,2)->16, (2,3)->17.
    
    # B3 is B[0:2, 4:7]. Indices: (0,4)->4, (0,5)->5, (0,6)->6, (1,4)->11, (1,5)->12, (1,6)->13.
    # B32 is B[2, 4:7]. Indices: (2,4)->18, (2,5)->19, (2,6)->20.
    
    # Global C index: k*n + i. p=7, n=2.
    # C1 is C[:, 0:2]. Indices: (0,0)->0, (0,1)->2, (1,0)->1, (1,1)->3.
    # C2 is C[:, 2:4]. Indices: (2,0)->4, (2,1)->6, (3,0)->5, (3,1)->7.
    # C3 is C[:, 4:7]. Indices: (4,0)->8, (4,1)->10, (5,0)->9, (5,1)->11, (6,0)->12, (6,1)->14.
    
    # Helper to add Strassen block
    def add_strassen(a_idx, b_idx, c_idx, offset):
        # a_idx: list of 4 indices for A1 (00, 01, 10, 11)
        # b_idx: list of 4 indices for B1 (00, 01, 10, 11)
        # c_idx: list of 4 indices for C1 (00, 01, 10, 11)
        
        # M1
        U[a_idx[0]][offset] += 1; U[a_idx[3]][offset] += 1
        V[b_idx[0]][offset] += 1; V[b_idx[3]][offset] += 1
        W[c_idx[0]][offset] += 1; W[c_idx[3]][offset] += 1
        offset += 1
        
        # M2
        U[a_idx[2]][offset] += 1; U[a_idx[3]][offset] += 1
        V[b_idx[0]][offset] += 1
        W[c_idx[2]][offset] += 1; W[c_idx[3]][offset] += 1
        offset += 1
        
        # M3
        U[a_idx[0]][offset] += 1
        V[b_idx[2]][offset] += 1; V[b_idx[3]][offset] -= 1
        W[c_idx[1]][offset] += 1; W[c_idx[3]][offset] += 1
        offset += 1
        
        # M4
        U[a_idx[3]][offset] += 1
        V[b_idx[1]][offset] += 1; V[b_idx[0]][offset] -= 1
        W[c_idx[0]][offset] += 1; W[c_idx[2]][offset] += 1
        offset += 1
        
        # M5
        U[a_idx[0]][offset] += 1; U[a_idx[1]][offset] += 1
        V[b_idx[3]][offset] += 1
        W[c_idx[0]][offset] -= 1; W[c_idx[1]][offset] += 1
        offset += 1
        
        # M6
        U[a_idx[2]][offset] += 1; U[a_idx[0]][offset] -= 1
        V[b_idx[0]][offset] += 1; V[b_idx[1]][offset] += 1
        W[c_idx[3]][offset] += 1
        offset += 1
        
        # M7
        U[a_idx[1]][offset] += 1; U[a_idx[3]][offset] -= 1
        V[b_idx[2]][offset] += 1; V[b_idx[3]][offset] += 1
        W[c_idx[0]][offset] += 1
        offset += 1
        
        return offset

    # Helper to add schoolbook block for 2x1x2 (A2 B12)
    # A2 indices: a_idx (2 indices)
    # B12 indices: b_idx (2 indices)
    # C12 indices: c_idx (2 indices)
    def add_schoolbook_2x1x2(a_idx, b_idx, c_idx, offset):
        # C[0] = A[0]*B[0] + A[1]*B[1] ? No.
        # A2 is 2x1. B12 is 1x2.
        # A2 = [a0, a1]^T. B12 = [b0, b1].
        # C = [a0*b0, a0*b1, a1*b0, a1*b1]? No, C is 2x2.
        # C[0,0] = a0*b0. C[0,1] = a0*b1.
        # C[1,0] = a1*b0. C[1,1] = a1*b1.
        # 4 terms.
        
        # Term 1: A[0] B[0] -> C[0,0]
        U[a_idx[0]][offset] += 1
        V[b_idx[0]][offset] += 1
        W[c_idx[0]][offset] += 1
        offset += 1
        
        # Term 2: A[0] B[1] -> C[0,1]
        U[a_idx[0]][offset] += 1
        V[b_idx[1]][offset] += 1
        W[c_idx[1]][offset] += 1
        offset += 1
        
        # Term 3: A[1] B[0] -> C[1,0]
        U[a_idx[1]][offset] += 1
        V[b_idx[0]][offset] += 1
        W[c_idx[2]][offset] += 1
        offset += 1
        
        # Term 4: A[1] B[1] -> C[1,1]
        U[a_idx[1]][offset] += 1
        V[b_idx[1]][offset] += 1
        W[c_idx[3]][offset] += 1
        offset += 1
        
        return offset

    # Helper to add schoolbook block for 2x2x3 (A1 B31)
    # A1 indices: a_idx (4 indices)
    # B31 indices: b_idx (6 indices)
    # C31 indices: c_idx (6 indices)
    def add_schoolbook_2x2x3(a_idx, b_idx, c_idx, offset):
        # 12 terms.
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    U[a_idx[i*2+j]][offset] += 1
                    V[b_idx[j*3+k]][offset] += 1
                    W[c_idx[k*2+i]][offset] += 1
                    offset += 1
        return offset

    # Helper to add schoolbook block for 2x1x3 (A2 B32)
    # A2 indices: a_idx (2 indices)
    # B32 indices: b_idx (3 indices)
    # C32 indices: c_idx (6 indices)
    def add_schoolbook_2x1x3(a_idx, b_idx, c_idx, offset):
        # 6 terms.
        for i in range(2):
            for k in range(3):
                U[a_idx[i]][offset] += 1
                V[b_idx[k]][offset] += 1
                W[c_idx[k*2+i]][offset] += 1
                offset += 1
        return offset

    # Indices for A1 (2x2)
    a1_idx = [0, 1, 3, 4] # (0,0), (0,1), (1,0), (1,1)
    # Indices for A2 (2x1)
    a2_idx = [2, 5] # (0,2), (1,2)
    
    # C1 (2x2)
    b11_idx = [0, 1, 7, 8] # B[0:2, 0:2]
    b12_idx = [14, 15] # B[2, 0:2]
    c1_idx = [0, 2, 1, 3] # C[0:2, 0:2] -> (0,0), (0,1), (1,0), (1,1)
    
    col = add_strassen(a1_idx, b11_idx, c1_idx, col)
    col = add_schoolbook_2x1x2(a2_idx, b12_idx, c1_idx, col)
    
    # C2 (2x2)
    b21_idx = [2, 3, 9, 10] # B[0:2, 2:4]
    b22_idx = [16, 17] # B[2, 2:4]
    c2_idx = [4, 6, 5, 7] # C[0:2, 2:4] -> (2,0), (2,1), (3,0), (3,1)
    
    col = add_strassen(a1_idx, b21_idx, c2_idx, col)
    col = add_schoolbook_2x1x2(a2_idx, b22_idx, c2_idx, col)
    
    # C3 (2x3)
    b31_idx = [4, 5, 6, 11, 12, 13] # B[0:2, 4:7]
    b32_idx = [18, 19, 20] # B[2, 4:7]
    c3_idx = [8, 10, 12, 9, 11, 13] # C[0:2, 4:7] -> (4,0), (4,1), (5,0), (5,1), (6,0), (6,1)
    # Wait, C index is k*n + i.
    # k=4, i=0 -> 8. k=4, i=1 -> 10.
    # k=5, i=0 -> 10? No, 5*2+0=10.
    # My c3_idx list:
    # (4,0)->8, (4,1)->10
    # (5,0)->10, (5,1)->12
    # (6,0)->12, (6,1)->14
    # So c3_idx = [8, 10, 10, 12, 12, 14]
    
    c3_idx = [8, 10, 10, 12, 12, 14]
    
    col = add_schoolbook_2x2x3(a1_idx, b31_idx, c3_idx, col)
    col = add_schoolbook_2x1x3(a2_idx, b32_idx, c3_idx, col)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
