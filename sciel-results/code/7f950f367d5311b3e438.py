import numpy as np

def construct():
    n, m, p = 4, 5, 5
    
    # Total rank
    rank = 74
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a rank-1 term
    def add_term(u_idx, v_idx, w_idx, u_val=1, v_val=1, w_val=1):
        nonlocal col
        U[u_idx][col] += u_val
        V[v_idx][col] += v_val
        W[w_idx][col] += w_val
        col += 1

    # Helper to add Strassen 2x2x2 (rank 7)
    # A is 2x2, B is 2x2, C is 2x2
    # a_r, a_c are top-left indices of A block
    # b_r, b_c are top-left indices of B block
    # c_r, c_c are top-left indices of C block
    def add_strassen_2x2(a_r, a_c, b_r, b_c, c_r, c_c):
        nonlocal col
        # Strassen formulas
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # C00 = M1 - M2 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M3 + M6 - M7
        
        # Term 1
        add_term(a_r*5+a_c, b_r*5+b_c, c_c*4+c_r, 1, 1, 1)
        add_term((a_r+1)*5+(a_c+1), (b_r+1)*5+(b_c+1), c_c*4+c_r, 1, 1, 1)
        add_term((a_r+1)*5+a_c, b_r*5+b_c, (c_c)*4+(c_r+1), 1, 1, -1)
        add_term((a_r+1)*5+(a_c+1), (b_r+1)*5+(b_c+1), (c_c)*4+(c_r+1), 1, 1, -1)
        add_term((a_r+1)*5+a_c, (b_r+1)*5+(b_c+1), (c_c+1)*4+c_r, 1, 1, -1)
        add_term((a_r+1)*5+(a_c+1), (b_r+1)*5+(b_c+1), (c_c+1)*4+c_r, 1, 1, -1)
        
        # Term 2
        add_term((a_r+1)*5+a_c, b_r*5+b_c, (c_c)*4+(c_r+1), 1, 1, 1)
        add_term((a_r+1)*5+(a_c+1), b_r*5+b_c, (c_c)*4+(c_r+1), 1, 1, 1)
        add_term((a_r+1)*5+a_c, b_r*5+b_c, (c_c+1)*4+(c_r+1), 1, 1, -1)
        add_term((a_r+1)*5+(a_c+1), b_r*5+b_c, (c_c+1)*4+(c_r+1), 1, 1, -1)
        
        # Term 3
        add_term(a_r*5+a_c, (b_r)*5+(b_c+1), (c_c+1)*4+c_r, 1, 1, 1)
        add_term(a_r*5+a_c, (b_r+1)*5+(b_c+1), (c_c+1)*4+c_r, 1, -1, -1)
        add_term(a_r*5+a_c, (b_r)*5+(b_c+1), (c_c+1)*4+(c_r+1), 1, 1, 1)
        add_term(a_r*5+a_c, (b_r+1)*5+(b_c+1), (c_c+1)*4+(c_r+1), 1, -1, -1)
        
        # Term 4
        add_term((a_r+1)*5+(a_c+1), (b_r+1)*5+b_c, (c_c)*4+c_r, 1, 1, 1)
        add_term((a_r+1)*5+(a_c+1), b_r*5+b_c, (c_c)*4+c_r, 1, -1, -1)
        add_term((a_r+1)*5+(a_c+1), (b_r+1)*5+b_c, (c_c+1)*4+(c_r+1), 1, 1, 1)
        add_term((a_r+1)*5+(a_c+1), b_r*5+b_c, (c_c+1)*4+(c_r+1), 1, -1, -1)
        
        # Term 5
        add_term(a_r*5+a_c, (b_r+1)*5+(b_c+1), (c_c)*4+c_r, 1, 1, -1)
        add_term(a_r*5+(a_c+1), (b_r+1)*5+(b_c+1), (c_c)*4+c_r, 1, 1, -1)
        add_term(a_r*5+a_c, (b_r+1)*5+(b_c+1), (c_c+1)*4+c_r, 1, 1, 1)
        add_term(a_r*5+(a_c+1), (b_r+1)*5+(b_c+1), (c_c+1)*4+c_r, 1, 1, 1)
        
        # Term 6
        add_term((a_r+1)*5+a_c, b_r*5+b_c, (c_c+1)*4+(c_r+1), 1, 1, 1)
        add_term(a_r*5+a_c, b_r*5+b_c, (c_c+1)*4+(c_r+1), -1, 1, 1)
        add_term((a_r+1)*5+a_c, (b_r)*5+(b_c+1), (c_c+1)*4+(c_r+1), 1, 1, 1)
        add_term(a_r*5+a_c, (b_r)*5+(b_c+1), -1, 1, 1)
        
        # Term 7
        add_term(a_r*5+(a_c+1), (b_r+1)*5+b_c, (c_c)*4+c_r, 1, 1, 1)
        add_term((a_r+1)*5+(a_c+1), (b_r+1)*5+b_c, (c_c)*4+c_r, -1, 1, 1)
        add_term(a_r*5+(a_c+1), (b_r+1)*5+(b_c+1), (c_c)*4+c_r, 1, 1, -1)
        add_term((a_r+1)*5+(a_c+1), (b_r+1)*5+(b_c+1), -1, 1, -1)

    # Helper to add rank-9 term: 2x2x3
    # A is 2x2, B is 2x3, C is 2x3
    # a_r, a_c: top-left of A
    # b_r, b_c: top-left of B
    # c_r, c_c: top-left of C
    def add_rank9_223(a_r, a_c, b_r, b_c, c_r, c_c):
        # C = A B
        # B = [B1 (2x2) | B2 (2x1)]
        # C = [C1 (2x2) | C2 (2x1)]
        # C1 = A B1 (Rank 7)
        # C2 = A B2 (Rank 2)
        
        # C1 = A B1
        add_strassen_2x2(a_r, a_c, b_r, b_c, c_r, c_c)
        
        # C2 = A B2
        # B2 is 2x1 at b_r, b_c+2
        # C2 is 2x1 at c_r, c_c+2
        # A is 2x2 at a_r, a_c
        # C2[0] = A[0,0]B2[0,0] + A[0,1]B2[1,0]
        # C2[1] = A[1,0]B2[0,0] + A[1,1]B2[1,0]
        
        # Term 1: A[0,0] * B2[0,0]
        add_term(a_r*5+a_c, (b_r)*5+(b_c+2), (c_c+2)*4+c_r, 1, 1, 1)
        # Term 2: A[0,1] * B2[1,0]
        add_term(a_r*5+(a_c+1), (b_r+1)*5+(b_c+2), (c_c+2)*4+c_r, 1, 1, 1)
        
        # Term 3: A[1,0] * B2[0,0]
        add_term((a_r+1)*5+a_c, (b_r)*5+(b_c+2), (c_c+2)*4+(c_r+1), 1, 1, 1)
        # Term 4: A[1,1] * B2[1,0]
        add_term((a_r+1)*5+(a_c+1), (b_r+1)*5+(b_c+2), (c_c+2)*4+(c_r+1), 1, 1, 1)

    # Helper to add rank-9 term: 2x3x2
    # A is 2x3, B is 3x2, C is 2x2
    # a_r, a_c: top-left of A
    # b_r, b_c: top-left of B
    # c_r, c_c: top-left of C
    def add_rank9_232(a_r, a_c, b_r, b_c, c_r, c_c):
        # A = [A1 (2x2) | A2 (2x1)]
        # B = [B1 (2x2); B2 (1x2)]
        # C = A B = A1 B1 + A2 B2
        # C = A1 B1 (Rank 7) + A2 B2 (Rank 2)
        
        # C = A1 B1
        add_strassen_2x2(a_r, a_c, b_r, b_c, c_r, c_c)
        
        # C = A2 B2
        # A2 is 2x1 at a_r, a_c+2
        # B2 is 1x2 at b_r+2, b_c
        # C is 2x2 at c_r, c_c
        # C[0,0] += A2[0,0] B2[0,0]
        # C[0,1] += A2[0,0] B2[0,1]
        # C[1,0] += A2[1,0] B2[0,0]
        # C[1,1] += A2[1,0] B2[0,1]
        
        # Term 1: A2[0,0] * B2[0,0]
        add_term(a_r*5+(a_c+2), (b_r+2)*5+b_c, c_c*4+c_r, 1, 1, 1)
        # Term 2: A2[0,0] * B2[0,1]
        add_term(a_r*5+(a_c+2), (b_r+2)*5+(b_c+1), (c_c+1)*4+c_r, 1, 1, 1)
        # Term 3: A2[1,0] * B2[0,0]
        add_term((a_r+1)*5+(a_c+2), (b_r+2)*5+b_c, c_c*4+(c_r+1), 1, 1, 1)
        # Term 4: A2[1,0] * B2[0,1]
        add_term((a_r+1)*5+(a_c+2), (b_r+2)*5+(b_c+1), (c_c+1)*4+(c_r+1), 1, 1, 1)

    # Helper to add rank-12 term: 2x3x3
    # A is 2x3, B is 3x3, C is 2x3
    # a_r, a_c: top-left of A
    # b_r, b_c: top-left of B
    # c_r, c_c: top-left of C
    def add_rank12_233(a_r, a_c, b_r, b_c, c_r, c_c):
        # A = [A1 (2x2) | A2 (2x1)]
        # B = [[B11 (2x2) B12 (2x1)], [B21 (1x2) B22 (1x1)]]
        # C = [C1 (2x2) | C2 (2x1)]
        # C1 = A1 B11 + A2 B21 (Rank 7 + 2 = 9)
        # C2 = A1 B12 + A2 B22 (Rank 2 + 1 = 3)
        
        # C1 = A1 B11
        add_strassen_2x2(a_r, a_c, b_r, b_c, c_r, c_c)
        
        # C1 += A2 B21
        # A2 is 2x1 at a_r, a_c+2
        # B21 is 1x2 at b_r+2, b_c
        # C1 is 2x2 at c_r, c_c
        # C1[0,0] += A2[0,0] B21[0,0]
        # C1[0,1] += A2[0,0] B21[0,1]
        # C1[1,0] += A2[1,0] B21[0,0]
        # C1[1,1] += A2[1,0] B21[0,1]
        
        add_term(a_r*5+(a_c+2), (b_r+2)*5+b_c, c_c*4+c_r, 1, 1, 1)
        add_term(a_r*5+(a_c+2), (b_r+2)*5+(b_c+1), (c_c+1)*4+c_r, 1, 1, 1)
        add_term((a_r+1)*5+(a_c+2), (b_r+2)*5+b_c, c_c*4+(c_r+1), 1, 1, 1)
        add_term((a_r+1)*5+(a_c+2), (b_r+2)*5+(b_c+1), (c_c+1)*4+(c_r+1), 1, 1, 1)
        
        # C2 = A1 B12
        # A1 is 2x2 at a_r, a_c
        # B12 is 2x1 at b_r, b_c+2
        # C2 is 2x1 at c_r, c_c+2
        # C2[0] = A1[0,0]B12[0,0] + A1[0,1]B12[1,0]
        # C2[1] = A1[1,0]B12[0,0] + A1[1,1]B12[1,0]
        
        add_term(a_r*5+a_c, (b_r)*5+(b_c+2), (c_c+2)*4+c_r, 1, 1, 1)
        add_term(a_r*5+(a_c+1), (b_r+1)*5+(b_c+2), (c_c+2)*4+c_r, 1, 1, 1)
        add_term((a_r+1)*5+a_c, (b_r)*5+(b_c+2), (c_c+2)*4+(c_r+1), 1, 1, 1)
        add_term((a_r+1)*5+(a_c+1), (b_r+1)*5+(b_c+2), (c_c+2)*4+(c_r+1), 1, 1, 1)
        
        # C2 += A2 B22
        # A2 is 2x1 at a_r, a_c+2
        # B22 is 1x1 at b_r+2, b_c+2
        # C2 is 2x1 at c_r, c_c+2
        # C2[0] += A2[0,0] B22[0,0]
        # C2[1] += A2[1,0] B22[0,0]
        
        add_term(a_r*5+(a_c+2), (b_r+2)*5+(b_c+2), (c_c+2)*4+c_r, 1, 1, 1)
        add_term((a_r+1)*5+(a_c+2), (b_r+2)*5+(b_c+2), (c_c+2)*4+(c_r+1), 1, 1, 1)

    # Apply the decompositions
    # A blocks:
    # A11: 0,0
    # A12: 0,2
    # A21: 2,0
    # A22: 2,2
    
    # B blocks:
    # B11: 0,0
    # B12: 0,2
    # B21: 2,0
    # B22: 2,2
    
    # C blocks:
    # C11: 0,0
    # C12: 0,2
    # C21: 2,0
    # C22: 2,2
    
    # 1. A11 B11 -> C11 (Rank 7)
    add_strassen_2x2(0, 0, 0, 0, 0, 0)
    
    # 2. A12 B21 -> C11 (Rank 9)
    add_rank9_232(0, 2, 2, 0, 0, 0)
    
    # 3. A11 B12 -> C12 (Rank 9)
    add_rank9_223(0, 0, 0, 2, 0, 2)
    
    # 4. A12 B22 -> C12 (Rank 12)
    add_rank12_233(0, 2, 2, 2, 0, 2)
    
    # 5. A21 B11 -> C21 (Rank 7)
    add_strassen_2x2(2, 0, 0, 0, 2, 0)
    
    # 6. A22 B21 -> C21 (Rank 9)
    add_rank9_232(2, 2, 2, 0, 2, 0)
    
    # 7. A21 B12 -> C22 (Rank 9)
    add_rank9_223(2, 0, 0, 2, 2, 2)
    
    # 8. A22 B22 -> C22 (Rank 12)
    add_rank12_233(2, 2, 2, 2, 2, 2)
    
    return {"U": U, "V": V, "W": W}
