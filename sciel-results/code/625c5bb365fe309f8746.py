import numpy as np

def construct():
    n, m, p = 2, 3, 4
    rank = 20
    U = [[0] * rank for _ in range(n * m)] # 6 x 20
    V = [[0] * rank for _ in range(m * p)] # 12 x 20
    W = [[0] * rank for _ in range(p * n)] # 8 x 20
    
    # Helper to add a rank-1 term
    # i, j for A (0-1, 0-2) -> U[i*m+j]
    # j, k for B (0-2, 0-3) -> V[j*p+k]
    # k, i for C (0-3, 0-1) -> W[k*n+i]
    
    # We use two blocks of 2x3x2
    # Block 1: C cols 0,1. B cols 0,1.
    # Block 2: C cols 2,3. B cols 2,3.
    
    def add_block(b_col_offset, w_col_offset):
        # 10 multiplications for 2x3x2
        # A is 2x3, B is 3x2
        # C is 2x2
        # m1 = a11*b11
        # m2 = a12*b21
        # m3 = a13*b31
        # m4 = a21*b11
        # m5 = a22*b21
        # m6 = a23*b31
        # m7 = (a11+a12+a13)*(b12+b22+b32)
        # m8 = (a11+a12)*(b12+b22)
        # m9 = (a11+a13)*(b12+b32)
        # m10 = (a12+a13)*(b22+b32)
        
        # C11 = m1+m2+m3 - m8 - m9 + m10 ? No.
        # Standard 2x3x2 rank 10:
        # C11 = m1 + m2 + m3
        # C12 = m7 - m8 - m9 + m10
        # C21 = m4 + m5 + m6
        # C22 = (a21+a22+a23)*(b12+b22+b32) - ...
        
        # I will use a known valid set.
        # Let's use the one from "Fast Matrix Multiplication" for 2x3x2.
        # Actually, I'll just use the schoolbook 6 for each 2x3x1? No, that's 12.
        # Rank 10 is better.
        
        # I will use the following 10 mults for 2x3x2:
        # 1. a11*b11
        # 2. a12*b21
        # 3. a13*b31
        # 4. a21*b11
        # 5. a22*b21
        # 6. a23*b31
        # 7. (a11+a12+a13)*(b12+b22+b32)
        # 8. (a21+a22+a23)*(b12+b22+b32)
        # 9. (a11+a12)*(b12+b22)
        # 10. (a21+a22)*(b12+b22)
        
        # C11 = 1+2+3
        # C21 = 4+5+6
        # C12 = 7 - 9
        # C22 = 8 - 10
        
        # This works for the second column of B (index 1 in local B).
        # What about the first column of B (index 0)?
        # We used b11, b21, b31 in 1-6.
        # We used b12, b22, b32 in 7-10.
        # So this covers both columns.
        
        # Local A indices: 0,1,2 (cols)
        # Local B indices: 0,1 (cols)
        
        # U factors (A)
        # V factors (B)
        # W factors (C)
        
        # Term 1: a11 * b11
        U[0][0] = 1
        V[0][0] = 1
        W[0][0] = 1 # C11 (k=0, i=0)
        
        # Term 2: a12 * b21
        U[1][1] = 1
        V[4][1] = 1 # j=1, k=0 -> 1*2+0=2? No. V[j*p+k]. p=2 local? No, global p=4.
        # Wait, V is global.
        # Local B col 0 corresponds to global B col b_col_offset.
        # Local B col 1 corresponds to global B col b_col_offset+1.
        # Local B row j is global row j.
        
        # V index: j * p + k.
        # Term 1: a11 (i=0, j=0) * b11 (j=0, k=0).
        # U[0*3+0] = U[0]
        # V[0*4+0] = V[0]
        # W[0*2+0] = W[0]
        
        # Term 2: a12 (i=0, j=1) * b21 (j=1, k=0).
        # U[0*3+1] = U[1]
        # V[1*4+0] = V[4]
        # W[0*2+0] = W[0]
        
        # Term 3: a13 (i=0, j=2) * b31 (j=2, k=0).
        # U[2]
        # V[8]
        # W[0]
        
        # Term 4: a21 (i=1, j=0) * b11 (j=0, k=0).
        # U[3]
        # V[0]
        # W[2] # C21 (k=0, i=1) -> 0*2+1=1? No. W[k*n+i]. k=0, i=1 -> 1.
        # C21 is row 1, col 0. k=0, i=1. W[1].
        
        # Term 5: a22 (i=1, j=1) * b21 (j=1, k=0).
        # U[4]
        # V[4]
        # W[1]
        
        # Term 6: a23 (i=1, j=2) * b31 (j=2, k=0).
        # U[5]
        # V[8]
        # W[1]
        
        # Term 7: (a11+a12+a13) * (b12+b22+b32)
        # U[0]+=1, U[1]+=1, U[2]+=1
        # V[1]+=1 (j=0, k=1), V[5]+=1 (j=1, k=1), V[9]+=1 (j=2, k=1)
        # W[3] (C12: k=1, i=0 -> 1*2+0=2? No. W[k*n+i]. k=1, i=0 -> 2.
        # C12 is row 0, col 1. k=1, i=0. W[2].
        
        # Term 8: (a21+a22+a23) * (b12+b22+b32)
        # U[3]+=1, U[4]+=1, U[5]+=1
        # V[1]+=1, V[5]+=1, V[9]+=1
        # W[3] (C22: k=1, i=1 -> 1*2+1=3)
        
        # Term 9: (a11+a12) * (b12+b22)
        # U[0]+=1, U[1]+=1
        # V[1]+=1, V[5]+=1
        # W[2] -= 1 (C12)
        
        # Term 10: (a21+a22) * (b12+b22)
        # U[3]+=1, U[4]+=1
        # V[1]+=1, V[5]+=1
        # W[3] -= 1 (C22)
        
        # Check C12: T7 - T9 = (a11+a12+a13)(b12+b22+b32) - (a11+a12)(b12+b22)
        # = a11b12+a11b22+a11b32 + a12b12+a12b22+a12b32 + a13b12+a13b22+a13b32
        #   - (a11b12+a11b22+a12b12+a12b22)
        # = a11b32 + a12b32 + a13b12+a13b22+a13b32
        # This is NOT a11b12+a12b22+a13b32.
        
        # My formula for 2x3x2 rank 10 is wrong.
        
        # I will use the schoolbook rank 6 for 2x3x1 twice.
        # Total 12 for 2x3x2.
        # Two blocks: 24. No improvement.
        
        # I will use the known rank 10 for 2x3x2.
        # Reference: "The rank of 2x3x2 matrix multiplication is 10".
        # The 10 mults are:
        # 1. a11*b11
        # 2. a12*b21
        # 3. a13*b31
        # 4. a21*b11
        # 5. a22*b21
        # 6. a23*b31
        # 7. (a11+a12+a13)*(b12+b22+b32)
        # 8. (a21+a22+a23)*(b12+b22+b32)
        # 9. (a11+a12)*(b12+b22)
        # 10. (a21+a22)*(b12+b22)
        
        # C11 = 1+2+3
        # C21 = 4+5+6
        # C12 = 7 - 9
        # C22 = 8 - 10
        
        # Let's re-verify C12.
        # 7 = a11b12+a11b22+a11b32 + a12b12+a12b22+a12b32 + a13b12+a13b22+a13b32
        # 9 = a11b12+a11b22+a12b12+a12b22
        # 7-9 = a11b32 + a12b32 + a13b12+a13b22+a13b32
        # We want C12 = a11b12 + a12b22 + a13b32.
        # It doesn't match.
        
        # I will stick to schoolbook 24. It is safe.
        # But I can do better with 20?
        # If I can't get the 10 right, I use 12 per block (6+6).
        # 12+12=24.
        
        # I will output 24.
        pass

    # I will just return the schoolbook 24.
    # It is the seed.
    # The directive says "stronger algorithm".
    # But if I can't find the 19/20, 24 is the only safe bet.
    # Wait, I can try to implement the 19 using the 13+6 logic if I find the 13.
    # I don't have the 13.
    
    # I will return the schoolbook.
    rank = 24
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
