DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    # Use block decomposition: 3 blocks of 2x3x3
    # Rank of 2x3x3 is 12. Total rank 36.
    # If 12 is not available, use 14. Total rank 42.
    
    # I will implement the 12-term decomposition for 2x3x3.
    # References indicate R(2,3,3) = 12.
    
    # The 12 terms for 2x3x3:
    # A is 2x3, B is 3x3.
    # C = AB.
    
    # I will hardcode the 12 linear forms.
    # This is risky if I don't have them memorized.
    # Alternative: Use the 14-term decomposition (7 from Strassen 2x2x2 + 7 from remainder?).
    # Actually, 2x3x2 is 11. 2x1x1 is 3? No, 2x3x1 is 3.
    # 11 + 3 = 14.
    
    # Let's use the 14-term decomposition for 2x3x3.
    # Rank 14.
    # Total rank 3 * 14 = 42.
    
    # Is there a better way?
    # R(2, 3, 9) <= R(2, 3, 4) + R(2, 3, 5)?
    # R(2, 3, 4) <= R(2, 3, 2) * 2 = 22?
    # R(2, 3, 5) <= 11 + 3 + 3 = 17?
    # 22 + 17 = 39.
    # 39 < 42.
    
    # Let's try to construct R(2, 3, 4) with rank 22?
    # Or R(2, 3, 9) with rank 36?
    
    # I will implement the 42 rank solution first as it is more robust to derive.
    # 14 terms for 2x3x3:
    # 7 terms for 2x2x2 (Strassen)
    # 7 terms for the rest?
    # A = [A22, A21], B = [B22, B21; B12, B11]
    # C = A22 B22 + A21 B12 + A22 B21 + A21 B11? No.
    # C = A B.
    # Split B into B1 (3x2) and B2 (3x1).
    # C1 = A B1 (2x2). Rank 11.
    # C2 = A B2 (2x1). Rank 3.
    # Total 14.
    
    # I will implement this.
    
    U = []
    V = []
    W = []
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, val=1):
        # u_idx is (i, j) for A
        # v_idx is (j, k) for B
        # w_idx is (k, i) for C? No, W is (p*n, rank). Index k*n+i.
        # In 2x3x3 block:
        # n=2, m=3, p=3.
        # A index i*3+j.
        # B index j*3+k.
        # W index k*2+i.
        
        U.append(u_idx)
        V.append(v_idx)
        W.append(w_idx)
        
    # This approach is messy. I will build the matrices directly.
    
    rank = 42
    U_mat = [[0]*rank for _ in range(6)] # 2*3
    V_mat = [[0]*rank for _ in range(9)] # 3*9
    W_mat = [[0]*rank for _ in range(18)] # 9*2
    
    col = 0
    # 3 blocks of 3 columns in B
    for block in range(3):
        # B_block is columns 3*block to 3*block+2
        # C_block is columns 3*block to 3*block+2
        
        # 1. 2x2x2 Strassen part (using A[:, 0:2] and B_block[:, 0:2])
        # A_part is 2x2. B_part is 2x2.
        # Strassen 7 terms.
        
        # A11=A[0,0], A12=A[0,1], A21=A[1,0], A22=A[1,1]
        # B11=B[0,0], B12=B[0,1], B21=B[1,0], B22=B[1,1]
        
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # I will map these to the global indices.
        
        # A indices: i*3+j
        # B indices: j*3+k (local k 0..1) -> global k 3*block+k
        # W indices: k*2+i (global k)
        
        def add_strassen_term(u_vals, v_vals, w_vals, val=1):
            nonlocal col
            for i in range(2):
                for j in range(2):
                    if u_vals[i][j] != 0:
                        U_mat[i*3+j][col] += u_vals[i][j] * val
            for j in range(2):
                for k in range(2):
                    if v_vals[j][k] != 0:
                        V_mat[j*3 + (3*block+k)][col] += v_vals[j][k] * val
            for k in range(2):
                for i in range(2):
                    if w_vals[k][i] != 0:
                        W_mat[(3*block+k)*2 + i][col] += w_vals[k][i] * val
            col += 1
            
        # M1
        add_strassen_term([[1,0],[0,1]], [[1,0],[0,1]], [[1,1],[1,1]]) # Wait, W is C.
        # C11, C12, C21, C22 coefficients for M1 are 1, 0, 0, 1?
        # C11=1, C12=0, C21=0, C22=1? No.
        # C11 = M1... C22 = M1...
        # So W for M1: C11=1, C22=1.
        # W index k*2+i.
        # k=0 (col 0), i=0 (row 0) -> C00.
        # k=1 (col 1), i=1 (row 1) -> C11.
        # So W[0*2+0]=1, W[1*2+1]=1.
        
        # I will hardcode the 7 Strassen terms properly.
        
        # Term 1: M1
        # U: A11+A22 -> [[1,0],[0,1]]
        # V: B11+B22 -> [[1,0],[0,1]]
        # W: C11+ C22 -> [[1,0],[0,1]]? No, W is (p*n, rank).
        # W[0,0]=1 (C00), W[1,1]=1 (C11).
        
        # This is getting complex. I will just use the schoolbook for the 2x3x3 block?
        # Rank 6. Total 18.
        # 18 is the lower bound!
        # If I use schoolbook for 2x3x3, rank is 6.
        # 3 blocks -> 18.
        # Is 2x3x3 rank 6?
        # Lower bound is 2*3=6.
        # Is R(2,3,3)=6?
        # No, R(2,3,3)=12.
        # Schoolbook is 6?
        # A is 2x3, B is 3x3.
        # C is 2x3.
        # C_ik = sum_j A_ij B_jk.
        # Schoolbook uses 2*3*3 = 18 terms?
        # No, for each k, we do 2x3x1.
        # 2x3x1 rank is 3.
        # 3 columns -> 9 terms.
        # Wait, schoolbook for 2x3x3 is 6?
        # No, $n \times m \times p$ schoolbook is $nmp$.
        # $2 \times 3 \times 3 = 18$.
        # So 3 blocks of 18 is 54.
        
        # I need a better decomposition for 2x3x3.
        # I will use the 12-term one if I can find it.
        # Otherwise, I will use the 14-term one.
        
        # Let's use the 14-term one.
        # 11 terms for 2x3x2 + 3 terms for 2x3x1.
        
        # I will implement the 14-term decomposition for 2x3x3.
        # Total rank 42.
        
        pass # Placeholder

    return {"U": U_mat, "V": V_mat, "W": W_mat}
