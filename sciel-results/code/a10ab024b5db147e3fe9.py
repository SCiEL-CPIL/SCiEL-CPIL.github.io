"""Tensor decomposition for 2x4 by 4x8 matrix multiplication."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=8
    # We model this as:
    #   A is n x m = 2 x 4
    #   B is m x p = 4 x 8
    #   C is n x p = 2 x 8
    #
    # Decompose m=4 as 2*2 (j1, j2), p=8 as 4*2 (k1, k2) where k1 is 4-dim, k2 is 2-dim
    # Actually, let me think more carefully.
    #
    # The multiplication tensor: C[i,k] = sum_j A[i,j] * B[j,k]
    # where i in [0,n), j in [0,m), k in [0,p)
    #
    # A natural decomposition: treat the 2x4x8 problem as a tensor product of smaller problems.
    #
    # Option 1: Use Strassen for 2x2x2 (rank 7) combined with identity on the remaining dims.
    #   n=2: single 2-dim factor
    #   m=4: split as 2*2 -> (j1, j2) each 2-dim
    #   p=8: split as 4*2 -> (k1, k2) where k1 is 4-dim, k2 is 2-dim
    #
    # This gets complicated. Let me try a different approach.
    #
    # Option 2: Think of 2x4x8 as a Kronecker product of 2x2x4 and 1x2x2.
    #   n=2 = 2*1, m=4 = 2*2, p=8 = 4*2
    #   So C = C1 ⊗ C2 where C1 is 2x2x4 (A1 is 2x2, B1 is 2x4, C1 is 2x4) and C2 is 1x2x2.
    #   But C2 with n2=1 is just standard multiplication: rank = 1*2*2 = 4.
    #   C1 is 2x2x4: this is A1[2x2] * B1[2x4]. We can decompose B1 as two 2x2 blocks.
    #   So C1 = A1 * B1a + A1 * B1b, where B1a and B1b are 2x2.
    #   Using Strassen for each: rank(C1) = 7 + 7 = 14? No, that's not right either.
    #
    # Let me try a cleaner approach.
    #
    # The tensor for matrix multiplication M_{n,m,p} has rank at most n*m*p (schoolbook).
    # For n=2, m=4, p=8: schoolbook rank = 64.
    #
    # Strassen's algorithm gives rank 7 for 2x2x2.
    # 
    # We can decompose the 2x4x8 problem:
    #   Write A as 2x4, B as 4x8, C as 2x8.
    #   Split j (4-dim) into j = j1*2 + j2, where j1, j2 in {0,1}
    #   Split k (8-dim) into k = k1*2 + k2, where k1 in {0,1,2,3}, k2 in {0,1}
    #   
    #   C[i, k1*2+k2] = sum_{j1,j2} A[i, j1*2+j2] * B[j1*2+j2, k1*2+k2]
    #
    # This is a 2x(2x2)x(4x2) = 2x4x8 problem.
    #
    # We can view this as: for each fixed (j2, k2), we have a 2x2x4 subproblem
    # (summing over j1 and k1). There are 2*2 = 4 such subproblems.
    # Each 2x2x4 subproblem: A_sub is 2x2, B_sub is 2x4, C_sub is 2x4.
    #
    # For a 2x2x4 multiplication: A[2,2] * B[2,4] = C[2,4]
    #   C[i, k] = sum_j A[i,j] * B[j,k]
    #   This can be seen as two independent 2x2x2 multiplications (one for k=0,1 and one for k=2,3)?
    #   No, B is 2x4, so for each k we have a 2x2x2 problem.
    #   Actually, C[i,0] = A[i,0]*B[0,0] + A[i,1]*B[1,0]
    #   C[i,1] = A[i,0]*B[0,1] + A[i,1]*B[1,1]
    #   C[i,2] = A[i,0]*B[0,2] + A[i,1]*B[1,2]
    #   C[i,3] = A[i,0]*B[0,3] + A[i,1]*B[1,3]
    #   So it's 4 independent 2x1x1 multiplications (scalar times 2-vector). Rank = 4*2 = 8? No.
    #   Actually each C[i,k] involves 2 multiplications and 1 addition. The tensor rank for 2x2x4 is...
    #   We can write it as: for each k, we have a 2x2x2 tensor slice. Using Strassen on each slice: 4*7 = 28? 
    #   But that's not right because the A matrix is shared across slices.
    #
    # Let me just directly construct a decomposition using Strassen's idea.
    #
    # Actually, the simplest improvement: use the fact that we can group the computation.
    # 
    # Let me try: treat the 4x8 part as a Kronecker product 2x2 ⊗ 2x4.
    #   m=4 = 2⊗2, p=8 = 2⊗4
    #   So B is a 4x8 matrix = (2x2) ⊗ (2x4) in some sense.
    #   And A is 2x4 = 2 ⊗ (1x4)? Or 2x4 = (2) ⊗ (1x4).
    #
    # This is getting too complicated. Let me just use a straightforward block approach with Strassen.
    #
    # Split A (2x4) into two 2x2 blocks: A = [A0 | A1] where A0, A1 are 2x2.
    # Split B (4x8) into four 2x4 blocks: B = [[B00, B01], [B10, B11]] where each is 2x4.
    # Then C = A0*B00 + A1*B10 (for the first 4 columns of C) and C = A0*B01 + A1*B11 (for the last 4 columns).
    #
    # Each 2x2 * 2x4 = 2x4 multiplication. We can further split each 2x4 into two 2x2 blocks.
    # So each 2x2*2x4 = two 2x2*2x2 multiplications.
    # Total: 4 * 2 = 8 Strassen 2x2x2 multiplications.
    # Rank = 8 * 7 = 56.
    #
    # But we can do better by sharing intermediate computations.
    #
    # Actually, let me just implement the 8-Strassen-block approach for rank 56.
    #
    # Strassen 2x2x2 decomposition:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # For each 2x2x2 subproblem, the U, V, W matrices are 4x7 each.
    #
    # Let me define the Strassen U, V, W for 2x2x2:
    # U[i*m+j, r] where i,j in {0,1}, r in {0,...,6}
    # V[j*p+k, r] where j,k in {0,1}, r in {0,...,6}
    # W[k*n+i, r] where k,i in {0,1}, r in {0,...,6}
    #
    # U:
    #   M1: A11+A22 -> U[0,0]=1, U[3,0]=1 (i=0,j=0 and i=1,j=1)
    #   M2: A21+A22 -> U[2,1]=1, U[3,1]=1
    #   M3: A11 -> U[0,2]=1
    #   M4: A22 -> U[3,3]=1
    #   M5: A11+A12 -> U[0,4]=1, U[1,4]=1
    #   M6: A21-A11 -> U[2,5]=1, U[0,5]=-1
    #   M7: A12-A22 -> U[1,6]=1, U[3,6]=-1
    #
    # V:
    #   M1: B11+B22 -> V[1,0]=1, V[3,0]=1
    #   M2: B11 -> V[1,1]=1
    #   M3: B21-B22 -> V[2,2]=1, V[3,2]=-1
    #   M4: B12-B11 -> V[2,3]=1, V[1,3]=-1
    #   M5: B22 -> V[3,4]=1
    #   M6: B11+B12 -> V[1,5]=1, V[2,5]=1
    #   M7: B21+B22 -> V[2,6]=1, V[3,6]=1
    #
    # W:
    #   M1: C11 += M1, C22 += M1 -> W[0,0]=1, W[3,0]=1
    #   M2: C21 += M2 -> W[2,1]=1
    #   M3: C12 += M3, C21 += M3 -> W[1,2]=1, W[2,2]=1
    #   M4: C11 += M4, C21 += M4 -> W[0,3]=1, W[2,3]=1
    #   M5: C11 -= M5, C12 += M5 -> W[0,4]=-1, W[1,4]=1
    #   M6: C22 += M6 -> W[3,5]=1
    #   M7: C11 += M7 -> W[0,6]=1
    #
    # Wait, let me double-check the W indices. W[k*n+i, r] where k is the column index (0 or 1) and i is the row index (0 or 1).
    # C11: k=0, i=0 -> W[0, r]
    # C12: k=1, i=0 -> W[1, r]
    # C21: k=0, i=1 -> W[2, r]
    # C22: k=1, i=1 -> W[3, r]
    #
    # C11 = M1 + M4 - M5 + M7 -> W[0,0]=1, W[0,3]=1, W[0,4]=-1, W[0,6]=1
    # C12 = M3 + M5 -> W[1,2]=1, W[1,4]=1
    # C21 = M2 + M4 -> W[2,1]=1, W[2,3]=1
    # C22 = M1 - M2 + M3 + M6 -> W[3,0]=1, W[3,1]=-1, W[3,2]=1, W[3,5]=1
    #
    # Now for the full 2x4x8 problem:
    # A is 2x4, B is 4x8, C is 2x8.
    # Split A into two 2x2 blocks: A = [A_left | A_right]
    # Split B into four 2x4 blocks: B = [[B_topleft, B_topright], [B_botleft, B_botright]]
    #
    # C = [A_left * B_topleft + A_right * B_botleft | A_left * B_topright + A_right * B_botright]
    #
    # Each 2x2 * 2x4 product: split the 2x4 into two 2x2 blocks.
    # A_left * B_topleft = A_left * [B_tl_left | B_tl_right] = [A_left*B_tl_left | A_left*B_tl_right]
    #
    # So we have 8 Strassen 2x2x2 multiplications.
    #
    # Let me index them:
    # For A block a in {0,1} (left=0, right=1):
    #   For B block b in {0,1,2,3} (tl=0, tr=1, bl=2, br=3):
    #     For C sub-block c in {0,1} (left half=0, right half=1):
    #       But actually, for a given (a, b), the result goes into specific columns of C.
    #
    # Let me be more precise:
    # A rows: i in {0,1}
    # A cols: j in {0,1,2,3}, split as j = a*2 + ja where a in {0,1}, ja in {0,1}
    # B rows: j in {0,1,2,3}, same split
    # B cols: k in {0,...,7}, split as k = c*4 + kc where c in {0,1}, kc in {0,1,2,3}
    #   Further split kc: kc = d*2 + kd where d in {0,1}, kd in {0,1}
    #   So k = c*4 + d*2 + kd
    #
    # C[i, k] = sum_j A[i,j] * B[j,k]
    #   = sum_{a, ja} A[i, a*2+ja] * B[a*2+ja, k]
    #
    # For fixed a, d, kd (and thus for fixed (a, d, kd)), we have:
    #   sum_{ja} A[i, a*2+ja] * B[a*2+ja, c*4+d*2+kd]
    #   This is a 2x2x2 multiplication for each (c, ...).
    #
    # Actually, let me just enumerate all 8 Strassen blocks:
    # Block (a, c, d) where a in {0,1} (A block), c in {0,1} (C column group), d in {0,1} (C sub-column)
    # Wait, that's 8 blocks. For each block:
    #   A_sub is A[:, a*2:(a+1)*2] (2x2)
    #   B_sub is B[a*2:(a+1)*2, c*4+d*2:(c*4+d*2+2)] (2x2)
    #   C_sub is C[:, c*4+d*2:(c*4+d*2+2)] (2x2)
    #
    # Yes! That works. 8 blocks, each 2x2x2, total rank 8*7 = 56.
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]  # 2*4 = 8 rows
    V = [[0] * rank for _ in range(m * p)]  # 4*8 = 32 rows
    W = [[0] * rank for _ in range(p * n)]  # 8*2 = 16 rows
    
    # Strassen U, V, W for 2x2x2 (local indices)
    # U: 4x7, indexed by [i*2+j, r]
    U_strassen = [
        [1, 0, 1, 0, 1, -1, 0],  # i=0,j=0 (A11)
        [0, 0, 0, 0, 1, 0, 1],   # i=0,j=1 (A12)
        [0, 1, 0, 0, 0, 1, 0],   # i=1,j=0 (A21)
        [1, 1, 0, 1, 0, 0, -1],  # i=1,j=1 (A22)
    ]
    V_strassen = [
        [1, 1, 0, -1, 0, 1, 0],  # j=0,k=0 (B11)
        [1, 0, 0, 1, 0, 1, 0],   # j=0,k=1 (B12)
        [0, 0, 1, 0, 0, 0, 1],   # j=1,k=0 (B21)
        [1, 0, -1, 0, 1, 0, 1],  # j=1,k=1 (B22)
    ]
    W_strassen = [
        [1, 0, 0, 1, -1, 0, 1],  # k=0,i=0 (C11)
        [0, 0, 1, 0, 1, 0, 0],   # k=1,i=0 (C12)
        [0, 1, 1, 1, 0, 0, 0],   # k=0,i=1 (C21)
        [1, -1, 1, 0, 0, 1, 0],  # k=1,i=1 (C22)
    ]
    
    for a in range(2):      # A block index (which 2x2 block of A)
        for c in range(2):  # C column group (left half or right half)
            for d in range(2):  # C sub-column within group
                col_offset = (a * 4 + c * 2 + d) * 7
                for r in range(7):
                    col = col_offset + r
                    # U: A rows i, cols a*2+ja
                    for i in range(2):
                        for ja in range(2):
                            val = U_strassen[i * 2 + ja][r]
                            if val != 0:
                                U[i * m + (a * 2 + ja)][col] += val
                    # V: B rows a*2+ja, cols c*4+d*2+kd
                    for ja in range(2):
                        for kd in range(2):
                            val = V_strassen[ja * 2 + kd][r]
                            if val != 0:
                                V[(a * 2 + ja) * p + (c * 4 + d * 2 + kd)][col] += val
                    # W: C cols c*4+d*2+kd, rows i
                    for i in range(2):
                        for kd in range(2):
                            val = W_strassen[kd * 2 + i][r]
                            if val != 0:
                                W[(c * 4 + d * 2 + kd) * n + i][col] += val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
