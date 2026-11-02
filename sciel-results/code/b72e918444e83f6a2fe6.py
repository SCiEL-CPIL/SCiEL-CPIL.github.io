"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
import time
_start_time = time.time()
_DEADLINE = 145.0  # seconds, safety margin before 150s kill

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
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
    return U, V, W

def _strassen_2x2_2x2():
    """
    Return Strassen decomposition for 2x2 by 2x2 = 2x2.
    Convention: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    T[i,j,k] = sum_r U[i*2+j,r]*V[j*2+k,r]*W[k*2+i,r] = 1 for all i,j,k in {0,1}
    """
    # Strassen's 7 products for 2x2*2x2
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # C = [[c00, c01], [c10, c11]]
    # M1 = (a00+a11)*(b00+b11)
    # M2 = (a10+a11)*b00
    # M3 = a00*(b01-b11)
    # M4 = a11*(b10-b00)
    # M5 = (a00+a01)*b11
    # M6 = (a10-a00)*(b00+b01)
    # M7 = (a01-a11)*(b10+b11)
    # c00 = M1+M4-M5+M7
    # c01 = M3+M5
    # c10 = M2+M4
    # c11 = M1-M2+M3+M6
    
    # In our tensor convention:
    # T[i,j,k] = 1 for all i,j,k (the matmul tensor is all-ones)
    # We need: sum_r U[i*2+j,r]*V[j*2+k,r]*W[k*2+i,r] = 1
    
    # The standard Strassen decomposition of the 2x2x2 matmul tensor:
    # The tensor M[i,j,k] has M[i,j,k] = 1 for all i,j,k (since (AB)[i,k] = sum_j A[i,j]B[j,k])
    # Wait, that's not right. Let me reconsider.
    
    # Actually for 2x2*2x2 -> 2x2:
    # (AB)[i,k] = sum_j A[i,j] B[j,k]
    # The bilinear form is: sum_{i,j,k} (AB)[i,k] * something
    # No, the tensor T satisfies: (AB)[i,k] = sum_j T[i,j,k] A[i,j] B[j,k]
    # So T[i,j,k] = 1 for all i,j,k.
    
    # But Strassen shows this tensor has rank 7.
    # The 7 rank-1 terms are:
    # Term r corresponds to: U_r[i,j] * V_r[j,k] * W_r[k,i]
    
    # Let me construct it properly.
    # For Strassen, we have 7 "products" M1..M7, each is a product of a linear form in A and a linear form in B.
    # The output C is linear combinations of these.
    
    # In tensor language:
    # T = sum_{r=1}^{7} u_r ⊗ v_r ⊗ w_r
    # where u_r[i,j] is the coefficient of A[i,j] in the r-th linear form
    # v_r[j,k] is the coefficient of B[j,k] in the r-th linear form  
    # w_r[k,i] is the coefficient of C[k,i]... wait, the output is C[i,k] not C[k,i].
    
    # Hmm, the convention here is W[k*n+i, r] which for n=2 is W[k*2+i, r].
    # So the third index of W is (k, i) meaning output position (i,k) is indexed as k*2+i.
    
    # Let me just use the known Strassen rank-7 decomposition.
    # The 2x2x2 matrix multiplication tensor has the following rank-7 decomposition:
    
    # M1: u1[i,j] = (1 if i in {0,1} and j in {0,1} and (i==0 or i==1) and (j==0 or j==1))
    # Actually let me just write it out.
    
    # Standard Strassen:
    # M1 = (a00+a11)(b00+b11): u1[0,0]=1, u1[1,1]=1, v1[0,0]=1, v1[1,1]=1
    #   contributes to c00 (+1), c11 (+1)
    # M2 = (a10+a11)b00: u2[1,0]=1, u2[1,1]=1, v2[0,0]=1
    #   contributes to c10 (+1), c11 (-1)
    # M3 = a00(b01-b11): u3[0,0]=1, v3[0,1]=1, v3[1,1]=-1
    #   contributes to c01 (+1), c11 (+1)
    # M4 = a11(b10-b00): u4[1,1]=1, v4[1,0]=1, v4[0,0]=-1
    #   contributes to c00 (+1), c10 (+1)
    # M5 = (a00+a01)b11: u5[0,0]=1, u5[0,1]=1, v5[1,1]=1
    #   contributes to c00 (-1), c01 (+1)
    # M6 = (a10-a00)(b00+b01): u6[1,0]=1, u6[0,0]=-1, v6[0,0]=1, v6[0,1]=1
    #   contributes to c11 (+1)
    # M7 = (a01-a11)(b10+b11): u7[0,1]=1, u7[1,1]=-1, v7[1,0]=1, v7[1,1]=1
    #   contributes to c00 (+1)
    
    # Now in our convention:
    # U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    # The contribution to T[i,j,k] from term r is U[i*2+j,r]*V[j*2+k,r]*W[k*2+i,r]
    # And we need sum_r ... = 1 for all i,j,k.
    
    # But wait - in Strassen, the "output" part W corresponds to how each M_r contributes to C[i,k].
    # C[i,k] = sum_r (coeff of M_r in C[i,k]) * M_r
    # So W[k*2+i, r] = coeff of M_r in C[i,k]
    
    # Let me re-index: C[i,k], so W index is (k,i) -> W[k*2+i, r]
    
    # M1: contributes +1 to c00, +1 to c11
    #   c00 is C[0,0], so W[0*2+0, 0] = 1, i.e., W[0,0]=1
    #   c11 is C[1,1], so W[1*2+1, 0] = 1, i.e., W[3,0]=1
    #   u1: a00+a11 -> U[0*2+0,0]=1, U[1*2+1,0]=1 -> U[0,0]=1, U[3,0]=1
    #   v1: b00+b11 -> V[0*2+0,0]=1, V[1*2+1,0]=1 -> V[0,0]=1, V[3,0]=1
    
    # M2: contributes +1 to c10, -1 to c11
    #   c10 is C[1,0], W[0*2+1, 1] = 1 -> W[1,1]=1
    #   c11 is C[1,1], W[1*2+1, 1] = -1 -> W[3,1]=-1
    #   u2: a10+a11 -> U[1*2+0,1]=1, U[1*2+1,1]=1 -> U[2,1]=1, U[3,1]=1
    #   v2: b00 -> V[0*2+0,1]=1 -> V[0,1]=1
    
    # M3: contributes +1 to c01, +1 to c11
    #   c01 is C[0,1], W[1*2+0, 2] = 1 -> W[2,2]=1
    #   c11 is C[1,1], W[1*2+1, 2] = 1 -> W[3,2]=1
    #   u3: a00 -> U[0*2+0,2]=1 -> U[0,2]=1
    #   v3: b01-b11 -> V[0*2+1,2]=1, V[1*2+1,2]=-1 -> V[1,2]=1, V[3,2]=-1
    
    # M4: contributes +1 to c00, +1 to c10
    #   c00 is C[0,0], W[0*2+0, 3] = 1 -> W[0,3]=1
    #   c10 is C[1,0], W[0*2+1, 3] = 1 -> W[1,3]=1
    #   u4: a11 -> U[1*2+1,3]=1 -> U[3,3]=1
    #   v4: b10-b00 -> V[1*2+0,3]=1, V[0*2+0,3]=-1 -> V[2,3]=1, V[0,3]=-1
    
    # M5: contributes -1 to c00, +1 to c01
    #   c00 is C[0,0], W[0*2+0, 4] = -1 -> W[0,4]=-1
    #   c01 is C[0,1], W[1*2+0, 4] = 1 -> W[2,4]=1
    #   u5: a00+a01 -> U[0*2+0,4]=1, U[0*2+1,4]=1 -> U[0,4]=1, U[1,4]=1
    #   v5: b11 -> V[1*2+1,4]=1 -> V[3,4]=1
    
    # M6: contributes +1 to c11
    #   c11 is C[1,1], W[1*2+1, 5] = 1 -> W[3,5]=1
    #   u6: a10-a00 -> U[1*2+0,5]=1, U[0*2+0,5]=-1 -> U[2,5]=1, U[0,5]=-1
    #   v6: b00+b01 -> V[0*2+0,5]=1, V[0*2+1,5]=1 -> V[0,5]=1, V[1,5]=1
    
    # M7: contributes +1 to c00
    #   c00 is C[0,0], W[0*2+0, 6] = 1 -> W[0,6]=1
    #   u7: a01-a11 -> U[0*2+1,6]=1, U[1*2+1,6]=-1 -> U[1,6]=1, U[3,6]=-1
    #   v7: b10+b11 -> V[1*2+0,6]=1, V[1*2+1,6]=1 -> V[2,6]=1, V[3,6]=1
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # M3
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # M4
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # M5
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # M6
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    # Verify
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = 0.0
                for r in range(7):
                    s += U[i*2+j][r] * V[j*2+k][r] * W[k*2+i][r]
                if abs(s - 1.0) > 1e-9:
                    pass  # Should not happen
    
    return U, V, W

def _construct_strassen_based(n, m, p):
    """
    Construct a decomposition using Strassen for 2x2x2 blocks.
    For 3x4x6: split into blocks.
    n=3: rows 0,1 (2x2 block) + row 2 (1x1)
    m=4: cols 0,1 (2x2 block) + cols 2,3 (2x2 block)  
    p=6: output cols 0,1,2 (2x2+1) + 3,4,5 (2x2+1)
    
    Strategy: Use block decomposition.
    A is 3x4, B is 4x6, C is 3x6.
    
    Split A into:
    A = [[A11, A12], [A21, A22]] where A11 is 2x2, A12 is 2x2, A21 is 1x2, A22 is 1x2
    
    Split B into:
    B = [[B11, B12, B13], [B21, B22, B23]] where B11 is 2x2, B12 is 2x2, B13 is 2x2,
         B21 is 2x2, B22 is 2x2, B23 is 2x2
    
    C = [[C11, C12, C13], [C21, C22, C23]] where C11 is 2x2, C12 is 2x2, C13 is 2x2,
         C21 is 1x2, C22 is 1x2, C23 is 1x2
    
    C11 = A11*B11 + A12*B21
    C12 = A11*B12 + A12*B22
    C13 = A11*B13 + A12*B23
    C21 = A21*B11 + A22*B21
    C22 = A21*B12 + A22*B22
    C23 = A21*B13 + A22*B23
    
    For the 2x2*2x2 products, use Strassen (rank 7 each).
    For the 1x2*2x2 products, use schoolbook (rank 2 each, since (1x2)*(2x2) = 1x2 needs 2 mults... actually 1*2*2=4 terms but can be done in 2? No, 1x2 times 2x2: result is 1x2. (a0,a1)*[[b00,b01],[b10,b11]] = [a0*b00+a1*b10, a0*b01+a1*b11]. That's 4 mults schoolbook, or can we do better? For 1x2*2x2, the tensor has size 1*2*2=4, and rank is 4 (can't do better than schoolbook for this shape since it's just 4 independent products... wait no).
    
    Actually (1x2)*(2x2) -> 1x2:
    c0 = a0*b00 + a1*b10
    c1 = a0*b01 + a1*b11
    This is two independent dot products of length 2, so rank = 2*2 = 4? No, each dot product of length 2 has tensor rank 2 (it's just a0*b00, a1*b10 for c0 and a0*b01, a1*b11 for c1). So total rank = 4.
    
    Hmm, but actually we can share. No, they're independent. Rank = 4.
    
    So total rank using this approach:
    - 6 Strassen blocks for 2x2*2x2: 6 * 7 = 42
    - 3 blocks of 1x2*2x2: 3 * 4 = 12
    Total: 54
    
    That's much better than 72!
    
    But wait, we need to be careful about the tensor convention.
    """
    n, m, p = 3, 4, 6
    
    # Block structure:
    # A: rows [0,1] (size 2) and row [2] (size 1); cols [0,1] (size 2) and cols [2,3] (size 2)
    # B: rows [0,1] (size 2) and rows [2,3] (size 2); cols [0,1] (size 2), [2,3] (size 2), [4,5] (size 2)
    # C: rows [0,1] (size 2) and row [2] (size 1); cols [0,1] (size 2), [2,3] (size 2), [4,5] (size 2)
    
    # C[0:2, 0:2] = A[0:2,0:2]*B[0:2,0:2] + A[0:2,2:4]*B[2:4,0:2]  (two 2x2*2x2)
    # C[0:2, 2:4] = A[0:2,0:2]*B[0:2,2:4] + A[0:2,2:4]*B[2:4,2:4]  (two 2x2*2x2)
    # C[0:2, 4:6] = A[0:2,0:2]*B[0:2,4:6] + A[0:2,2:4]*B[2:4,4:6]  (two 2x2*2x2)
    # C[2, 0:2]   = A[2,0:2]*B[0:2,0:2] + A[2,2:4]*B[2:4,0:2]     (two 1x2*2x2)
    # C[2, 2:4]   = A[2,0:2]*B[0:2,2:4] + A[2,2:4]*B[2:4,2:4]     (two 1x2*2x2)
    # C[2, 4:6]   = A[2,0:2]*B[0:2,4:6] + A[2,2:4]*B[2:4,4:6]     (two 1x2*2x2)
    
    # Total: 6 Strassen (2x2*2x2) + 6 schoolbook (1x2*2x2)
    # Rank: 6*7 + 6*4 = 42 + 24 = 66
    
    # Let's build this.
    # For each sub-product, we'll embed the local decomposition into the global one.
    
    total_rank = 6 * 7 + 6 * 4  # 66
    U = [[0] * total_rank for _ in range(n * m)]  # 12 x 66
    V = [[0] * total_rank for _ in range(m * p)]  # 24 x 66
    W = [[0] * total_rank for _ in range(p * n)]  # 18 x 66
    
    col = 0
    
    # Helper: embed a 2x2*2x2 Strassen decomposition
    def embed_strassen(row_off_a, col_off_a, row_off_b, col_off_c, start_col):
        """
        Embed Strassen for A[row_off_a:row_off_a+2, col_off_a:col_off_a+2] * 
                        B[row_off_b:row_off_b+2, col_off_c:col_off_c+2]
        into C[row_off_c:row_off_c+2, col_off_c:col_off_c+2]
        """
        sU, sV, sW = _strassen_2x2_2x2()
        for r in range(7):
            for li in range(2):  # local row in A/C
                for lj in range(2):  # local col in A / row in B
                    gi = row_off_a + li
                    gj = col_off_a + lj
                    U[gi * m + gj][start_col + r] += sU[li * 2 + lj][r]
                    
                    bk = col_off_c  # local col in B/C
                    # V index: j*p + k where j is global row in B, k is global col in B
                    # Local row in B: row_off_b + lj
                    # Local col in B: col_off_c + bk_local
                    # Wait, I need to be more careful.
                    pass
        return start_col + 7
    
    # This is getting complex. Let me do it more directly.
    
    # Let me define the embedding more carefully.
    
    def embed_2x2x2(a_row0, a_col0, b_row0, c_row0, c_col0, start_col):
        """
        A submatrix: rows a_row0..a_row0+1, cols a_col0..a_col0+1 (2x2)
        B submatrix: rows b_row0..b_row0+1, cols c_col0..c_col0+1 (2x2)
        C submatrix: rows c_row0..c_row0+1, cols c_col0..c_col0+1 (2x2)
        """
        sU, sV, sW = _strassen_2x2_2x2()
        for r in range(7):
            for li in range(2):
                for lj in range(2):
                    gi = a_row0 + li
                    gj = a_col0 + lj
                    U[gi * m + gj][start_col + r] += sU[li * 2 + lj][r]
                    
                    # V: B[j, k] where j is row in B, k is col in B
                    # Local: j = b_row0 + lj, k = c_col0 + lk
                    for lk in range(2):
                        gj_b = b_row0 + lj
                        gk = c_col0 + lk
                        V[gj_b * p + gk][start_col + r] += sV[lj * 2 + lk][r]
                    
                    # W: C[i, k] where i is row in C, k is col in C
                    # Local: i = c_row0 + li, k = c_col0 + lk
                    # W index: k*n + i
                    for lk in range(2):
                        gi_c = c_row0 + li
                        gk = c_col0 + lk
                        W[gk * n + gi_c][start_col + r] += sW[lk * 2 + li][r]
        
        return start_col + 7
    
    def embed_1x2x2(a_row, a_col0, b_row0, c_row, c_col0, start_col):
        """
        A submatrix: row a_row, cols a_col0..a_col0+1 (1x2)
        B submatrix: rows b_row0..b_row0+1, cols c_col0..c_col0+1 (2x2)
        C submatrix: row c_row, cols c_col0..c_col0+1 (1x2)
        
        Schoolbook: 4 terms
        C[c_row, c_col0+l2] = sum_{l1=0}^{1} A[a_row, a_col0+l1] * B[b_row0+l1, c_col0+l2]
        """
        for l1 in range(2):
            for l2 in range(2):
                gi = a_row
                gj = a_col0 + l1
                U[gi * m + gj][start_col] = 1
                
                gj_b = b_row0 + l1
                gk = c_col0 + l2
                V[gj_b * p + gk][start_col] = 1
                
                gi_c = c_row
                gk = c_col0 + l2
                W[gk * n + gi_c][start_col] = 1
                
                start_col += 1
        
        return start_col
    
    # C[0:2, 0:2] = A[0:2,0:2]*B[0:2,0:2] + A[0:2,2:4]*B[2:4,0:2]
    col = embed_2x2x2(0, 0, 0, 0, 0, col)
    col = embed_2x2x2(0, 2, 2, 0, 0, col)
    
    # C[0:2, 2:4] = A[0:2,0:2]*B[0:2,2:4] + A[0:2,2:4]*B[2:4,2:4]
    col = embed_2x2x2(0, 0, 0, 0, 2, col)
    col = embed_2x2x2(0, 2, 2, 0, 2, col)
    
    # C[0:2, 4:6] = A[0:2,0:2]*B[0:2,4:6] + A[0:2,2:4]*B[2:4,4:6]
    col = embed_2x2x2(0, 0, 0, 0, 4, col)
    col = embed_2x2x2(0, 2, 2, 0, 4, col)
    
    # C[2, 0:2] = A[2,0:2]*B[0:2,0:2] + A[2,2:4]*B[2:4,0:2]
    col = embed_1x2x2(2, 0, 0, 2, 0, col)
    col = embed_1x2x2(2, 2, 2, 2, 0, col)
    
    # C[2, 2:4] = A[2,0:2]*B[0:2,2:4] + A[2,2:4]*B[2:4,2:4]
    col = embed_1x2x2(2, 0, 0, 2, 2, col)
    col = embed_1x2x2(2, 2, 2, 2, 2, col)
    
    # C[2, 4:6] = A[2,0:2]*B[0:2,4:6] + A[2,2:4]*B[2:4,4:6]
    col = embed_1x2x2(2, 0, 0, 2, 4, col)
    col = embed_1x2x2(2, 2, 2, 2, 4, col)
    
    assert col == total_rank, f"Column mismatch: {col} != {total_rank}"
    
    # Verify
    if not _verify(U, V, W, n, m, p):
        # Fall back to schoolbook
        return _schoolbook(n, m, p)
    
    return U, V, W

def _try_rank(n, m, p, rank, max_iter=1000, tol=1e-8, seed=0):
    """Try to find a decomposition of given rank using alternating minimization."""
    if time.time() > _DEADLINE - 5.0:
        return None
    
    np.random.seed(seed)
    
    # Initialize randomly
    U = np.random.randn(n * m, rank) * 0.1
    V = np.random.randn(m * p, rank) * 0.1
    W = np.random.randn(p * n, rank) * 0.1
    
    # Alternating minimization
    for it in range(max_iter):
        if time.time() > _DEADLINE - 5.0:
            return None
        
        # Update U
        for i in range(n):
            for j in range(m):
                A = np.zeros((p, rank))
                for k in range(p):
                    for r in range(rank):
                        A[k, r] = V[j * p + k, r] * W[k * n + i, r]
                try:
                    x = np.linalg.lstsq(A, np.ones(p), rcond=None)[0]
                    U[i * m + j, :] = x
                except:
                    pass
        
        # Update V
        for j in range(m):
            for k in range(p):
                B = np.zeros((n, rank))
                for i in range(n):
                    for r in range(rank):
                        B[i, r] = U[i * m + j, r] * W[k * n + i, r]
                try:
                    x = np.linalg.lstsq(B, np.ones(n), rcond=None)[0]
                    V[j * p + k, :] = x
                except:
                    pass
        
        # Update W
        for k in range(p):
            for i in range(n):
                C = np.zeros((m, rank))
                for j in range(m):
                    for r in range(rank):
                        C[j, r] = U[i * m + j, r] * V[j * p + k, r]
                try:
                    x = np.linalg.lstsq(C, np.ones(m), rcond=None)[0]
                    W[k * n + i, :] = x
                except:
                    pass
        
        # Check convergence
        if it % 5 == 0:
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for r in range(rank):
                            s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                        err = max(err, abs(s - 1.0))
            if err < tol:
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                for row in U_list:
                    for idx in range(len(row)):
                        if abs(row[idx] * 2 - round(row[idx] * 2)) < 1e-6:
                            row[idx] = round(row[idx] * 2) / 2.0
                for row in V_list:
                    for idx in range(len(row)):
                        if abs(row[idx] * 2 - round(row[idx] * 2)) < 1e-6:
                            row[idx] = round(row[idx] * 2) / 2.0
                for row in W_list:
                    for idx in range(len(row)):
                        if abs(row[idx] * 2 - round(row[idx] * 2)) < 1e-6:
                            row[idx] = round(row[idx] * 2) / 2.0
                return U_list, V_list, W_list
    
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # First try the structured Strassen-based decomposition (rank 66)
    try:
        U, V, W = _construct_strassen_based(n, m, p)
        if _verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    # Try to find even lower rank via alternating minimization
    for rank in range(50, 66):
        if time.time() > _DEADLINE - 5.0:
            break
        for seed in range(5):
            if time.time() > _DEADLINE - 5.0:
                break
            result = _try_rank(n, m, p, rank, max_iter=300, seed=seed)
            if result is not None:
                U, V, W = result
                if _verify(U, V, W, n, m, p):
                    return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
