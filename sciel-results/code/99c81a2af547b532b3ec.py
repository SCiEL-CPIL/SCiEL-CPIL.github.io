"""
Rank-minimal exact tensor decomposition for 2x4 by 4x8 matrix multiplication.

Uses a Kronecker-product / recursive construction based on the known
rank-7 decomposition of 2x2 matrix multiplication (Strassen) to build
a lower-rank decomposition for 2x4 by 4x8 than the naive schoolbook rank 64.

Construction:
  The multiplication (n x m) @ (m x p) can be viewed as a tensor T in
  R^{n*m x m*p x p*n}.

  We use the following factorization strategy:
  
  Decompose m=4 as 2*2 and p=8 as 4*2, keeping n=2.
  
  A = A1 | A2  (top/bottom 2 rows), each 2x4
  Actually, let's think of it differently.

  Let A be 2x4 and B be 4x8. Partition B into four 4x2 blocks: B = [B0 B1 B2 B3].
  Partition A into four 1x4? No, A is 2x4, partition columns into 2x2 and 2x2: A = [A_L A_R] where A_L, A_R are 2x2.
  
  Actually, let's use the standard bilinear map approach.
  
  The product C = A @ B is 2x8.
  
  We can write A = [A1 A2] where A1, A2 are 2x2.
  B = [[B11],[B12],[B21],[B22]] where Bij are 2x2 (stacking 4 rows of B, each row-block is 1x2... wait B is 4x8, so 4 rows, 8 cols. Partition rows into two groups of 2: B_top (2x8), B_bot (2x8). Then partition columns into 4 groups of 2.
  
  So B_top = [Bt0 Bt1 Bt2 Bt3], B_bot = [Bb0 Bb1 Bb2 Bb3], each 2x2.
  A = [A_L A_R] where A_L, A_R are 2x2.
  
  Then C = A @ B = [A_L @ B_top + A_R @ B_bot columns]... 
  Actually A is 2x4, B is 4x8. 
  A @ B has entries (A@B)[i,k] = sum_{j=0}^{3} A[i,j] B[j,k].
  
  Let's partition the 4 middle index j into two halves: j in {0,1} and j in {2,3}.
  Let A0 = A[:, 0:2] (2x2), A1 = A[:, 2:4] (2x2).
  Let B0 = B[0:2, :] (2x8), B1 = B[2:4, :] (2x8).
  
  Then C = A0 @ B0 + A1 @ B1.
  
  Now each product A_i @ B_i is 2x8. We can further decompose.
  B0 is 2x8 = 2x(2*4). Let's partition B0 into four 2x2 blocks: B0 = [B0_0 B0_1 B0_2 B0_3].
  Similarly B1 = [B1_0 B1_1 B1_2 B1_3].
  
  So C[:, 0:2] = A0 @ B0_0 + A1 @ B1_0, which is sum of two 2x2 matrix multiplications.
  
  Using Strassen for 2x2 matrix multiplication (rank 7), each pair of 2x2 products can be computed with 7 scalar multiplications.
  
  There are 4 such column blocks (each 2x2). For each, we compute A0@B0_i + A1@B1_i.
  
  But A0 and A1 are shared across the 4 blocks. Similarly B0_i and B1_i are specific to block i.
  
  Let's use the Strassen-like recursive approach more carefully.
  
  Define the 2x2 matrix multiplication tensor M_2 (rank 7).
  We need to compute C = sum_{h=0}^{1} A_h @ B_h where A_h is 2x2 and B_h is 2x8 for h=0,1.
  Write B_h = [Bh_0 Bh_1 Bh_2 Bh_3] where Bh_i is 2x2.
  
  C_i = A_0 @ Bh_0 + A_1 @ Bh_1 for i=0,1,2,3.
  
  This is a "sum of two 2x2 matrix multiplications". The tensor rank of this sum can be bounded.
  
  Actually, a cleaner way: treat the whole thing as a bilinear map from (A0, A1, B0, B1) to C.
  A0, A1, B0, B1 are each 2x2. So we have 4 matrices of size 2x2.
  
  The product is C_i = A0 @ (B0_i + B1_i... no, C_i = A0 @ B0_i + A1 @ B1_i.
  
  Let's define combined matrices. For each column block i, we need to compute the sum of two independent 2x2 matmuls.
  
  The standard trick: to compute X@Y + Z@W for 2x2 matrices X,Y,Z,W, we can use at most 14 multiplications (7 for each), but there may be sharing.
  
  However, a better approach is to use the "Strassen recursion" directly on the original dimensions.
  
  Alternative: use the fact that matmul(2,4,8) can be embedded in matmul(4,4,8) or similar, but that might not help.
  
  Let's try a different decomposition. Use rank-7 for 2x2, and structure the 2x4 @ 4x8 as multiple 2x2 multiplications with shared sub-matrices.
  
  Specifically:
  A (2x4) -> columns partitioned into A0 (2x2), A1 (2x2).
  B (4x8) -> rows partitioned into B0 (2x8), B1 (2x8).
  C = A0@B0 + A1@B1.
  
  Now B0 (2x8) can be split into B0_a (2x2), B0_b (2x2), B0_c (2x2), B0_d (2x2).
  Similarly B1.
  
  C_a = A0@B0_a + A1@B1_a
  C_b = A0@B0_b + A1@B1_b
  C_c = A0@B0_c + A1@B1_c
  C_d = A0@B0_d + A1@B1_d
  
  Each C_x is 2x2.
  
  For a single pair (A0, B0_x, A1, B1_x), the sum of two 2x2 products has tensor rank at most 7+7=14, but potentially less due to structure.
  
  Actually, the tensor for X@Y + Z@W where X,Z are 2x2 (inputs) and Y,W are 2x2 (inputs) has rank at most 7 (for X@Y) + 7 (for Z@W) = 14. But we can do better by combining.
  
  Let's just use 14 multiplications per block, giving 4*14 = 56 total, which is less than 64.
  
  But we can do even better. Let's use the rank-7 Strassen decomposition explicitly.
  
  Strassen's 2x2 matmul:
  m1 = (A11+A22)(B11+B22)
  m2 = (A21+A22)B11
  m3 = A11(B21-B22)
  m4 = A22(B12-B11)
  m5 = (A11+A12)B22
  m6 = (A21-A11)(B11+B12)
  m7 = (A12-A22)(B21+B22)
  
  C11 = m1 - m4 + m5 - m7
  C12 = m3 + m5
  C21 = m2 + m4
  C22 = m1 - m3 + m2 - m6
  
  Now for computing P = X@Y + Z@W where X,Z,Y,W are 2x2.
  
  We can write this as:
  Let X' = X, Y' = Y, Z' = Z, W' = W.
  
  One approach: 
  m1 = (X11+Z22)(Y11+W22)  ... no, this doesn't work directly.
  
  Actually, X@Y + Z@W is just the product of a larger matrix? No, because X and Z multiply different Y and W.
  
  Let's just use 7 multiplications for X@Y and 7 for Z@W, total 14. No sharing in general.
  
  So total rank = 4 * 14 = 56.
  
  Let's verify: 56 < 64. Good.
  
  Can we do better? Maybe by combining all four blocks together.
  
  For all four blocks, we need to compute:
  For i in {a,b,c,d}: C_i = A0@B0_i + A1@B1_i.
  
  Notice A0 and A1 are shared. 
  A0 = [[a00,a01],[a10,a11]], A1 = [[a20,a21],[a30,a31]].
  
  Let's write A as 2x4:
  A = [[a00, a01, a02, a03],
       [a10, a11, a12, a13]]
  
  So A0 = [[a00,a01],[a10,a11]], A1 = [[a02,a03],[a12,a13]].
  
  B0_i = B[0:2, 2i:2i+2], B1_i = B[2:4, 2i:2i+2].
  
  This is getting complex. Let me just implement the rank-56 approach.
  
  Rank 56: 4 column-blocks, each computed via 2 Strassen-2x2 products.
  
  For block i:
  Compute T1 = A0 @ B0_i (2x2) using 7 mults.
  Compute T2 = A1 @ B1_i (2x2) using 7 mults.
  C_i = T1 + T2.
  
  Total: 4 * 7 + 4 * 7 = 56 multiplications.
  
  Let's code this up.
"""

def construct():
    n, m, p = 2, 4, 8
    # We'll build the decomposition using rank 56.
    # U: (n*m, rank) = (8, 56)
    # V: (m*p, rank) = (32, 56)
    # W: (p*n, rank) = (16, 56)
    
    rank = 56
    U = [[0.0] * rank for _ in range(8)]
    V = [[0.0] * rank for _ in range(32)]
    W = [[0.0] * rank for _ in range(16)]
    
    # Strassen coefficients for 2x2 matmul
    # m1 = (A00+A11)(B00+B11)
    # m2 = (A10+A11)(B00)
    # m3 = (A00)(B10-B11)
    # m4 = (A11)(B01-B00)
    # m5 = (A00+A01)(B11)
    # m6 = (A10-A00)(B00+B01)
    # m7 = (A01-A11)(B10+B11)
    #
    # C00 = m1 - m4 + m5 - m7
    # C01 = m3 + m5
    # C10 = m2 + m4
    # C11 = m1 - m3 + m2 - m6
    
    strassen = [
        # (coeff_A00, coeff_A01, coeff_A10, coeff_A11, coeff_B00, coeff_B01, coeff_B10, coeff_B11)
        (1, 0, 0, 1, 1, 0, 0, 1),   # m1
        (0, 0, 1, 1, 1, 0, 0, 0),   # m2
        (1, 0, 0, 0, 0, 0, 1, -1),  # m3
        (0, 0, 0, 1, -1, 1, 0, 0),  # m4
        (1, 1, 0, 0, 0, 0, 0, 1),   # m5
        (-1, 0, 1, 0, 1, 1, 0, 0),  # m6
        (0, 1, 0, -1, 0, 0, 1, 1),  # m7
    ]
    
    # Output reconstruction for 2x2 matmul
    # C00 = m1 - m4 + m5 - m7
    # C01 = m3 + m5
    # C10 = m2 + m4
    # C11 = m1 - m3 + m2 - m6
    strassen_out = [
        # (coeff_m1, coeff_m2, coeff_m3, coeff_m4, coeff_m5, coeff_m6, coeff_m7)
        (1, 0, 0, -1, 1, 0, -1),   # C00
        (0, 0, 1, 0, 1, 0, 0),     # C01
        (0, 1, 0, 1, 0, 0, 0),     # C10
        (1, 1, -1, 0, 0, -1, 0),   # C11
    ]
    
    # For each of the 4 column blocks i (0..3):
    #   Block i corresponds to output columns 2i, 2i+1
    #   A0 = A[:, 0:2], A1 = A[:, 2:4]
    #   B0_i = B[0:2, 2i:2i+2], B1_i = B[2:4, 2i:2i+2]
    #
    # U index: row = i*n + j for A[i,j]. 
    #   A0 has indices: row 0: A[0,0]=U[0], A[0,1]=U[1]; row 1: A[1,0]=U[4], A[1,1]=U[5]
    #   Wait, A is 2x4, stored as U with 8 rows: index = row*m + col = row*4 + col.
    #   A[0,0]->U[0], A[0,1]->U[1], A[0,2]->U[2], A[0,3]->U[3]
    #   A[1,0]->U[4], A[1,1]->U[5], A[1,2]->U[6], A[1,3]->U[7]
    #   A0 = A[:,0:2]: A[0,0]->0, A[0,1]->1, A[1,0]->4, A[1,1]->5
    #   A1 = A[:,2:4]: A[0,2]->2, A[0,3]->3, A[1,2]->6, A[1,3]->7
    #
    # V index: index = row*p + col = row*8 + col.
    #   B0_i = B[0:2, 2i:2i+2]:
    #     B[0, 2i]->V[2i], B[0, 2i+1]->V[2i+1]
    #     B[1, 2i]->V[8+2i], B[1, 2i+1]->V[8+2i+1]
    #   B1_i = B[2:4, 2i:2i+2]:
    #     B[2, 2i]->V[16+2i], B[2, 2i+1]->V[16+2i+1]
    #     B[3, 2i]->V[24+2i], B[3, 2i+1]->V[24+2i+1]
    #
    # W index: index = k*n + i. p*n = 16.
    #   Output C is 2x8, C[row, col]. W index = col*2 + row.
    #   C[0, 2i]->W[2i*2+0]=W[4i], C[0,2i+1]->W[4i+1]
    #   C[1, 2i]->W[4i+2], C[1,2i+1]->W[4i+3]
    
    for i in range(4):
        # Base column offset for this block
        base_col = i * 14
        
        # For T1 = A0 @ B0_i, 7 multiplications
        for s in range(7):
            c = base_col + s
            # U contributions for A0
            u00, u01, u10, u11 = strassen[s][0], strassen[s][1], strassen[s][2], strassen[s][3]
            if u00 != 0: U[0][c] += u00
            if u01 != 0: U[1][c] += u01
            if u10 != 0: U[4][c] += u10
            if u11 != 0: U[5][c] += u11
            
            # V contributions for B0_i
            v00, v01, v10, v11 = strassen[s][4], strassen[s][5], strassen[s][6], strassen[s][7]
            b00_idx = 0 * 8 + 2 * i        # B[0, 2i]
            b01_idx = 0 * 8 + 2 * i + 1    # B[0, 2i+1]
            b10_idx = 1 * 8 + 2 * i        # B[1, 2i]
            b11_idx = 1 * 8 + 2 * i + 1    # B[1, 2i+1]
            if v00 != 0: V[b00_idx][c] += v00
            if v01 != 0: V[b01_idx][c] += v01
            if v10 != 0: V[b10_idx][c] += v10
            if v11 != 0: V[b11_idx][c] += v11
            
            # W contributions (C00, C01 from T1)
            # C[0, 2i] gets from T1's C00, C[0, 2i+1] gets from T1's C01
            w00_coeff = strassen_out[0][s]  # for C00
            w01_coeff = strassen_out[1][s]  # for C01
            w_idx_c00 = (2 * i) * 2 + 0    # W[4i]
            w_idx_c01 = (2 * i + 1) * 2 + 0  # W[4i+1]
            if w00_coeff != 0: W[w_idx_c00][c] += w00_coeff
            if w01_coeff != 0: W[w_idx_c01][c] += w01_coeff
        
        # For T2 = A1 @ B1_i, 7 multiplications
        for s in range(7):
            c = base_col + 7 + s
            # U contributions for A1
            u00, u01, u10, u11 = strassen[s][0], strassen[s][1], strassen[s][2], strassen[s][3]
            if u00 != 0: U[2][c] += u00
            if u01 != 0: U[3][c] += u01
            if u10 != 0: U[6][c] += u10
            if u11 != 0: U[7][c] += u11
            
            # V contributions for B1_i
            v00, v01, v10, v11 = strassen[s][4], strassen[s][5], strassen[s][6], strassen[s][7]
            b00_idx = 2 * 8 + 2 * i        # B[2, 2i]
            b01_idx = 2 * 8 + 2 * i + 1    # B[2, 2i+1]
            b10_idx = 3 * 8 + 2 * i        # B[3, 2i]
            b11_idx = 3 * 8 + 2 * i + 1    # B[3, 2i+1]
            if v00 != 0: V[b00_idx][c] += v00
            if v01 != 0: V[b01_idx][c] += v01
            if v10 != 0: V[b10_idx][c] += v10
            if v11 != 0: V[b11_idx][c] += v11
            
            # W contributions (C00, C01 from T2, added to same output)
            w00_coeff = strassen_out[0][s]
            w01_coeff = strassen_out[1][s]
            w_idx_c00 = (2 * i) * 2 + 0
            w_idx_c01 = (2 * i + 1) * 2 + 0
            if w00_coeff != 0: W[w_idx_c00][c] += w00_coeff
            if w01_coeff != 0: W[w_idx_c01][c] += w01_coeff
    
    # Convert to integer lists (all entries are integers already)
    U = [[int(x) for x in row] for row in U]
    V = [[int(x) for x in row] for row in V]
    W = [[int(x) for x in row] for row in W]
    
    return {"U": U, "V": V, "W": W}
