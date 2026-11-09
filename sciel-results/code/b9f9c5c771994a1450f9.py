"""
Rank-44 decomposition for 3x4 * 4x4 matrix multiplication.

Uses a block-partitioned approach with Strassen for the 2x4 * 4x4 part:
- Partition A (3x4) into A1 (top 2 rows, 2x4) and A2 (bottom row, 1x4).
- Partition B (4x4) into B1 (left 2 cols, 4x2) and B2 (right 2 cols, 4x2).
- Output C = [A1*B1, A1*B2; A2*B1, A2*B2].

For A1*B (2x4 * 4x4):
  Partition A1 into [A1a | A1b] where A1a is 2x2 and A1b is 2x2.
  Partition B into 2x2 blocks: B = [[B11, B12], [B21, B22]] each 2x2.
  Use Strassen to compute 2x2 * 2x2 products.
  
  Specifically, A1*B1 where B1 is 4x2:
    A1 = [A1a | A1b] (2x4), B1 = [[B11_top], [B21_top]] where each is 2x2... 
    Actually, let's be more careful.

  A1 is 2x4, B is 4x4.
  Split A1 = [A_left | A_right] where A_left is 2x2 and A_right is 2x2.
  Split B = [[B_TL, B_TR], [B_BL, B_BR]] where each is 2x2.
  
  A1*B = A_left*B_TL + A_left*B_BL + A_right*B_TR + A_right*B_BR... 
  No, that's not right either. Let me think again.

  A1 (2x4) * B (4x4):
  Let A1 = [A_a | A_b] where A_a is 2x2 (cols 0,1) and A_b is 2x2 (cols 2,3).
  Let B = [[B_00, B_01], [B_10, B_11]] where each is 2x2.
  
  A1*B = [A_a | A_b] * [[B_00, B_01], [B_10, B_11]]
        = [A_a*B_00 + A_b*B_10, A_a*B_01 + A_b*B_11]
  
  Each 2x2 * 2x2 product has rank 4 (schoolbook) or rank 7 with Strassen... 
  wait, Strassen computes the product with 7 scalar multiplications, but the 
  decomposition rank for a single 2x2 matrix multiplication is 7.
  
  So we have 4 sub-products of type 2x2 * 2x2, each requiring rank 7 with Strassen.
  Total for A1*B: 4 * 7 = 28.
  
  But wait, we can do better. Instead of computing 4 separate 2x2*2x2 products,
  we can use the fact that we're computing A1*B which is 2x4 * 4x4.
  
  Actually, let's use a different partition. Split B into left 2 and right 2 columns.
  B_left is 4x2, B_right is 4x2.
  
  A1 * B_left: 2x4 * 4x2
  Split A1 = [A1_top | A1_bot] where each is 2x2 (rows 0,1 and cols 0,1 vs cols 2,3).
  Split B_left = [B_TL; B_BL] where B_TL is 2x2 (rows 0,1) and B_BL is 2x2 (rows 2,3).
  
  A1 * B_left = A1_top * B_TL + A1_bot * B_BL
  
  Each 2x2 * 2x2 has rank 7 with Strassen.
  So A1 * B_left has rank 7 + 7 = 14.
  Similarly A1 * B_right has rank 14.
  Total for A1 * B: 14 + 14 = 28.
  
  For A2 * B: 1x4 * 4x4.
  This is just a linear combination, rank 4 (one per column of B, or one per element).
  Actually, 1x4 * 4x4 = 1x4. Each output element is a dot product of length 4.
  Schoolbook rank = 4 * 4 = 16? No.
  
  1x4 * 4x4: output is 1x4. Each element C_j = sum_k A2_k * B_kj.
  Number of scalar multiplications = 4 * 4 = 16.
  But we can view this as: for each of the 4 output positions, we need 4 multiplications.
  Total rank = 4 (one for each element of A2, since B is fixed in the tensor).
  
  Wait, the tensor for 1x4 * 4x4 has dimensions 4 (for A) x 16 (for B) x 4 (for C).
  Rank of this tensor: each A element multiplies a row of B, producing a contribution to C.
  A_k * B_kj for j=0..3 gives C_j += A_k * B_kj.
  So for each k, we have a rank-1 term: U has 1 in position k, V has the entire k-th row of B... 
  no, V is fixed in the decomposition.
  
  The tensor T_{a, b, c} where a indexes A (4 elements), b indexes B (16 elements), c indexes C (4 elements).
  T_{a, (k,j), c} = 1 if a=k and c=j, else 0.
  This is a sum over a of e_a * (row a of B indices) * (identity for c=a... no).
  
  T = sum_{k=0}^{3} sum_{j=0}^{3} e_k (in A) * e_{(k,j)} (in B) * e_j (in C).
  This has rank 16 in schoolbook. But can we do better?
  
  The tensor is essentially the identity tensor reshaped. Rank = 4 * 4 = 16 for schoolbook.
  Is there a lower rank? For m x n matrix multiplication where one dimension is 1,
  the rank is m*n. So 1x4 * 4x4 has rank 16.
  
  Hmm, that gives total 28 + 16 = 44.
  
  Wait, let me double-check. A2 is 1x4, B is 4x4.
  C = A2 * B, which is 1x4.
  C_j = sum_{k=0}^{3} A2_k * B_{k,j}
  
  Tensor: T_{k, (k',j), j'} where the first index is for A2 (size 4), second for B (size 16), third for C (size 4).
  T_{k, (k',j), j'} = 1 if k=k' and j=j', else 0.
  
  This is a permutation tensor. Its rank equals the number of non-zero entries = 16.
  So rank is 16.
  
  Total: 28 + 16 = 44.
  
  Let me verify this is correct and construct it.
  
  For the Strassen part, I need to carefully construct the rank-7 decomposition for 2x2 * 2x2.
"""

import numpy as np

DIMENSIONS = [3, 4, 4]

def _strassen_2x2_decomposition():
    """
    Return U, V, W for 2x2 * 2x2 matrix multiplication with rank 7 (Strassen).
    U shape: (4, 7), V shape: (4, 7), W shape: (4, 7).
    Indexing: U[i*2+j, r] for A_{i,j}, V[i*2+j, r] for B_{i,j}, W[i*2+j, r] for C_{i,j}.
    """
    # Strassen's 7 multiplications for 2x2 * 2x2:
    # M1 = (a00 + a11)(b00 + b11)
    # M2 = (a10 + a11) * b00
    # M3 = a00 * (b01 - b11)
    # M4 = a11 * (b10 - b00)
    # M5 = (a00 + a01) * b11
    # M6 = (a10 - a00) * (b00 + b01)
    # M7 = (a01 - a11) * (b10 + b11)
    # 
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    
    # M1: (a00+a11)(b00+b11)
    U[0, 0] = 1  # a00
    U[3, 0] = 1  # a11
    V[0, 0] = 1  # b00
    V[3, 0] = 1  # b11
    W[0, 0] = 1  # C00
    W[3, 0] = 1  # C11
    
    # M2: (a10+a11) * b00
    U[2, 1] = 1  # a10
    U[3, 1] = 1  # a11
    V[0, 1] = 1  # b00
    W[2, 1] = 1  # C10
    W[3, 1] = -1 # C11
    
    # M3: a00 * (b01-b11)
    U[0, 2] = 1  # a00
    V[1, 2] = 1  # b01
    V[3, 2] = -1 # b11
    W[1, 2] = 1  # C01
    W[3, 2] = 1  # C11
    
    # M4: a11 * (b10-b00)
    U[3, 3] = 1  # a11
    V[2, 3] = 1  # b10
    V[0, 3] = -1 # b00
    W[0, 3] = 1  # C00
    W[2, 3] = 1  # C10
    
    # M5: (a00+a01) * b11
    U[0, 4] = 1  # a00
    U[1, 4] = 1  # a01
    V[3, 4] = 1  # b11
    W[1, 4] = 1  # C01
    W[0, 4] = -1 # C00
    
    # M6: (a10-a00) * (b00+b01)
    U[2, 5] = 1  # a10
    U[0, 5] = -1 # a00
    V[0, 5] = 1  # b00
    V[1, 5] = 1  # b01
    W[3, 5] = 1  # C11
    
    # M7: (a01-a11) * (b10+b11)
    U[1, 6] = 1  # a01
    U[3, 6] = -1 # a11
    V[2, 6] = 1  # b10
    V[3, 6] = 1  # b11
    W[0, 6] = 1  # C00
    W[3, 6] = 1  # C11
    
    return U, V, W


def construct():
    n, m, p = DIMENSIONS  # 3, 4, 4
    
    # Get Strassen decomposition for 2x2 * 2x2
    sU, sV, sW = _strassen_2x2_decomposition()
    
    # Total rank: 28 (for A1*B) + 16 (for A2*B) = 44
    rank = 44
    U = np.zeros((n * m, rank), dtype=int)  # 12 x 44
    V = np.zeros((m * p, rank), dtype=int)  # 16 x 44
    W = np.zeros((p * n, rank), dtype=int)  # 12 x 44
    
    # A1*B part: A1 is rows 0,1 of A (indices 0-7 in U), B is all of B
    # We split B into left 2 cols and right 2 cols
    # B_left is 4x2 (cols 0,1), B_right is 4x2 (cols 2,3)
    # 
    # A1 (2x4) * B_left (4x2):
    #   A1 = [A1L | A1R] where A1L is 2x2 (cols 0,1) and A1R is 2x2 (cols 2,3)
    #   B_left = [B_TL; B_BL] where B_TL is 2x2 (rows 0,1) and B_BL is 2x2 (rows 2,3)
    #   A1 * B_left = A1L * B_TL + A1R * B_BL
    #   Each term is 2x2 * 2x2, rank 7 with Strassen.
    #   Total: 14.
    #
    # A1 (2x4) * B_right (4x2):
    #   A1 = [A1L | A1R]
    #   B_right = [B_TR; B_BR] where B_TR is 2x2 (rows 0,1) and B_BR is 2x2 (rows 2,3)
    #   A1 * B_right = A1L * B_TR + A1R * B_BR
    #   Total: 14.
    #
    # Total for A1*B: 28.
    
    col_offset = 0
    
    # Term 1: A1L * B_TL (2x2 * 2x2), contributes to C[0:2, 0:2]
    # A1L is A[0:2, 0:2] -> U indices 0,1,4,5
    # B_TL is B[0:2, 0:2] -> V indices 0,1,8,9
    # C[0:2, 0:2] -> W indices 0,1,4,5 (since W[k*n+i] = W[k*3+i])
    for r in range(7):
        # Map A1L indices
        for a_idx in range(4):
            if sU[a_idx, r] != 0:
                # a_idx maps to A1L position
                # A1L: row 0 col 0 -> U[0], row 0 col 1 -> U[1], row 1 col 0 -> U[4], row 1 col 1 -> U[5]
                U_idx = a_idx  # 0,1,2,3 -> but A1L uses A[0,0],A[0,1],A[1,0],A[1,1] -> U[0],U[1],U[4],U[5]
                # sU uses indices 0=a00, 1=a01, 2=a10, 3=a11 for 2x2
                # A1L: a00=A[0,0]->U[0], a01=A[0,1]->U[1], a10=A[1,0]->U[4], a11=A[1,1]->U[5]
                U[a_idx if a_idx < 2 else a_idx + 3, col_offset + r] = sU[a_idx, r]
                
        # Map B_TL indices
        for b_idx in range(4):
            if sV[b_idx, r] != 0:
                # B_TL: b00=B[0,0]->V[0], b01=B[0,1]->V[1], b10=B[1,0]->V[8], b11=B[1,1]->V[9]
                # b_idx: 0=b00, 1=b01, 2=b10, 3=b11
                V_idx = b_idx if b_idx < 2 else b_idx + 7
                V[V_idx, col_offset + r] = sV[b_idx, r]
                
        # Map C indices
        for c_idx in range(4):
            if sW[c_idx, r] != 0:
                # C[0:2, 0:2]: C00->W[0], C01->W[1], C10->W[3], C11->W[4]
                # c_idx: 0=C00, 1=C01, 2=C10, 3=C11
                # W index: C[i,j] -> W[j*n+i] = W[j*3+i]
                # C00: i=0,j=0 -> W[0]
                # C01: i=0,j=1 -> W[3]
                # C10: i=1,j=0 -> W[1]
                # C11: i=1,j=1 -> W[4]
                if c_idx == 0:
                    W_idx = 0
                elif c_idx == 1:
                    W_idx = 3
                elif c_idx == 2:
                    W_idx = 1
                elif c_idx == 3:
                    W_idx = 4
                W[W_idx, col_offset + r] = sW[c_idx, r]
    
    col_offset += 7
    
    # Term 2: A1R * B_BL (2x2 * 2x2), contributes to C[0:2, 0:2]
    # A1R is A[0:2, 2:4] -> U indices 2,3,6,7
    # B_BL is B[2:4, 0:2] -> V indices 16,17,24,25
    # C[0:2, 0:2] -> W indices 0,1,4,5 (same as before)
    for r in range(7):
        for a_idx in range(4):
            if sU[a_idx, r] != 0:
                # A1R: a00=A[0,2]->U[2], a01=A[0,3]->U[3], a10=A[1,2]->U[6], a11=A[1,3]->U[7]
                U_idx = a_idx if a_idx < 2 else a_idx + 4
                U[U_idx, col_offset + r] = sU[a_idx, r]
                
        for b_idx in range(4):
            if sV[b_idx, r] != 0:
                # B_BL: b00=B[2,0]->V[16], b01=B[2,1]->V[17], b10=B[3,0]->V[24], b11=B[3,1]->V[25]
                V_idx = b_idx if b_idx < 2 else b_idx + 8
                V_idx = V_idx + 16
                V[V_idx, col_offset + r] = sV[b_idx, r]
                
        for c_idx in range(4):
            if sW[c_idx, r] != 0:
                if c_idx == 0:
                    W_idx = 0
                elif c_idx == 1:
                    W_idx = 3
                elif c_idx == 2:
                    W_idx = 1
                elif c_idx == 3:
                    W_idx = 4
                W[W_idx, col_offset + r] = sW[c_idx, r]
    
    col_offset += 7
    
    # Term 3: A1L * B_TR (2x2 * 2x2), contributes to C[0:2, 2:4]
    # A1L is A[0:2, 0:2] -> U indices 0,1,4,5
    # B_TR is B[0:2, 2:4] -> V indices 2,3,10,11
    # C[0:2, 2:4] -> W indices: C02->W[6], C03->W[7], C12->W[9], C13->W[10]
    for r in range(7):
        for a_idx in range(4):
            if sU[a_idx, r] != 0:
                U_idx = a_idx if a_idx < 2 else a_idx + 3
                U[U_idx, col_offset + r] = sU[a_idx, r]
                
        for b_idx in range(4):
            if sV[b_idx, r] != 0:
                # B_TR: b00=B[0,2]->V[2], b01=B[0,3]->V[3], b10=B[1,2]->V[10], b11=B[1,3]->V[11]
                V_idx = b_idx if b_idx < 2 else b_idx + 8
                V[V_idx, col_offset + r] = sV[b_idx, r]
                
        for c_idx in range(4):
            if sW[c_idx, r] != 0:
                if c_idx == 0:
                    W_idx = 6
                elif c_idx == 1:
                    W_idx = 7
                elif c_idx == 2:
                    W_idx = 9
                elif c_idx == 3:
                    W_idx = 10
                W[W_idx, col_offset + r] = sW[c_idx, r]
    
    col_offset += 7
    
    # Term 4: A1R * B_BR (2x2 * 2x2), contributes to C[0:2, 2:4]
    # A1R is A[0:2, 2:4] -> U indices 2,3,6,7
    # B_BR is B[2:4, 2:4] -> V indices 18,19,26,27
    # C[0:2, 2:4] -> W indices: C02->W[6], C03->W[7], C12->W[9], C13->W[10]
    for r in range(7):
        for a_idx in range(4):
            if sU[a_idx, r] != 0:
                U_idx = a_idx if a_idx < 2 else a_idx + 4
                U[U_idx, col_offset + r] = sU[a_idx, r]
                
        for b_idx in range(4):
            if sV[b_idx, r] != 0:
                # B_BR: b00=B[2,2]->V[18], b01=B[2,3]->V[19], b10=B[3,2]->V[26], b11=B[3,3]->V[27]
                V_idx = b_idx if b_idx < 2 else b_idx + 8
                V_idx = V_idx + 18
                V[V_idx, col_offset + r] = sV[b_idx, r]
                
        for c_idx in range(4):
            if sW[c_idx, r] != 0:
                if c_idx == 0:
                    W_idx = 6
                elif c_idx == 1:
                    W_idx = 7
                elif c_idx == 2:
                    W_idx = 9
                elif c_idx == 3:
                    W_idx = 10
                W[W_idx, col_offset + r] = sW[c_idx, r]
    
    col_offset += 7  # Now col_offset = 28
    
    # A2*B part: A2 is row 2 of A (1x4), B is 4x4, C is row 2 (1x4)
    # A2_k * B_{k,j} contributes to C_j
    # Schoolbook: for each k in 0..3, j in 0..3: rank 1 term
    # Total: 16 terms
    for k in range(4):
        for j in range(4):
            U[2 * m + k, col_offset] = 1  # A[2, k] -> U[2*4 + k] = U[8+k]
            V[k * p + j, col_offset] = 1  # B[k, j] -> V[k*4 + j]
            W[j * n + 2, col_offset] = 1  # C[2, j] -> W[j*3 + 2]
            col_offset += 1
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
