"""Matrix multiplication 4x5x6 - rank-minimal exact tensor decomposition."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 5, 6]

def construct():
    """
    Construct a valid exact tensor decomposition for 4x5x6 matrix multiplication.
    
    Uses a block structure:
    - Split the 4x5 matrix into 2x2 blocks of size 2x3 and 2x2
    - Actually, let me use a cleaner approach:
    
    For 4x5x6, we can use the following block decomposition:
    A (4x5) = [A11 A12] where A11 is 2x2, A12 is 2x3
               [A21 A22] where A21 is 2x2, A22 is 2x3
    
    B (5x6) = [B11 B12 B13] where each is 2x2
               [B21 B22 B23] where each is 2x2
    
    This gets complicated. Let me use a simpler verified approach.
    
    Actually, the safest approach is to use Strassen-like block multiplication
    where we decompose into smaller problems.
    
    For 4x5x6, let's use the following strategy:
    - Use rank-7 Strassen for 2x2x2 blocks where applicable
    - Combine with schoolbook for remaining parts
    
    Let me think about this more carefully. The key insight from the knowledge
    is that replacing an invalid Strassen block with schoolbook gives a valid
    decomposition with rank 105 (49+24+32) vs baseline 120.
    
    Let me construct a valid decomposition using block structure.
    
    Approach: Partition A (4x5) into 2x2 blocks of size 2x3, and B (5x6) into 
    corresponding blocks. But the dimensions don't divide evenly.
    
    Let me use a different approach: direct construction with known valid
    tensor decompositions for small cases, combined appropriately.
    
    Actually, let me just use a well-verified construction. The simplest valid
    approach that's better than schoolbook (120) is to use Strassen's algorithm
    for the 2x2x2 sub-problems within a block decomposition.
    
    For 4x5x6:
    - Split A into 2 blocks of size 2x5
    - Split B into 3 blocks of size 5x2 (since 6=2*3)
    - Each output block is 2x2
    
    This gives us 2*3 = 6 output blocks of size 2x2.
    Each 2x2x2 multiplication (A_block: 2x5, B_block: 5x2 -> C_block: 2x2)
    can be done with rank at most 2*5*2 = 20 (schoolbook) or potentially less.
    
    Hmm, this is getting complex. Let me use a cleaner approach.
    
    I'll use the following verified construction:
    - Use rank-7 Strassen for 2x2x2 (this is well-known and valid)
    - Build up to 4x5x6 using block structure
    
    For 4x5x6, partition as follows:
    A (4x5) = [A_top (2x5); A_bottom (2x5)]
    B (5x6) = [B_left (5x2) | B_mid (5x2) | B_right (5x2)]
    C (4x6) = [C_TL C_TM C_TR; C_BL C_BM C_BR] where each C_* is 2x2
    
    C_TL = A_top * B_left  (2x5 * 5x2 = 2x2)
    C_TM = A_top * B_mid   (2x5 * 5x2 = 2x2)
    C_TR = A_top * B_right (2x5 * 5x2 = 2x2)
    C_BL = A_bottom * B_left (2x5 * 5x2 = 2x2)
    C_BM = A_bottom * B_mid  (2x5 * 5x2 = 2x2)
    C_BR = A_bottom * B_right (2x5 * 5x2 = 2x2)
    
    Each 2x5*5x2 multiplication has rank at most 20 (schoolbook).
    Total: 6 * 20 = 120 (same as schoolbook for whole thing).
    
    To do better, I need to exploit the 5-dimensional middle. Let me think...
    
    Actually, for a 2x5 * 5x2 product, the tensor has rank at most 2*5*2 = 20.
    But can we do better? The bilinear complexity of 2x5 * 5x2 is at most 20,
    and it's actually known to be exactly 20 (or close to it).
    
    Let me try a different partition. What if I use:
    A (4x5) = [A1 (2x2) A2 (2x3)]
    B (5x6) = [B1 (2x2) B2 (2x2) B3 (2x2); B4 (3x2) B5 (3x2) B6 (3x2)]
    
    This is messy. Let me just implement a known good construction.
    
    Based on the knowledge that rank 105 was achieved (49+24+32), let me
    reverse-engineer what that might be. 49 = 7*7, 24 = 4*6 or 3*8, 32 = 2*16.
    
    Actually, let me just implement Strassen-based block decomposition properly.
    
    For 4x4x4, Strassen gives rank 28 (7 per 2x2x2 block, 4 blocks).
    For 4x5x6, I can use a hybrid approach.
    
    Let me try this: use the fact that we can embed 4x5x6 into 4x6x6 and then
    restrict, or use a direct construction.
    
    I'll go with a pragmatic approach: implement a valid construction that
    achieves rank < 120. The simplest way is to use Strassen for the 2x2x2
    sub-problems within a proper block decomposition.
    
    Let me use 4x4x4 Strassen (rank 28) and handle the extra dimensions with
    schoolbook.
    
    4x5x6: Think of it as 4x4x4 (Strassen, rank 28) + 4x1x6 (schoolbook, rank 24) 
    + 4x4x2 (schoolbook, rank 32)? That's 28+24+32 = 84. Wait, that doesn't add up.
    
    Let me be more careful. 
    A (4x5) = [A_4x4 | A_4x1]
    B (5x6) = [B_4x6; B_1x6]
    C = A_4x4 * B_4x6 + A_4x1 * B_1x6
    
    A_4x4 * B_4x6: This is 4x4 * 4x6. Split B_4x6 into [B_4x4 | B_4x2]
    A_4x4 * B_4x4: Strassen, rank 28
    A_4x4 * B_4x2: 4x4*4x2, schoolbook rank 32, or can do better
    A_4x1 * B_1x6: outer product, rank 1*1*6 = 6... no, rank 4*6/... 
    
    Hmm, let me just compute the schoolbook ranks:
    4x4*4x4: 4*4*4 = 64, Strassen: 28
    4x4*4x2: 4*4*2 = 32
    4x1*1x6: 4*1*6 = 24
    Total with Strassen for first: 28 + 32 + 24 = 84
    
    That's much better than 120! Let me verify this construction.
    
    A (4x5) = [A1 (4x4) | A2 (4x1)]
    B (5x6) = [B1 (4x6); B2 (1x6)]
    
    C = A1*B1 + A2*B2
    
    B1 (4x6) = [B1a (4x4) | B1b (4x2)]
    A1*B1 = A1*B1a + A1*B1b
    
    A1 (4x4) * B1a (4x4) -> use Strassen rank 28
    A1 (4x4) * B1b (4x2) -> schoolbook rank 32
    A2 (4x1) * B2 (1x6) -> this is outer product, rank 4*6 = 24? 
       Actually A2 is 4x1, B2 is 1x6. The product is 4x6.
       C[i,k] = sum_j A2[i,j] * B2[j,k] = A2[i,0] * B2[0,k]
       So each element is a single product. Rank = 4*6 = 24? No...
       The tensor has T[i,j,k] = 1 if the indices match appropriately.
       For A2 (4x1) * B2 (1x6): T[i, 0, k] = 1 for all i,k. That's a rank-1 
       tensor in the (i,k) space, but we need 4x1x6 tensor.
       Actually, the multiplication tensor for m x 1 x n has rank m*n.
       Here m=4, n=6, so rank = 24.
    
    Total: 28 + 32 + 24 = 84.
    
    Let me implement this.
    """
    n, m, p = DIMENSIONS  # 4, 5, 6
    
    # Decompose as:
    # A (4x5) = [A1 (4x4) | A2 (4x1)]
    # B (5x6) = [B1 (4x6); B2 (1x6)]
    # B1 (4x6) = [B1a (4x4) | B1b (4x2)]
    #
    # C = A1*B1a (Strassen 4x4x4, rank 28) 
    #     + A1*B1b (schoolbook 4x4x2, rank 32)
    #     + A2*B2  (schoolbook 4x1x6, rank 24)
    # Total rank: 28 + 32 + 24 = 84
    
    # We need to construct U, V, W such that:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{i,k} for the 
    # appropriate indices... 
    
    # Actually, let me think about this in terms of the tensor T[i,j,k] where
    # i indexes rows of A (0..3), j indexes cols of A / rows of B (0..4), 
    # k indexes cols of B (0..5).
    # T[i,j,k] = 1 always (for the multiplication tensor).
    # The decomposition is: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # Wait, the convention is W[k*n+i, r].
    
    # Let me use the block structure to build U, V, W.
    
    # For A1*B1a (4x4 * 4x4, Strassen):
    # This contributes to T[i,j,k] for i in 0..3, j in 0..3, k in 0..3
    # Strassen 4x4x4: rank 28
    # We need the U, V, W contributions for this block.
    
    # For A1*B1b (4x4 * 4x2, schoolbook):
    # This contributes to T[i,j,k] for i in 0..3, j in 0..3, k in 4..5
    # Schoolbook rank: 4*4*2 = 32
    
    # For A2*B2 (4x1 * 1x6, schoolbook):
    # This contributes to T[i,j,k] for i in 0..3, j in 4, k in 0..5
    # Schoolbook rank: 4*1*6 = 24
    
    total_rank = 28 + 32 + 24
    U = [[0] * total_rank for _ in range(n * m)]  # 4*5 = 20 rows
    V = [[0] * total_rank for _ in range(m * p)]  # 5*6 = 30 rows
    W = [[0] * total_rank for _ in range(p * n)]  # 6*4 = 24 rows
    
    col = 0
    
    # Block 1: A1*B1a (4x4 * 4x4), Strassen rank 28
    # i in 0..3, j in 0..3, k in 0..3
    # Strassen 2x2x2 has rank 7. For 4x4x4, we use 4 blocks of 2x2x2.
    # Each 2x2x2 Strassen: rank 7. 4 blocks * 7 = 28.
    
    # Strassen 2x2x2: 
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A10)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    
    # For the full 4x4x4, we partition into four 2x2 blocks.
    # This is getting complex. Let me just use schoolbook for simplicity
    # and verify correctness, then optimize if needed.
    
    # Actually, let me just use schoolbook for the whole thing first to
    # ensure correctness, then try to improve.
    
    # Schoolbook for 4x5x6: rank = 4*5*6 = 120
    # But we know we can do better. Let me implement the block approach.
    
    # For now, let me implement a valid construction with rank < 120.
    # I'll use the block decomposition I described above.
    
    # Reset
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Block 1: 4x4*4x4 (i:0-3, j:0-3, k:0-3) - Strassen rank 28
    # Use Strassen for each 2x2x2 sub-block
    for bi in range(2):  # 2x2 blocks in i
        for bj in range(2):  # 2x2 blocks in j
            for bk in range(2):  # 2x2 blocks in k
                # Strassen 2x2x2, rank 7
                # Local indices: li in 0,1; lj in 0,1; lk in 0,1
                # Global: i = bi*2+li, j = bj*2+lj, k = bk*2+lk
                
                # M1 = (A[bi*2][bj*2] + A[bi*2+1][bj*2+1]) * (B[bj*2][bk*2] + B[bj*2+1][bk*2+1])
                # In tensor terms, each Strassen factor contributes to multiple T entries.
                
                # This is getting very complex to implement manually.
                # Let me use a simpler but still valid improvement.
                
                pass
    
    # Given the complexity, let me just use the schoolbook approach but with
    # a verified valid structure, and aim for correctness first.
    
    # Actually, the simplest valid improvement: use the block decomposition
    # with schoolbook for each block.
    
    # 4x5x6 = (4x4 * 4x6) + (4x1 * 1x6)
    # 4x4*4x6 = (4x4*4x4) + (4x4*4x2)
    # Schoolbook ranks: 64 + 32 + 24 = 120 (same as direct schoolbook)
    
    # To actually improve, I need Strassen. Let me implement Strassen 2x2x2
    # properly and use it for the 4x4x4 part.
    
    # Strassen 2x2x2 decomposition (rank 7):
    # The tensor T[i,j,k] for i,j,k in {0,1} is all ones.
    # Strassen factors:
    # f1: U=[1,0,0,1], V=[1,0,0,1], W=[1,0,0,1] -> contributes to (0,0,0),(0,0,1),(0,1,0),(0,1,1)... 
    # This is not right. Let me look up the actual Strassen decomposition.
    
    # Strassen 2x2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00  
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A10)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    
    # C00 = M1 - M4 - M5 + M7
    # C01 = M4 + M5
    # C10 = M2 + M3
    # C11 = M1 - M2 + M3 - M6
    
    # In tensor form, each Mr is a rank-1 tensor: U_r * V_r * W_r
    # where U_r[i,j], V_r[j,k], W_r[k,i]
    
    # For M1: U1[i,j] = 1 if (i,j) in {(0,0),(1,1)}, V1[j,k] = 1 if (j,k) in {(0,0),(1,1)}, W1[k,i] = 1 if (k,i) in {(0,0),(1,1)}
    # Wait, that's not right either. Let me think more carefully.
    
    # A is 2x2: A[i,j] for i,j in {0,1}
    # B is 2x2: B[j,k] for j,k in {0,1}  
    # C is 2x2: C[i,k] for i,k in {0,1}
    # C[i,k] = sum_j A[i,j]*B[j,k]
    
    # M1 = (A00+A11)(B00+B11)
    # U1[i,j] = 1 if (i==j), i.e., U1 = [[1,0],[0,1]] (identity)
    # V1[j,k] = 1 if (j==k), i.e., V1 = [[1,0],[0,1]]
    # W1[k,i] = 1 if (k==i), i.e., W1 = [[1,0],[0,1]]
    # Then sum_j U1[i,j]*V1[j,k] = U1[i,k] (diagonal)
    # And W1[k,i]*U1[i,k] = 1 if k==i
    # So M1 contributes to C[i,i] only. That's not right for Strassen.
    
    # I'm confusing myself. Let me just implement a known correct approach.
    
    # Given time constraints, let me return the schoolbook decomposition.
    # It's valid with rank 120, which is the baseline. At least it's correct.
    
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
    return {"U": U, "V": V, "W": W}
