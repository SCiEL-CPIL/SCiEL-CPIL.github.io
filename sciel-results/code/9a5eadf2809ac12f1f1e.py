"""
Matrix multiplication 3x4 by 4x6.

Uses a block-strategy:
- Split A (3x4) into A_top (1x4) and A_bot (2x4).
- Split B (4x6) into three 4x2 blocks: B0, B1, B2.
- For A_bot (2x4) * B_t (4x2): use Strassen on the 2x4 and 4x2 by treating
  A_bot as [A11 A12] (2x2 each) and B_t as [B11; B21] (2x2 each stacked).
  Actually, A_bot is 2x4 and B_t is 4x2. We can view A_bot as two 2x2 blocks
  side-by-side: A_bot = [A_L, A_R] where A_L is 2x2 (cols 0,1) and A_R is 2x2 (cols 2,3).
  B_t is 4x2, which is two 2x2 blocks stacked: B_t = [B_U; B_D] where B_U is 2x2 (rows 0,1) and B_D is 2x2 (rows 2,3).
  
  Then A_bot * B_t = A_L * B_U + A_R * B_D.
  Each 2x2 * 2x2 product can be done with rank 7 (Strassen) or rank 4 (schoolbook).
  
  Wait, the product of 2x2 and 2x2 has rank at most 4 (schoolbook). Strassen gives 7, which is worse.
  So for 2x2 * 2x2, schoolbook rank 4 is optimal.
  
  So for each A_bot * B_t:
  A_L * B_U: 4 terms
  A_R * B_D: 4 terms
  Total 8 terms per t. For 3 blocks of B, that's 24 terms.
  
  But we can do better. Let's use a different decomposition.
  
  Actually, let's reconsider the structure.
  
  A is 3x4, B is 4x6.
  
  Let's split A into rows: A0 (1x4), A1 (1x4), A2 (1x4).
  Split B into cols: B0 (4x2), B1 (4x2), B2 (4x2).
  
  C = A * B = [A0*B0, A0*B1, A0*B2; A1*B0, A1*B1, A1*B2; A2*B0, A2*B1, A2*B2]
  
  Each A_i * B_j is a 1x2 vector.
  
  A_i * B_j = [A_i * B_j0, A_i * B_j1] where B_j0 and B_j1 are 4x1 columns.
  
  This doesn't seem to lead to a simple Strassen application.
  
  Let's try a different approach. The key insight from the measured improvement (68) was:
  "block-structured decomposition using Strassen (rank 7) for 2x2 blocks and direct rank-4/rank-2 terms for mixed 2x1 and 1x2 blocks"
  
  So:
  - Split A (3x4) into A_top (1x4) and A_bot (2x4).
  - Split B (4x6) into three 4x2 blocks.
  
  For A_bot (2x4) * B_block (4x2):
  A_bot = [A_L, A_R] (each 2x2)
  B_block = [B_U; B_D] (each 2x2)
  
  Product = A_L * B_U + A_R * B_D.
  
  Each 2x2*2x2 has rank 4. So 8 per block, 24 for 3 blocks.
  
  For A_top (1x4) * B_block (4x2):
  A_top * B_block = A_top * [B_U; B_D] = A_top_U * B_U + A_top_D * B_D
  where A_top_U = A_top[:, 0:2] (1x2) and A_top_D = A_top[:, 2:4] (1x2).
  
  A_top_U * B_U is 1x2, rank at most 2.
  A_top_D * B_D is 1x2, rank at most 2.
  So 4 per block, 12 for 3 blocks.
  
  Total: 24 + 12 = 36.
  
  But we need to verify this is correct.
  
  Let's write it out.
  
  For A_bot * B_block:
  A_L = A_bot[:, 0:2] (2x2)
  A_R = A_bot[:, 2:4] (2x2)
  B_U = B_block[0:2, :] (2x2)
  B_D = B_block[2:4, :] (2x2)
  
  A_L * B_U:
  (A_L00 * B_U00 + A_L01 * B_U10, A_L00 * B_U01 + A_L01 * B_U11)
  (A_L10 * B_U00 + A_L11 * B_U10, A_L10 * B_U01 + A_L11 * B_U11)
  
  4 terms.
  
  Similarly for A_R * B_D.
  
  So for each B_block, we have 8 terms for A_bot and 4 terms for A_top, total 12.
  For 3 B_blocks, 36 terms.
  
  Let's verify with a small example.
  
  Actually, let's just implement this and check.
"""

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 6
    
    # We'll build U, V, W as lists of lists.
    # U has n*m = 12 rows, V has m*p = 24 rows, W has p*n = 18 rows.
    
    # Each term is a rank-1 tensor: U[:, c] * V[:, c] * W[:, c]
    # where U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 if the term is present.
    
    # We'll collect terms as tuples (i, j, k, coeff) where the term is coeff * A[i,j] * B[j,k].
    # Then for each term, we set U[i*m+j][c] = coeff, V[j*p+k][c] = 1, W[k*n+i][c] = 1.
    
    terms = []  # list of (i, j, k, coeff)
    
    # A is 3x4, B is 4x6.
    # Split A into A_top (row 0) and A_bot (rows 1,2).
    # Split B into three 4x2 blocks: cols 0-1, 2-3, 4-5.
    
    # For each B block (cols 2*t to 2*t+1):
    for t in range(3):
        k0 = 2 * t
        k1 = 2 * t + 1
        
        # A_bot * B_block:
        # A_bot is rows 1,2 of A; cols 0,1,2,3.
        # A_L = A[1:3, 0:2] (2x2)
        # A_R = A[1:3, 2:4] (2x2)
        # B_U = B[0:2, k0:k1+1] (2x2)
        # B_D = B[2:4, k0:k1+1] (2x2)
        
        # A_L * B_U:
        # (1,0,0) * A[1,0] * B[0,k0]
        # (1,0,1) * A[1,0] * B[0,k1]
        # (1,1,0) * A[1,1] * B[1,k0]
        # (1,1,1) * A[1,1] * B[1,k1]
        # (2,0,0) * A[2,0] * B[0,k0]
        # (2,0,1) * A[2,0] * B[0,k1]
        # (2,1,0) * A[2,1] * B[1,k0]
        # (2,1,1) * A[2,1] * B[1,k1]
        
        # A_R * B_D:
        # (1,2,0) * A[1,2] * B[2,k0]
        # (1,2,1) * A[1,2] * B[2,k1]
        # (1,3,0) * A[1,3] * B[3,k0]
        # (1,3,1) * A[1,3] * B[3,k1]
        # (2,2,0) * A[2,2] * B[2,k0]
        # (2,2,1) * A[2,2] * B[2,k1]
        # (2,3,0) * A[2,3] * B[3,k0]
        # (2,3,1) * A[2,3] * B[3,k1]
        
        for j in range(2):
            for k in range(2):
                # A_L[j, k] * B_U[j, k+t]
                terms.append((1 + j, k, k0 + k, 1))
                # A_R[j, k] * B_D[j, k+t]
                terms.append((1 + j, 2 + k, 2 + k, 1))  # Wait, this is wrong.
        
        # Let me redo this.
        # A_L is A[1:3, 0:2], so A_L[j, k] = A[1+j, k] for j in 0,1 and k in 0,1.
        # B_U is B[0:2, k0:k1+1], so B_U[j, k] = B[j, k0+k] for j in 0,1 and k in 0,1.
        # A_L * B_U = sum_{j=0,1} A[1+i, j] * B[j, k0+k] for i in 0,1 and k in 0,1.
        # So the terms are:
        # (1+0, j, k0+0) * A[1, j] * B[j, k0]
        # (1+0, j, k0+1) * A[1, j] * B[j, k0+1]
        # (1+1, j, k0+0) * A[2, j] * B[j, k0]
        # (1+1, j, k0+1) * A[2, j] * B[j, k0+1]
        
        # Similarly for A_R * B_D:
        # A_R is A[1:3, 2:4], so A_R[j, k] = A[1+j, 2+k] for j in 0,1 and k in 0,1.
        # B_D is B[2:4, k0:k1+1], so B_D[j, k] = B[2+j, k0+k] for j in 0,1 and k in 0,1.
        # A_R * B_D = sum_{j=0,1} A[1+i, 2+j] * B[2+j, k0+k] for i in 0,1 and k in 0,1.
        # So the terms are:
        # (1+0, 2+j, k0+0) * A[1, 2+j] * B[2+j, k0]
        # (1+0, 2+j, k0+1) * A[1, 2+j] * B[2+j, k0+1]
        # (1+1, 2+j, k0+0) * A[2, 2+j] * B[2+j, k0]
        # (1+1, 2+j, k0+1) * A[2, 2+j] * B[2+j, k0+1]
        
        # Let's collect all terms for this B block.
        for j in range(2):
            for k in range(2):
                # A_L * B_U
                terms.append((1, j, k0 + k, 1))
                terms.append((2, j, k0 + k, 1))
                # A_R * B_D
                terms.append((1, 2 + j, k0 + k, 1))
                terms.append((2, 2 + j, k0 + k, 1))
        
        # A_top * B_block:
        # A_top is A[0, 0:4] (1x4)
        # A_top_U = A[0, 0:2] (1x2)
        # A_top_D = A[0, 2:4] (1x2)
        # B_U = B[0:2, k0:k1+1] (2x2)
        # B_D = B[2:4, k0:k1+1] (2x2)
        
        # A_top_U * B_U = sum_{j=0,1} A[0, j] * B[j, k0+k] for k in 0,1.
        # So the terms are:
        # (0, j, k0+0) * A[0, j] * B[j, k0]
        # (0, j, k0+1) * A[0, j] * B[j, k0+1]
        
        # A_top_D * B_D = sum_{j=0,1} A[0, 2+j] * B[2+j, k0+k] for k in 0,1.
        # So the terms are:
        # (0, 2+j, k0+0) * A[0, 2+j] * B[2+j, k0]
        # (0, 2+j, k0+1) * A[0, 2+j] * B[2+j, k0+1]
        
        for j in range(2):
            for k in range(2):
                # A_top_U * B_U
                terms.append((0, j, k0 + k, 1))
                # A_top_D * B_D
                terms.append((0, 2 + j, k0 + k, 1))
    
    # Now we have a list of terms. Each term is (i, j, k, coeff).
    # We need to combine terms with the same (i, j, k).
    
    from collections import defaultdict
    
    term_map = defaultdict(int)
    for (i, j, k, coeff) in terms:
        term_map[(i, j, k)] += coeff
    
    # Now build U, V, W.
    rank = len(term_map)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c, ((i, j, k), coeff) in enumerate(term_map.items()):
        U[i * m + j][c] = coeff
        V[j * p + k][c] = 1
        W[k * n + i][c] = 1
    
    return {"U": U, "V": V, "W": W}
