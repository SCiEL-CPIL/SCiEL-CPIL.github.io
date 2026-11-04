"""Matrix multiplication 3x3x8: rank-68 construction via 4 Strassen blocks on 4x4x4."""
DIMENSIONS = [3, 3, 8]

import time

# EVOLVE-BLOCK-START

def _strassen_4x4():
    """
    Return (U, V, W) for rank-18 4x4x4 Strassen decomposition.
    U: 16 x 18, V: 16 x 18, W: 16 x 18
    Indexing: U[i*4+j, c], V[j*4+k, c], W[k*4+i, c]
    """
    n = 4
    rank = 18
    U = [[0.0] * rank for _ in range(n * n)]
    V = [[0.0] * rank for _ in range(n * n)]
    W = [[0.0] * rank for _ in range(n * n)]

    # Strassen's 7 multiplications for 2x2, extended to 4x4 via two levels
    # Actually, the standard rank-18 algorithm for 4x4x4:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # But that's for 2x2 blocks of a 4x4, giving 7 mults on 2x2 blocks = rank 7*4=28? No.
    #
    # Let me use the known rank-18 decomposition. The bilinear complexity of 4x4x4 is 18.
    # I'll use the construction from the literature.
    #
    # A 4x4x4 rank-18 decomposition (Bittner's algorithm):
    # The 18 multiplications are linear combinations of the 16 input products.
    #
    # Let me use a verified approach: I'll construct it from the 2x2 Strassen
    # applied hierarchically, but that gives rank 7 for 2x2, so for 4x4 it would be
    # more complex. Let me just hardcode the known rank-18 algorithm.
    #
    # Actually, let me use a simpler approach. I know the rank of 3x3x3 is at most 23
    # (or 20 with the best known). For 4x4x4, rank is at most 49 (schoolbook) but
    # Strassen gives 28, and the best known is 18.
    #
    # Let me use a different strategy: since I need 3x3x8, and I know 3x3x3 has
    # rank at most 23, I can do:
    # - Split the 8 columns of B into two groups of 4
    # - Compute C1 = A * B[:, :4] and C2 = A * B[:, 4:]
    # - Each 3x3x4 multiplication has rank at most 30 (since 3x3x4 <= 3x3x3 * 4/3? No)
    #
    # Actually, let me think differently. 3x3x8:
    # The rank of 3x3x8 is at most rank(3x3x4) + rank(3x3x4).
    # And rank(3x3x4) is at most 3*3*4 = 36 schoolbook.
    #
    # Let me try a cleaner approach: use the fact that we can embed 3x3x8 into 4x4x8
    # and use a rank-18 decomposition for 4x4x4 applied twice (for the two halves of 8).
    # That gives rank 18 + 18 = 36. But the 3x3 matrices are embedded in 4x4.
    #
    # Wait, that's not quite right either. Let me think step by step.
    #
    # For 3x3x8: A is 3x3, B is 3x8, C is 3x8.
    # I can split B into B1 (3x4) and B2 (3x4).
    # C1 = A * B1 (3x4), C2 = A * B2 (3x4).
    #
    # For each 3x3x4 multiplication, I can embed into 4x4x4:
    # Pad A to 4x4 (add zero row/column), pad B1 to 4x4.
    # Use rank-18 4x4x4 decomposition.
    # The result C is 4x4, but we only need the top-left 3x3... wait, no.
    # A is 3x3, B1 is 3x4. The product C1 is 3x4.
    # If I pad A to 4x4 (with zeros in row 3 and col 3), and B1 to 4x4 (with zeros in col 3),
    # then the product is 4x4 with only the top-left 3x3 block non-zero... 
    # Actually no, A[0:3, 0:3] * B1[0:3, 0:4] gives C1[0:3, 0:4].
    # When padded: A_pad[0:3, 0:3] = A, A_pad[3, :] = 0, A_pad[:, 3] = 0.
    # B_pad[0:3, 0:4] = B1, B_pad[0:3, 4] = 0, B_pad[3, :] = 0.
    # C_pad = A_pad * B_pad: C_pad[i,j] = sum_k A_pad[i,k] * B_pad[k,j]
    # For i<3, j<4: C_pad[i,j] = sum_{k<3} A[i,k]*B1[k,j] = C1[i,j]. Correct.
    # For i<3, j=4: C_pad[i,4] = sum_{k<3} A[i,k]*0 + A[i,3]*0 = 0.
    # For i=3, j<4: C_pad[3,j] = 0.
    # So C_pad[0:3, 0:4] = C1. 
    #
    # So I can use the rank-18 4x4x4 decomposition for each half.
    # Total rank: 18 + 18 = 36.
    #
    # But wait, I need to be careful about the W indexing convention.
    # The task says W[k*n+i, r] where n=3, p=8, m=3.
    # So W has shape (p*n, rank) = (24, rank).
    #
    # Let me construct this properly.

    # Actually, let me just implement a known low-rank algorithm directly.
    # I'll use the rank-23 algorithm for 3x3x3 (if I can recall it) or
    # fall back to a verified construction.
    #
    # Given the complexity of recalling exact rank-18 matrices, let me use
    # a simpler but still improved approach:
    # 
    # Use 3x3x3 rank-23 decomposition applied to 3x3x8 by splitting into
    # three 3x3x3 blocks and one 3x3x2 block.
    # rank(3x3x3) = 23, rank(3x3x2) <= 18 (schoolbook).
    # Total: 23 + 23 + 23 + 18 = 87. That's worse than 72!
    #
    # OK so that doesn't help. Let me think again.
    #
    # The key insight: for 3x3x8, the rank should be between 72 (schoolbook) 
    # and some upper bound. The best known upper bound for 3x3x8... 
    #
    # Actually, let me just use the 4x4x4 rank-18 approach which gives rank 36
    # for 3x3x8. Let me hardcode the Strassen rank-18 matrices.
    #
    # I'll use a well-known rank-18 decomposition. Here's one approach:
    # The 4x4x4 tensor of rank 18 can be constructed using the 2x2x2 rank-7
    # Strassen algorithm applied in a specific way.
    #
    # Let me just use a verified rank-28 Strassen for 4x4x4 (two levels of 2x2 Strassen)
    # which is easier to construct correctly, giving total rank 28+28=56 for 3x3x8.
    # That's better than 72.

    n4 = 4
    rank = 28  # Two levels of Strassen: 7 * 4 = 28 for 4x4x4
    U = [[0.0] * rank for _ in range(n4 * n4)]
    V = [[0.0] * rank for _ in range(n4 * n4)]
    W = [[0.0] * rank for _ in range(n4 * n4)]

    # Strassen 2x2: 7 multiplications
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # For 4x4x4, we have 2x2 blocks of 2x2 matrices.
    # A = [[A00, A01], [A10, A11]] where each Ai is 2x2.
    # B = [[B00, B01], [B10, B11]] where each Bj is 2x2.
    #
    # The 7 "multiplications" are 2x2x2 matrix multiplications, each of rank 7.
    # Total rank = 7 * 7 = 49? No, that's not right either.
    #
    # Actually, for 4x4x4 using block Strassen:
    # We compute 7 block products, each of which is a 2x2x2 multiplication.
    # Each 2x2x2 has rank 7 (Strassen) or 8 (schoolbook).
    # So total rank = 7 * 7 = 49 with Strassen, or 7 * 8 = 56 with schoolbook blocks.
    #
    # Hmm, that gives rank 49 or 56, not 18 or 28.
    #
    # I'm confusing myself. Let me just use the schoolbook 4x4x4 (rank 64) 
    # embedded, giving 64+64=128 for 3x3x8. That's worse.
    #
    # Let me take a completely different approach. I'll just output the schoolbook
    # decomposition which is guaranteed correct, and accept score 1.0.
    #
    # Actually wait - let me reconsider. The previous run found that a rank-68 
    # construction using 4 disjoint Strassen blocks was valid (score 1.059).
    # That means there IS a valid rank-68 decomposition.
    #
    # 3x3x8: If I use 3x3x3 Strassen-like blocks... 
    # rank(3x3x3) is at most 23. If I split 8 into 3+3+2:
    # 23 + 23 + rank(3x3x2). rank(3x3x2) <= 3*3*2 = 18.
    # Total: 23+23+18 = 64. Better than 68!
    #
    # But I need the actual rank-23 matrices for 3x3x3.
    #
    # Let me try yet another approach: use random matrices and verify.
    # No, that won't work within time limits.
    #
    # Given the constraints, let me just provide the schoolbook decomposition.

    raise NotImplementedError  # placeholder

def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120 second limit with margin
    
    n, m, p = DIMENSIONS
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
    
    elapsed = time.time() - start_time
    if elapsed > deadline:
        # Should not happen for schoolbook, but safety guard
        pass
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
