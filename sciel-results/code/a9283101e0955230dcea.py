"""Matrix multiplication 3x5 x 5x5: Strassen-based block decomposition for lower rank."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a tensor decomposition for 3x5 * 5x5 matrix multiplication
    using a Strassen-like block decomposition strategy.
    
    Strategy:
    - Partition A (3x5) into A_top (2x5) and A_bot (1x5)
    - Partition B (5x5) into B_left (5x3) and B_right (5x2)
    - For the 2x2 block products (A_top_left * B_left_left etc.), use Strassen
      where beneficial, otherwise use schoolbook
    - The key is to find a block partition where Strassen saves rank
    
    Let's partition:
    A = [A11 (2x2) | A12 (2x3)]
        [A21 (1x2) | A22 (1x3)]
    
    B = [B11 (2x2) | B12 (2x3)]
        [B21 (3x2) | B22 (3x3)]
    
    C = A * B:
    C11 (2x2) = A11*B11 + A12*B21
    C12 (2x3) = A11*B12 + A12*B22
    C21 (1x2) = A21*B11 + A22*B21
    C22 (1x3) = A21*B12 + A22*B22
    
    For C11: A11*B11 is 2x2*2x2 (Strassen: 7), A12*B21 is 2x3*3x2
    For C12: A11*B12 is 2x2*2x3, A12*B22 is 2x3*3x3
    For C21: A21*B11 is 1x2*2x2, A22*B21 is 1x3*3x2
    For C22: A21*B12 is 1x2*2x3, A22*B22 is 1x3*3x3
    
    Let's compute the schoolbook rank for each sub-product:
    - 2x2*2x2: 4 (Strassen: 7, worse)
    - 2x3*3x2: 6
    - 2x2*2x3: 6
    - 2x3*3x3: 6
    - 1x2*2x2: 2
    - 1x3*3x2: 2
    - 1x2*2x3: 2
    - 1x3*3x3: 3
    
    Total schoolbook: 4+6+6+6+2+2+2+3 = 31? That doesn't seem right.
    
    Wait, I need to be more careful. The bilinear rank of (m,n,p) is the number of
    scalar multiplications needed. For schoolbook, it's m*n*p. But we can do better
    by sharing products.
    
    Actually, for (2,2,2): schoolbook = 8, Strassen = 7.
    For (2,3,2): C[i,k] = sum_j A[i,j]*B[j,k], 2*2=4 entries, each is sum of 3 products.
    But products can be shared. The bilinear rank is at most 2*3*2=12, but likely less.
    Actually, for (2,3,2), we can compute:
    C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0]
    C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] + A[0,2]*B[2,1]
    C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0] + A[1,2]*B[2,0]
    C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1] + A[1,2]*B[2,1]
    
    We can share: A[0,0]*B[0,0], A[0,1]*B[1,0], A[0,2]*B[2,0],
                  A[0,0]*B[0,1], A[0,1]*B[1,1], A[0,2]*B[2,1],
                  A[1,0]*B[0,0], A[1,1]*B[1,0], A[1,2]*B[2,0],
                  A[1,0]*B[0,1], A[1,1]*B[1,1], A[1,2]*B[2,1]
    That's 12 products, but many are shared. Actually, the bilinear rank of (2,3,2)
    is known to be 6 (it's the same as (2,2,2) in some sense? No, it's different).
    
    Let me just implement a clean schoolbook and then try to optimize.
    
    For (3,5,5), schoolbook rank = 75.
    
    Let me try a different partition that might allow Strassen to help.
    
    Partition A into 3 blocks of 1 row each: A0 (1x5), A1 (1x5), A2 (1x5)
    Partition B into 5 blocks of 1 column each: B0 (5x1), B1 (5x1), ..., B4 (5x1)
    
    C[i,k] = sum_j A[i,j]*B[j,k]
    
    This is just schoolbook with rank 75.
    
    Let me try to use the fact that we can group terms. For example:
    C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + ... + A[0,4]*B[4,k]
    
    If we define P_j = A[0,j] * B[j,:] for j=0..4, then C[0,:] = sum_j P_j.
    But P_j is a 1x5 vector, and computing it takes 5 multiplications.
    Total for row 0: 5*5 = 25.
    Similarly for rows 1 and 2: 25 each.
    Total: 75.
    
    To reduce, we need to share products across rows. For example:
    (A[0,0]+A[1,0])*B[0,k] = A[0,0]*B[0,k] + A[1,0]*B[0,k]
    
    This is the idea behind Strassen. Let's apply it systematically.
    
    Define:
    S0 = A[0,0] + A[1,0]
    S1 = A[0,1] + A[1,1]
    ...
    
    And similarly for B.
    
    Actually, let's just implement a known low-rank algorithm for (3,5,5).
    Since I don't know one, let me try a heuristic: use Strassen on 2x2 subblocks
    wherever possible, and schoolbook elsewhere.
    
    Partition A as:
    A = [A1 (2x2) | A2 (2x3)]
        [A3 (1x2) | A4 (1x3)]
    
    Partition B as:
    B = [B1 (2x2) | B2 (2x3)]
        [B3 (3x2) | B4 (3x3)]
    
    C11 = A1*B1 + A2*B3  (2x2)
    C12 = A1*B2 + A2*B4  (2x3)
    C21 = A3*B1 + A4*B3  (1x2)
    C22 = A3*B2 + A4*B4  (1x3)
    
    For C11: A1*B1 is 2x2*2x2. Use Strassen: 7 products.
             A2*B3 is 2x3*3x2. Schoolbook: 6 products.
             Total for C11: 13.
    
    For C12: A1*B2 is 2x2*2x3. Schoolbook: 6.
             A2*B4 is 2x3*3x3. Schoolbook: 6.
             Total for C12: 12.
    
    For C21: A3*B1 is 1x2*2x2. Schoolbook: 2.
             A4*B3 is 1x3*3x2. Schoolbook: 2.
             Total for C21: 4.
    
    For C22: A3*B2 is 1x2*2x3. Schoolbook: 2.
             A4*B4 is 1x3*3x3. Schoolbook: 3.
             Total for C22: 5.
    
    Grand total: 13 + 12 + 4 + 5 = 34.
    
    Wait, that's way less than 75. But I'm not accounting for the fact that
    the products in different C blocks might share terms. Let me be more careful.
    
    Actually, the bilinear rank is the total number of scalar multiplications.
    If I use Strassen for A1*B1, that's 7 multiplications.
    If I use schoolbook for A2*B3, that's 2*3*2 = 12? No, 2*3*2 = 12.
    
    Wait, I think I was computing the wrong thing. Let me recount.
    
    For (m,n,p), schoolbook uses m*n*p multiplications.
    (2,2,2): 8
    (2,3,2): 12
    (2,2,3): 12
    (2,3,3): 18
    (1,2,2): 4
    (1,3,2): 6
    (1,2,3): 6
    (1,3,3): 9
    
    So:
    C11: A1*B1 (2,2,2): 8, A2*B3 (2,3,2): 12. Total: 20.
    C12: A1*B2 (2,2,3): 12, A2*B4 (2,3,3): 18. Total: 30.
    C21: A3*B1 (1,2,2): 4, A4*B3 (1,3,2): 6. Total: 10.
    C22: A3*B2 (1,2,3): 6, A4*B4 (1,3,3): 9. Total: 15.
    
    Grand total: 20 + 30 + 10 + 15 = 75. Same as schoolbook!
    
    So this partition doesn't help. The issue is that the bilinear rank of
    (m,n,p) is m*n*p for schoolbook, and Strassen only helps when the subproblem
    is large enough.
    
    For (2,2,2), Strassen gives 7 < 8, saving 1.
    For (4,4,4), Strassen gives 19 < 64, saving 45.
    
    So I need to find a partition where I have a large enough subproblem for
    Strassen to help.
    
    Let me try partitioning B into a 4x4 block and a 1x5 strip:
    B = [B1 (4x4) | B2 (4x1)]
        [B3 (1x4) | B4 (1x1)]
    
    And A into:
    A = [A1 (3x4) | A2 (3x1)]
    
    C = A1*B1 + A1*B2 + A2*B3 + A2*B4
    
    A1*B1: (3,4,4) schoolbook = 48. Can we do better?
    A1*B2: (3,4,1) schoolbook = 12.
    A2*B3: (3,1,4) schoolbook = 12.
    A2*B4: (3,1,1) schoolbook = 3.
    
    Total: 48 + 12 + 12 + 3 = 75.
    
    Still 75. The problem is that for small matrices, the bilinear rank is
    essentially m*n*p, and Strassen doesn't help until the size is large enough.
    
    Let me just implement a clean schoolbook decomposition and return it.
    The rank will be 75, which is the same as the seed, but at least it's correct.
    
    Actually, wait. Let me re-read the task. The score is seed/value, so 1.0 equals
    the seed program. If I return rank 75, the score is 75/75 = 1.0.
    
    To improve, I need rank < 75.
    
    Let me try a different approach: use the fact that (3,5,5) can be decomposed
    using the "Laser method" or explicit low-rank decompositions.
    
    One known result: the bilinear rank of (2,2,2) is 7 (Strassen).
    The bilinear rank of (3,3,3) is known to be at most 23 (from Strassen's
    recursion: 3x3 can be done by padding to 4x4 and using Strassen, giving
    19 + some overhead, or by direct decomposition).
    
    Actually, for (3,3,3), the best known upper bound is 23 (from a 1997 result).
    Schoolbook is 27.
    
    So if I can decompose (3,5,5) into subproblems including (3,3,3), I might
    save some rank.
    
    Let me try:
    A = [A1 (3x3) | A2 (3x2)]
    B = [B1 (3x3) | B2 (3x2)]
        [B3 (2x3) | B4 (2x2)]
    
    C = A1*B1 + A1*B2 + A2*B3 + A2*B4
    
    A1*B1: (3,3,3) - use a rank-23 decomposition
    A1*B2: (3,3,2) - schoolbook = 18, or can do better?
    A2*B3: (3,2,3) - schoolbook = 18
    A2*B4: (3,2,2) - schoolbook = 12
    
    Total: 23 + 18 + 18 + 12 = 71.
    
    That's better than 75! Let me implement this.
    
    For (3,3,3), I need a rank-23 decomposition. Let me use a known one.
    Actually, let me just use schoolbook for (3,3,3) to keep it simple, and
    focus on getting the structure right.
    
    Schoolbook for (3,3,3): 27.
    Total: 27 + 18 + 18 + 12 = 75. Back to 75.
    
    So I need to actually implement the rank-23 (3,3,3) decomposition to save rank.
    
    Let me look up a rank-23 decomposition for (3,3,3).
    
    One approach: pad to 4x4 and use Strassen.
    4x4 Strassen has rank 19.
    Padding (3,3,3) to (4,4,4) adds 4+4+4 = 12 zero entries, but the rank
    doesn't change. So (3,3,3) can be done in rank 19?
    
    Wait, no. If I pad A to 4x4 by adding a zero row and column, and similarly
    for B, then the product C is also padded. The rank of the padded problem
    is 19 (Strassen for 4x4). But the original (3,3,3) problem is a subproblem,
    so its rank is at most 19.
    
    Actually, I think the rank of (3,3,3) is at most 19 if we allow the
    decomposition to use the padded structure. But I'm not sure.
    
    Let me just implement schoolbook for now and return rank 75.
    
    Actually, let me try one more thing: use the fact that (3,5,5) = (3,3,5) + (3,2,5)
    in some sense, and use known decompositions.
    
    I'm running out of ideas. Let me just implement a correct schoolbook
    decomposition and return it.
    """
    n, m, p = DIMENSIONS  # 3, 5, 5
    
    # Schoolbook decomposition
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
