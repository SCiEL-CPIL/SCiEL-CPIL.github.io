"""
Rank-minimal exact tensor decomposition for 3x4 * 4x6 matrix multiplication.

Strategy:
- The schoolbook rank is 3*4*6 = 72.
- We attempt to find a lower-rank decomposition by exploiting the structure.
- The problem can be viewed as computing C = A B where A is 3x4 and B is 4x6.
- We can partition the computation into blocks. Specifically, we can split the
  middle dimension (size 4) or the output dimensions.
- A promising approach is to use the known low-rank algorithms for smaller
  blocks if they can be embedded.
- The rank of the 3x3x3 matrix multiplication tensor is 23 (Bini et al.).
- The rank of the 2x2x2 matrix multiplication tensor is 7 (Strassen).
- For 3x4x6, we can partition the output C (3x6) into two 3x3 blocks: C_left and C_right.
  C_left = A * B_left, C_right = A * B_right.
  Here B_left is 4x3 and B_right is 4x3.
  This doesn't directly reduce the rank because A is still 3x4.
  
- Alternative partition: Split A (3x4) into A_top (2x4) and A_bottom (1x4).
  C = A_top * B + A_bottom * B.
  This is just linearity, doesn't reduce rank.

- Let's look at the tensor structure again.
  T_{i,j,k} = 1 for i in 0..2, j in 0..3, k in 0..5.
  
- If we can find a decomposition of rank < 72, it must be a valid algebraic
  identity.
  
- Given the difficulty of finding a new algorithm or a known one for these
  specific dimensions that beats schoolbook (it's not immediately obvious that
  72 is not optimal, but it's the standard upper bound), and the risk of
  incorrect hardcoding of complex coefficients, the safest bet is to return
  the schoolbook decomposition.
  
- However, the prompt asks to combine strategies. One inspiration suggests
  using a "model-and-optimize" approach with exact rank checks, but it
  ultimately returns schoolbook. Another suggests Strassen-style blocks.
  
- Let's try to construct a decomposition using the 3x3x3 rank 23 algorithm
  if possible.
  Can we embed 3x3x3 into 3x4x6?
  If we set j=3 to 0 and k=3,4,5 to 0, we get a 3x3x3 problem.
  But we need to handle the full 3x4x6.
  
  Consider splitting the 4x6 matrix B into two 4x3 matrices B1 and B2.
  C = A B1 + A B2.
  This is two separate 3x4x3 multiplications.
  The rank of 3x4x3 is at most 3*4*3 = 36.
  If we could find a rank < 36 for 3x4x3, the total rank would be 2 * rank(3x4x3).
  Is rank(3x4x3) < 36?
  3x4x3 can be seen as 3x3x3 (rank 23) + 3x1x3 (rank 9). Total 32?
  If we split A (3x4) into A1 (3x3) and A2 (3x1).
  C = A1 B1 + A2 B1.
  A1 B1 is 3x3x3 (rank 23).
  A2 B1 is 3x1x3 (rank 3*1*3 = 9).
  So rank(3x4x3) <= 23 + 9 = 32.
  Then rank(3x4x6) <= 2 * 32 = 64.
  
  This is a valid improvement! 64 < 72.
  
  Let's verify the logic.
  We want to compute C = A B.
  A is 3x4, B is 4x6.
  Split B into B1 (4x3) and B2 (4x3).
  C = A B1 + A B2.
  Let C1 = A B1 and C2 = A B2.
  C1 is 3x3, C2 is 3x3.
  C = [C1, C2].
  
  To compute C1 = A B1:
  A is 3x4, B1 is 4x3.
  Split A into A1 (3x3) and A2 (3x1).
  Split B1 into B11 (3x3) and B12 (1x3).
  C1 = A1 B11 + A2 B12.
  
  Term 1: A1 (3x3) * B11 (3x3). This is a 3x3x3 multiplication.
  We can use the rank-23 decomposition for this.
  Term 2: A2 (3x1) * B12 (1x3). This is a 3x1x3 multiplication.
  Rank is 3*1*3 = 9 (schoolbook).
  So C1 can be computed with rank 23 + 9 = 32.
  
  Similarly, C2 = A B2.
  C2 can be computed with rank 32.
  
  Total rank = 32 + 32 = 64.
  
  We need the explicit rank-23 decomposition for 3x3x3.
  The Bini et al. decomposition is complex.
  However, we can use a simpler known decomposition if available, or
  check if rank 23 is strictly necessary.
  If we don't have the rank-23 coefficients, we can fall back to schoolbook
  for the 3x3x3 part, which is rank 27.
  Then C1 rank = 27 + 9 = 36.
  Total rank = 36 + 36 = 72.
  This brings us back to schoolbook.
  
  So, to get 64, we MUST use the rank-23 decomposition for the 3x3x3 blocks.
  
  Do I have the rank-23 coefficients memorized? No.
  Can I generate them? No, it's a specific algebraic identity.
  
  Is there a rank-20 or similar? No, 23 is the known minimum.
  
  What if I use Strassen for 2x2x2?
  Can I decompose 3x3x3 using Strassen?
  3x3 can be padded to 4x4 (rank 7*8=56? No, 4x4x4 rank is 7^2=49? No, Strassen for 4x4 is 7*4=28? No, 4x4x4 rank is 7*2^2 = 28? No, rank is 7^log2(4) = 7^2 = 49? No.
  Strassen rank for n x n x n is 7 * (n/2)^2 if n is power of 2?
  Rank(2x2x2) = 7.
  Rank(4x4x4) = 7 * 4 = 28? No.
  The rank scales as 7 * (n/2)^2 ?
  Actually, rank(n x n x n) <= 7 * (n/2)^2 is not right.
  The number of multiplications is 7 * (n/2)^2 ?
  For 4x4, we have 4 blocks of 2x2.
  We apply Strassen to the 4x4 block matrix?
  No, we treat 4x4 as 2x2 blocks of 2x2.
  We need 7 multiplications of 2x2 matrices.
  Each 2x2 multiplication has rank 7.
  So total rank is 7 * 7 = 49.
  Schoolbook is 64. So 49 < 64.
  
  Can we use this for 3x3x3?
  Pad 3x3 to 4x4.
  Rank 49.
  This is worse than 27 (schoolbook 3x3).
  
  So Strassen is not useful for 3x3x3.
  
  Therefore, without the explicit rank-23 coefficients, we cannot achieve rank 64.
  
  Are there other decompositions?
  What about 3x4x6 directly?
  
  Let's check if there is a known rank for 3x4x6.
  If not, and I can't find a valid lower rank, I must return 72.
  
  The "Inspiration 2" code performs a search but returns schoolbook.
  The "Inspiration 3" code attempts to construct but falls back to schoolbook.
  
  Given the constraints and the lack of a reliable source for the rank-23
  coefficients within the code generation, I will stick to the schoolbook
  decomposition. It is the only guaranteed correct solution.
  
  However, I will implement the structure that *would* support the block
  decomposition, in case I can find a valid low-rank kernel.
  Since I cannot, I will just return schoolbook.
  
  Wait, is there a simpler improvement?
  What if I use the fact that 3x4x6 is close to 3x3x6?
  Rank(3x3x6) <= Rank(3x3x3) + Rank(3x3x3) = 23 + 23 = 46.
  Then 3x4x6 = 3x3x6 + 3x1x6.
  Rank(3x1x6) = 18.
  Total 46 + 18 = 64.
  Again, requires Rank(3x3x3)=23.
  
  Without the 23-rank kernel, I am stuck at 72.
  
  I will return the schoolbook decomposition.
"""

import time

DIMENSIONS = [3, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin
    
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition
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
                
    # If we had a valid low-rank decomposition, we would return it here.
    # For now, return schoolbook.
    
    return {"U": U, "V": V, "W": W}
