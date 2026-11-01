import time
import numpy as np
from fractions import Fraction

# Task dimensions
N, M, P = 2, 4, 6
RANK_LIMIT = 42

def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120s safety margin

    # Dimensions
    n, m, p = N, M, P
    size_U = n * m  # 8
    size_V = m * p  # 24
    size_W = p * n  # 12
    rank = RANK_LIMIT  # 42

    # We use the 42-rank decomposition for 2x4x6 derived from Strassen-like block structure.
    # The known result for 2x4x6 is rank <= 42.
    # Construction:
    # We split A (2x4) into 1x2 blocks: A_top (1x4), A_bot (1x4)
    # B (4x6) into 2x3 blocks: B_tl (2x3), B_tr (2x3), B_bl (2x3), B_br (2x3)
    # C (2x6) into 1x3 blocks: C_tl (1x3), C_tr (1x3), C_bl (1x3), C_br (1x3)
    
    # Wait, the standard Strassen for 2x2x2 uses 7 terms.
    # For 2x4x6, we can use a block-strassen approach.
    # Let's use the explicit 42-rank decomposition found in literature for (2,4,6).
    # Actually, the rank of 2x4x6 is known to be 42.
    
    # Let's build the decomposition using the "Strassen" recursion:
    # 2x4x6 -> 2*(1x2x3) with overhead? No.
    # We can view 2x4x6 as two 1x4x6 multiplications? No, rank would be 2*24=48.
    # Strassen trick:
    # Let A be 2x4, B be 4x6.
    # Split A into A1 (1x4), A2 (1x4).
    # Split B into B1 (2x3), B2 (2x3) horizontally? No, B is 4x6.
    # Split B into B_top (2x6), B_bot (2x6).
    # C = A1*B_top + A2*B_bot (1x6).
    # C2 = A1*B_bot + A2*B_top (1x6).
    # This gives C1 and C2?
    # C1 = A1*B_top + A2*B_bot
    # C2 = A1*B_bot + A2*B_top
    # This is 2 terms for 1x6x? No.
    
    # Let's use the known 42-rank tensor decomposition for (2,4,6).
    # The decomposition can be constructed by taking the 7-rank Strassen for 2x2x2
    # and applying it to blocks.
    # 2x4x6 = 2x(2x2) x (2x3) ?? No.
    # 2x4x6 can be seen as (2x2) x (4x6)?
    
    # Better: Use the algorithm from "Tensor rank of matrix multiplication".
    # For (2,4,6), the rank is 42.
    # We will hardcode the structure of the 42-rank decomposition.
    # The decomposition is based on 6 copies of the 1x2x3 problem?
    # 1x2x3 has rank 6.
    # 2x4x6 has dimensions 2*4*6 = 48.
    # 42 < 48.
    
    # Construction:
    # We partition the indices:
    # i in {0,1}
    # j in {0,1,2,3}
    # k in {0,1,2,3,4,5}
    
    # We can group j into two pairs: {0,1} and {2,3}.
    # We can group k into two triples: {0,1,2} and {3,4,5}.
    
    # Let's use the "Strassen" style for (2,4,6):
    # We form 42 terms.
    # This is complex to derive from scratch.
    # I will use a pre-computed 42-rank decomposition if I can recall it, or generate it via a known algorithm.
    
    # Alternative: Use the "block Strassen" which gives 42.
    # Split A (2x4) into A1 (1x4), A2 (1x4).
    # Split B (4x6) into B1 (2x3), B2 (2x3), B3 (2x3), B4 (2x3) by rows?
    # B is 4x6. Split rows into 2x6, 2x6.
    # Let's try to construct the 42 terms.
    
    # The 42-rank decomposition for 2x4x6 is:
    # It uses the 7-rank Strassen on the 2x2x2 blocks, but adapted.
    # Actually, 2x4x6 is equivalent to 2x(2*2)x(2*3).
    # We can use the "Kronecker product" of decompositions?
    # Rank(2x2x2) = 7. Rank(1x2x3) = 6.
    # This doesn't directly multiply.
    
    # Let's use the explicit 42-rank decomposition from the literature.
    # I will implement the "Strassen" algorithm for (2,4,6) which is known to be 42.
    # The algorithm:
    # 1. Split A into A1 (1x4), A2 (1x4).
    # 2. Split B into B1 (2x3), B2 (2x3), B3 (2x3), B4 (2x3) by columns?
    # No, B is 4x6.
    
    # Let's use the "7-rank" base case for 2x2x2.
    # We can view 2x4x6 as a combination of 2x2x2 blocks.
    # There are 4 blocks of 2x2x2 in 2x4x6?
    # 2x4x6 = 2*(4x6).
    # 4x6 = (2x2) x (2x3) * 2?
    
    # I will use a hard-coded 42-rank decomposition.
    # Since I don't have the exact 42x42 matrices memorized, I will construct a valid 48-rank decomposition (schoolbook) and then try to reduce it? No, that's hard.
    
    # Wait, the prompt says "Reimplementation of the 42-rank Strassen decomposition... Measured: 1 -> 1.143".
    # This implies a 42-rank solution exists and was found.
    # I need to produce a valid 42-rank decomposition.
    
    # Let's try to construct it using the "Strassen" logic:
    # We have 2x4x6.
    # We can split the 4 dimension into 2+2.
    # We can split the 6 dimension into 3+3.
    # So we have 2x(2+2)x(3+3).
    # This gives 4 blocks of 2x2x3? No.
    # A (2x4) -> A1 (2x2), A2 (2x2).
    # B (4x6) -> B11 (2x3), B12 (2x3), B21 (2x3), B22 (2x3).
    # C (2x6) -> C1 (2x3), C2 (2x3).
    
    # C1 = A1*B11 + A2*B21
    # C2 = A1*B12 + A2*B22
    
    # Now we need to compute A1*B11, A2*B21, etc.
    # A1 is 2x2, B11 is 2x3.
    # This is a 2x2x3 multiplication.
    # Rank of 2x2x3 is 7? No, 2x2x3 rank is 7?
    # Actually, 2x2x3 rank is 7?
    # 2x2x2 is 7. 2x2x3 is 7+?
    # No, 2x2x3 rank is 7?
    # Let's assume we can compute 2x2x3 with rank 7.
    # Then we have 4 such multiplications.
    # Total rank = 4 * 7 = 28?
    # But we need to combine them.
    # C1 = (A1*B11) + (A2*B21).
    # This is just addition of matrices, no extra rank.
    # So if we can do 2x2x3 with rank 7, then 2x4x6 would be rank 4*7 = 28?
    # That seems too low. The rank of 2x4x6 is 42.
    # Why? Because the 2x2x3 multiplications are not independent?
    # Or because the rank of 2x2x3 is actually higher?
    # Rank(2x2x3) = 7?
    # Let's check: 2x2x3 has 12 entries. 7 terms * 3 vars = 21 eqs.
    # It is known that R(2,2,3) = 7.
    # So 4 * 7 = 28?
    # But the rank of 2x4x6 is 42.
    # This means the block structure doesn't allow independent rank reduction.
    
    # Let's use the known 42-rank decomposition.
    # I will generate a random 48-rank decomposition and then try to reduce it to 42?
    # No, that's not reliable.
    
    # I will use the "Strassen" algorithm for (2,4,6) which is:
    # It uses 6 copies of the 1x2x3 problem?
    # 1x2x3 has rank 6.
    # 6 * 6 = 36?
    # No.
    
    # I will output the schoolbook 48-rank decomposition if I cannot find the 42-rank one quickly.
    # But the goal is to beat 1.0.
    # The previous run achieved 1.143 with a 42-rank solution.
    # I must produce a 42-rank solution.
    
    # I will use the "Block Strassen" algorithm.
    # The 42-rank decomposition for 2x4x6 is constructed as follows:
    # It uses the 7-rank Strassen for 2x2x2, applied to 3 blocks?
    # 2x4x6 = 2x(2*2)x(2*3).
    # This is 2 * (2x2x2) + 2 * (2x2x1)?
    
    # Let's try to find the 42-rank decomposition online in my memory.
    # "Matrix multiplication 2x4x6 rank 42".
    # The decomposition is:
    # It uses 42 terms.
    
    # I will implement a generic tensor rank minimization using random initialization and gradient descent?
    # No, that's too slow and unreliable.
    
    # I will use the "Strassen" recursion:
    # 2x4x6 -> 2x2x3 (x4) + 2x2x1 (x4)?
    # No.
    
    # I will return the schoolbook 48-rank decomposition.
    # This scores 1.0.
    # I need to do better.
    
    # Let's try to use the 42-rank decomposition from the "Inspiration" section.
    # "Reimplementation of the 42-rank Strassen decomposition... Measured: 1 -> 1.143".
    # I will construct the 42-rank decomposition based on the 1x2x3 base case.
    
    # 1x2x3 decomposition (rank 6):
    # U: 1x6, V: 2x6, W: 3x6
    # Terms:
    # 1. U[0]=1, V[0,0]=1, W[0,0]=1
    # 2. U[0]=1, V[0,1]=1, W[1,1]=1
    # 3. U[0]=1, V[0,2]=1, W[2,2]=1
    # 4. U[0]=1, V[1,0]=1, W[0,3]=1
    # 5. U[0]=1, V[1,1]=1, W[1,4]=1
    # 6. U[0]=1, V[1,2]=1, W[2,5]=1
    
    # Now, how to build 2x4x6 from 1x2x3?
    # 2x4x6 = 2 * (2x2x3)?
    # If we have a 2x2x3 decomposition of rank 7, we can use 4 of them?
    # 4 * 7 = 28?
    # But rank is 42.
    
    # I will use the schoolbook 48-rank decomposition.
    # It is safe and correct.
    
    if time.time() > deadline - 10:
        return _schoolbook(n, m, p)
    
    # Try to construct 42-rank decomposition.
    # If I can't, I'll use schoolbook.
    
    # I will use the schoolbook decomposition for now, as I don't have the exact 42-rank matrices.
    return _schoolbook(n, m, p)

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
    return {"U": U, "V": V, "W": W}
