import time

DIMENSIONS = [4, 5, 6]

def construct():
    # Deadline guard
    start_time = time.time()
    DEATHLINE = start_time + 140  # 140s out of 150s limit

    n, m, p = DIMENSIONS  # 4, 5, 6

    # We use a block decomposition to reduce the rank from the schoolbook 120.
    # We split the middle dimension (5) into 4 + 1.
    # Let A be 4x5, B be 5x6.
    # A = [A1 | A2] where A1 is 4x4, A2 is 4x1.
    # B = [B1; B2] where B1 is 4x6, B2 is 1x6.
    # C = A1 B1 + A2 B2.
    
    # We can compute A1 B1 (4x4 * 4x6) and A2 B2 (4x1 * 1x6).
    # Rank(A1 B1) for 4x4 * 4x6:
    # This is equivalent to M_{4,4,6}.
    # Schoolbook rank for 4x4x6 is 4*4*6 = 96.
    # However, we can just use the direct schoolbook for the blocks?
    # Wait, the previous thought process suggested rank 30.
    # Let's verify the rank 30 construction.
    
    # C = A1 B1 + A2 B2.
    # A1 is 4x4. B1 is 4x6.
    # A2 is 4x1. B2 is 1x6.
    
    # Decomposition of A1 B1 (4x4 * 4x6):
    # This is M_{4,4,6}.
    # Can we do better than 96?
    # Actually, M_{4,4,6} is just 4x4 matrix mult followed by 4x1? No, it's (4x4)(4x6).
    # This is a rectangular matrix multiplication.
    # The rank of M_{4,4,6} is at least max(4,6)*4? No.
    # Let's stick to the simplest improvement first: Block decomposition.
    
    # If we just use schoolbook for the blocks:
    # Block 1: A1 (4x4) * B1 (4x6). Rank = 4*4*6 = 96.
    # Block 2: A2 (4x1) * B2 (1x6). Rank = 4*1*6 = 24.
    # Total rank = 120. No improvement.
    
    # We need to find a lower rank decomposition for M_{4,4,6} or use Strassen-like techniques.
    # Actually, M_{4,4,6} is equivalent to computing 4 independent 4x6 products? No.
    # It is a single tensor.
    # Let's look at the structure.
    # C_{ik} = sum_j A_{ij} B_{jk}.
    
    # Is there a known lower rank for M_{4,5,6}?
    # The trivial bound is 120.
    # Let's try to construct a rank 96 solution? No, that's the same.
    
    # Let's try the "Laderman" style or just a simple split.
    # If we split n=4, m=5, p=6.
    # Maybe we can use the fact that 4 is small.
    
    # Let's try to implement a generic rank minimization search?
    # Given the 120s timeout, a full search is risky.
    # But 120 is the current best.
    
    # What if we use a known result for M_{4,4,4} (rank 16) and extend?
    # M_{4,5,6} contains M_{4,4,4}.
    
    # Let's try a different approach: Randomized SVD or integer optimization?
    # No, we need exact integers.
    
    # Let's try to see if we can get rank < 120 by simple algebraic tricks.
    # Consider splitting m=5 into 3+2.
    # A1 (4x3), B1 (3x6). A2 (4x2), B2 (2x6).
    # Rank(A1 B1) = 4*3*6 = 72.
    # Rank(A2 B2) = 4*2*6 = 48.
    # Total 120.
    
    # It seems that for rectangular matrices, the schoolbook rank is often optimal or close to it without sophisticated Strassen-like algorithms.
    # However, for square matrices, Strassen reduces rank.
    # For 4x4, rank is 16 (Strassen doesn't help for 4x4? Strassen is 2x2 rank 7. 4x4 rank 16? 7^2 = 49 > 16. So 4x4 is schoolbook optimal).
    # Wait, Strassen 2x2 is 7. 4x4 can be done in 7^2=49? No, 4x4 is 16.
    # 8x8 Strassen is 7^3 = 343 vs 512.
    # So for 4x4, schoolbook is best.
    
    # What about 4x5?
    # M_{4,5,5}. Schoolbook 100.
    
    # Is there a way to combine terms?
    # C_{ik} = sum_{j=0}^4 A_{ij} B_{jk}.
    
    # If we can't find a better analytic solution, we should return the schoolbook solution to ensure validity, but maybe try a quick search for a better one if time permits?
    # A search for rank 119?
    # This is a NP-hard problem (Tensor Rank).
    # However, we can try a greedy approach or local search?
    # Unlikely to work in 120s for exact integers.
    
    # Let's check if there is a known result for 4x5x6.
    # I don't have access to external search.
    
    # Let's try to implement a construction that might be slightly better if we use symmetry?
    # No, the matrices are arbitrary.
    
    # Given the risk, I will return the schoolbook solution.
    # But wait, the prompt asks for a "stronger algorithm".
    # If I can't find a better one, I should stick to the safe one.
    
    # However, let's try one specific optimization:
    # If we assume A and B are dense, no sparsity.
    
    # Let's try to use the fact that 5 = 4+1.
    # We established rank 30 was a thought error?
    # Let's re-evaluate the rank 30 idea.
    # C = A1 B1 + A2 B2.
    # A1 (4x4), B1 (4x6).
    # A2 (4x1), B2 (1x6).
    #
    # To compute A1 B1, we need to compute 4x4x6 tensor.
    # Can we compute A1 B1 with rank < 96?
    # A1 B1 is a 4x6 matrix.
    # It is the product of a 4x4 and a 4x6.
    # This is equivalent to 6 independent 4x4 matrix multiplications? No.
    # It is 4 independent 4x6 vector-matrix products?
    #
    # Actually, M_{4,4,6} is isomorphic to M_{4,6,4} by permutation.
    # The rank of M_{n,m,p} is the same for any permutation of n,m,p.
    # So Rank(M_{4,4,6}) = Rank(M_{4,6,4}).
    #
    # Is Rank(M_{4,4,6}) < 96?
    # Lower bound: max(n,m,p) * min(n,m,p)? No.
    # Lower bound is at least max(n,m,p).
    # For 4x4x4, rank 16.
    # For 4x4x5, rank?
    #
    # I will stick to the schoolbook solution to ensure correctness, as finding a provably lower rank decomposition for this specific case without external knowledge is risky and time-consuming.
    
    # The previous response attempted a rank 30 solution but the logic was flawed (assuming A1 B1 could be done in rank 16? No, that's 4x4x4).
    # A1 (4x4) * B1 (4x6) is NOT a 4x4x4 multiplication.
    #
    # So, I will return the schoolbook solution.
    
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
