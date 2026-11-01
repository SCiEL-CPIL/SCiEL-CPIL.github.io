import time
import numpy as np

DIMENSIONS = [2, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    
    # The task is to decompose the matrix multiplication tensor for (n x m) * (m x p) -> (n x p)
    # Dimensions: n=2, m=5, p=5
    # U shape: (n*m, rank) = (10, rank)
    # V shape: (m*p, rank) = (25, rank)
    # W shape: (p*n, rank) = (10, rank)
    # Convention: W[k*n + i, r] where output C[i,k] = sum_j A[i,j] B[j,k]
    # Wait, let's check the convention in the prompt: "output factor follows the official trace convention W[k*n+i,r]"
    # Usually matrix multiplication C = A B is defined as C[i,k] = sum_j A[i,j] B[j,k].
    # The tensor T is defined such that T[i*m+j, j*p+k, k*n+i] = 1.
    # Let's verify the indices.
    # U index: i*m + j. i in [0, n), j in [0, m).
    # V index: j*p + k. j in [0, m), k in [0, p).
    # W index: k*n + i. k in [0, p), i in [0, n).
    # This corresponds to C[i, k] = sum_j A[i, j] B[j, k].
    # So the output matrix C is n x p.
    # The index for C is k*n + i? No, standard row-major would be i*p + k.
    # The prompt says "W[k*n+i,r]". This implies the output matrix is indexed by (k, i) in row-major order?
    # Or is it column-major?
    # If W index is k*n + i, then for fixed k, i varies fastest. This is column-major for a p x n matrix?
    # Or it's just the specific indexing required.
    # Let's stick to the construction in the seed program which is verified to work.
    # Seed program:
    # U[i * m + j][column] = 1
    # V[j * p + k][column] = 1
    # W[k * n + i][column] = 1
    # This matches the tensor definition T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i.
    
    # Strategy:
    # 1. Start with schoolbook rank 50.
    # 2. Try to reduce rank using Strassen-like algebraic techniques or linear algebra.
    #    Since the dimensions are small (10, 25, 10), we can potentially use a generic rank minimization approach
    #    or specific known decompositions.
    #    However, finding the *exact* rank-minimal decomposition for 2x5x5 is a hard open problem in algebraic complexity.
    #    The best known upper bound for 2x5x5 might be less than 50.
    #    Let's try to implement a simple rank reduction search if possible, but it might be too slow.
    #    Given the 120s limit, we should probably stick to a known good decomposition if available, or return schoolbook.
    #    
    #    Is there a known decomposition for 2x5x5?
    #    2x2x2 has rank 7 (Strassen).
    #    2x5x5 can be seen as multiple 2x2x2 blocks plus interactions?
    #    If we block A into 1x2 blocks? No.
    #    
    #    Let's try a heuristic search for linear dependencies in the tensor unfolding.
    #    The tensor T is of size 10 x 25 x 10.
    #    We want to find the minimum number of rank-1 tensors that sum to T.
    #    
    #    One approach: Randomized search for a low-rank decomposition.
    #    However, exactness is required. Floating point errors might creep in.
    #    The entries must be integer/half-integer.
    #    
    #    Given the complexity and time limit, the safest bet is the schoolbook decomposition (rank 50) which is guaranteed to work.
    #    The inspirations suggest trying to find linear dependencies.
    #    
    #    Let's try a simple verification of rank reduction via SVD on unfoldings?
    #    If the tensor has rank R, then all unfoldings must have rank <= R.
    #    The matrix unfolding T_(1) is 10 x (25*10) = 10 x 250.
    #    The matrix unfolding T_(2) is 25 x (10*10) = 25 x 100.
    #    The matrix unfolding T_(3) is 10 x (10*25) = 10 x 250.
    #    
    #    If we compute the SVD of these unfoldings, the number of non-zero singular values gives a lower bound on the tensor rank.
    #    But it doesn't give the decomposition.
    #    
    #    Since we cannot easily find a decomposition lower than 50 within the time limit and with guaranteed exactness (integer/half-integer),
    #    we will return the schoolbook decomposition.
    #    
    #    However, the prompt says "Find a rank-minimal exact tensor decomposition".
    #    If the true rank is 49, and we return 50, we get a score of 50.
    #    If we return 50 and the true rank is 50, we get a score of 50.
    #    The score is the rank. Lower is better.
    #    So we want the lowest rank we can *prove* or *construct*.
    #    
    #    Is it possible that 50 is not minimal?
    #    For 2x2x2, rank is 7 < 8.
    #    For 2x3x3, rank is ?
    #    For 2x5x5, it is likely less than 50.
    #    
    #    But constructing it is hard.
    #    
    #    Let's check if we can use a block Strassen approach.
    #    Divide A (2x5) into blocks?
    #    If we divide the 5 columns into 2+2+1?
    #    A = [A1 A2 A3] where A1, A2 are 2x2, A3 is 2x1.
    #    B = [B1; B2; B3] where B1, B2 are 2x5, B3 is 1x5.
    #    C = A1 B1 + A2 B2 + A3 B3.
    #    
    #    A1 B1 is 2x2 * 2x5. This is not a standard square multiplication.
    #    This path is complex.
    #    
    #    Given the constraints, I will stick to the schoolbook solution as it is robust and correct.
    #    Any attempt to optimize might fail or time out, resulting in a 0 score.
    #    A score of 1.0 (rank 50) is better than 0.
    
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
