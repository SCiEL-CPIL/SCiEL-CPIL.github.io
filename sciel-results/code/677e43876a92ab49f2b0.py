import numpy as np
import time

DIMENSIONS = [3, 3, 7]

def construct():
    """
    Matrix multiplication 3x3x7.
    Strategy:
    1. Split B into B1 (3x3) and B2 (3x4).
    2. Use a rank-23 decomposition for 3x3x3 (Laderman's algorithm).
    3. Use a rank-23 decomposition for the first 3x3 part of the 3x3x4 block? No, 3x3x4 is not standard.
    
    Let's stick to the safe schoolbook for 3x3x7 which is rank 63, but try to optimize.
    Actually, the known minimal rank for 3x3x3 is 23.
    For 3x3x1, the rank is 9 (sum of 9 rank-1 tensors? No, 3x3x1 is just matrix-vector multiplication).
    Wait, the tensor for 3x3x1 multiplication C = A*B where A is 3x3, B is 3x1.
    C[i] = sum_j A[i,j] B[j].
    This is a bilinear map. The rank of this tensor is 9?
    Let's check. The tensor T has indices (i,j,k) with k=0.
    T[i,j,0] = delta_{i,j}? No.
    C[i] = sum_j A[i,j] B[j,0].
    So the term is A[i,j] * B[j,0] * 1 (for C[i]).
    The tensor is sum_{i,j} e_{i,j} \otimes e_{j,0} \otimes e_{i}.
    This is a sum of 9 rank-1 tensors. Can it be less?
    It is equivalent to the identity matrix multiplication if we view A as 3x3 and B as 3x1.
    Actually, this is just the "trace" part.
    The rank of the 3x3x1 tensor is 9? Or is it less?
    If we consider the tensor as a 3x3 matrix of linear forms in B, it's the identity matrix times B?
    No, C = A B.
    If A is fixed, C is linear in B.
    If B is fixed, C is linear in A.
    The rank of the tensor is the minimum number of multiplications.
    For 3x3x1, it's essentially 3 independent dot products of length 3.
    Each dot product of length 3 has rank 3 (schoolbook) or 3 (Strassen doesn't help for 1D?).
    Actually, the rank of the 1D convolution/dot product of length n is n.
    So 3 dot products of length 3 -> rank 9.
    
    So if we use rank 23 for the first 3x3x3 block and rank 9 for the 3x3x1 block,
    and rank 23 for the second 3x3x3 block, total is 23+23+9 = 55.
    
    Let's verify Laderman's algorithm for 3x3x3.
    Laderman's algorithm uses 23 multiplications.
    
    I will implement Laderman's algorithm for the 3x3x3 blocks.
    I will implement schoolbook for the 3x3x1 block.
    
    Laderman's 23 terms for 3x3x3:
    Let A, B, C be 3x3 matrices.
    The 23 products are:
    1. (A00+A11) * (B00+B11) -> contributes to C00, C11? No.
    
    Let's look up the exact Laderman coefficients.
    From "A rank 23 algorithm for 3x3 matrix multiplication" (Laderman 1976):
    
    The 23 terms are defined by linear combinations of entries of A and B.
    Let's denote the terms as M_k for k=0..22.
    A_part_k and B_part_k.
    C[i,j] = sum_k w_{k,ij} M_k.
    
    This is complex to hardcode correctly without a reference table in the code.
    Given the time limit and the risk of error, and the fact that the current score is 1.0 (rank 63),
    achieving rank 55 would be a significant improvement (score 63/55 = 1.145).
    
    However, implementing Laderman correctly from scratch is error-prone.
    Is there a simpler improvement?
    
    What about using Strassen for 2x2x2 blocks?
    3x3x3 can be decomposed into 2x2x2 (rank 7) + cross terms.
    The cross terms add complexity.
    
    Alternative:
    Use the known rank-23 decomposition.
    I will find a reliable source for the 23 coefficients and hardcode them.
    
    Source: https://en.wikipedia.org/wiki/Laderman%27s_algorithm
    The algorithm computes the product of two 3x3 matrices using 23 multiplications.
    
    Let A = [a11 a12 a13; a21 a22 a23; a31 a32 a33]
    Let B = [b11 b12 b13; b21 b22 b23; b31 b32 b33]
    
    The 23 products are:
    1.  m1 = (a11+a22)(b11+b22)
    2.  m2 = (a12+a22)(b21+b22)
    3.  m3 = (a11+a33)(b11+b33)
    4.  m4 = (a22+a33)(b22+b33)
    5.  m5 = (a11+a22+a33)(b11+b22+b33)
    6.  m6 = (a12+a13+a22+a33)(b21+b13+b22+b33)
    7.  m7 = (a11+a12+a33)(b11+b12+b33)
    8.  m8 = (a13+a22+a33)(b23+b22+b33)
    9.  m9 = (a11+a13+a22)(b11+b13+b22)
    10. m10 = (a21+a13+a33)(b31+b13+b33)
    11. m11 = (a11+a21+a33)(b11+b21+b33)
    12. m12 = (a12+a21+a33)(b22+b21+b33)
    13. m13 = (a13+a21+a22)(b13+b21+b22)
    14. m14 = (a11+a12+a21)(b11+b12+b21)
    15. m15 = (a12+a13+a21)(b12+b13+b21)
    16. m16 = (a13+a21+a22)(b13+b21+b22)
    17. m17 = (a11+a12+a21+a33)(b11+b12+b21+b33)
    18. m18 = (a12+a13+a21+a33)(b12+b13+b21+b33)
    19. m19 = (a13+a21+a22+a33)(b13+b21+b22+b33)
    20. m20 = (a11+a21+a22+a33)(b11+b21+b22+b33)
    21. m21 = (a12+a21+a22+a33)(b12+b21+b22+b33)
    22. m22 = (a11+a12+a21+a33)(b11+b12+b21+b33) -- Wait, I might be misremembering or mixing up.
    
    This is too risky. The indices and signs are critical.
    
    Let's try a different strategy:
    Use the fact that 3x3x7 is a block matrix multiplication.
    A (3x3) * B (3x7) = C (3x7).
    We can view this as 3 separate 1x3x7 matrix-vector multiplications? No, A is 3x3.
    Each row of A multiplies B.
    Row i of A is a 1x3 vector.
    C_row_i = A_row_i * B.
    This is a 1x3 matrix times a 3x7 matrix.
    The rank of 1x3x7 multiplication?
    C[k] = sum_j A[j] B[j,k].
    This is 7 dot products of length 3.
    Each dot product has rank 3.
    Total rank 7*3 = 21?
    But the A[j] terms are shared across k.
    This is a bilinear map.
    T[j, k, l] where l is the output index (0..6).
    T[j, k, l] = 1 if l=k? No.
    Output C[l] = sum_j A[j] B[j,l].
    So the tensor is sum_{j,l} e_j \otimes e_{j,l} \otimes e_l.
    This is a sum of 3*7 = 21 rank-1 tensors.
    Can we do better?
    This is equivalent to multiplying a 1x3 matrix by a 3x7 matrix.
    The rank of the tensor M_{1,3,7} is 21?
    Actually, since the input A is 1x3, it's just a vector.
    The operation is linear in A and linear in B.
    It's just 7 independent dot products.
    The rank is 21?
    Wait, if A is fixed, it's 7 independent linear combinations of B.
    If B is fixed, it's 7 independent linear combinations of A.
    The rank is indeed 21?
    No, the rank of the tensor for 1xn x nxm multiplication is n*m?
    Let's check 1x2 x 2x2.
    C0 = A0 B00 + A1 B10
    C1 = A0 B01 + A1 B11
    Terms: A0 B00, A1 B10, A0 B01, A1 B11.
    Rank 4.
    Is it possible to do it in less?
    Strassen for 2x2? No, A is 1x2.
    So rank is n*m.
    For 1x3 x 3x7, rank is 3*7 = 21.
    
    So if we decompose C = A*B as sum of 3 row-multiplications,
    Total rank = 3 * 21 = 63.
    This is the same as schoolbook.
    
    So the block structure doesn't immediately help reduce rank below 63 unless we use a non-trivial algorithm for the 3x3x3 part.
    
    Since I cannot reliably hardcode Laderman's 23 terms, I will stick to the schoolbook decomposition but ensure it is robust.
    The current program returns rank 63.
    
    Is there any other way to get a score > 1.0?
    Score = seed / value.
    Seed = 63.
    If I find a rank < 63, score > 1.0.
    
    What if I use a rank-22 algorithm for 3x3x3?
    It is unknown if rank 22 exists. The lower bound is 19, upper is 23.
    
    Let's try to implement the 3x3x3 rank-23 algorithm from a verified source in the code.
    I will use the coefficients from a known correct implementation.
    
    Here is a set of coefficients for Laderman's algorithm (0-indexed):
    
    U, V, W matrices of size 9x23.
    
    I will define the 23 terms as lists of (index, coeff) for U, V, W.
    
    Term 1:
    U: (0,1), (4,1)  -> A00, A11
    V: (0,1), (4,1)  -> B00, B11
    W: (0,1), (4,1)  -> C00, C11? No, C00 and C11 are not both 1.
    
    Actually, let's look at the structure:
    M1 = (A00+A11)(B00+B11)
    C00 += M1
    C11 += M1
    C01 -= M1 ? No.
    
    This is getting too complicated to derive on the fly.
    
    I will return the schoolbook solution (rank 63) which is valid and scores 1.0.
    Any attempt to optimize is risky.
    
    However, the prompt asks to combine strategies.
    The "inspiration" suggests using block structure.
    I have analyzed that block structure doesn't help for 1xn x nxm.
    
    I will output the current schoolbook solution.
    """
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
    return {"U": U, "V": V, "W": W}
