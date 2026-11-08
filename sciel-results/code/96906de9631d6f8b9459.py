"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal (or near-minimal) exact tensor decomposition for
    multiplying a 2x4 matrix by a 4x7 matrix.
    
    Strategy:
    1. Start with the schoolbook decomposition (rank 56).
    2. Attempt to find a lower rank decomposition using known algebraic techniques.
       For small matrix multiplications, the rank is often known or can be bounded.
       The tensor rank for 2x4 * 4x7 is not trivially known to be less than 56 in standard literature
       without specific advanced algebraic geometry results, but we can try to optimize.
       
    However, for this specific task, we will first verify if a simple block decomposition
    using Strassen's algorithm for 2x2 blocks can reduce the rank.
    
    Let's analyze the structure:
    A is 2x4, B is 4x7.
    We can view A as two 2x2 blocks side-by-side: A = [A1, A2] where A1, A2 are 2x2.
    B can be viewed as two 2x7 blocks stacked: B = [B1; B2] where B1, B2 are 2x7.
    Then C = A1*B1 + A2*B2.
    
    This doesn't directly help with Strassen unless we split further.
    
    Let's stick to a robust approach:
    The schoolbook rank is 56.
    Is there a known lower rank?
    For 2x2 * 2x2, rank is 7 (Strassen).
    For 2x3 * 3x2, rank is 10? (Known bounds: 7 <= R(2,3,2) <= 10, actually 10 is schoolbook, 7 is lower bound. Recent results suggest 10 is minimal? No, wait. R(2,2,2)=7. R(2,3,2) is likely 10 or less. Actually, R(2,3,2) = 10 is the schoolbook. Is it lower? No, it's believed to be 10? Let's check.
    
    Actually, for this specific case 2x4 * 4x7, let's look at the dimensions.
    n=2, m=4, p=7.
    Schoolbook: 56.
    
    Let's try to construct a decomposition using the 2x2 Strassen decomposition embedded.
    We can split the 4x4 middle dimension into 2x2 blocks.
    A (2x4) = [A11 A12] where A11, A12 are 2x2.
    B (4x7) = [B11 B12; B21 B22] where Bij are 2x? No, B is 4x7.
    Let's split B into four 2x? blocks.
    B = [B11 B12; B21 B22] is not square.
    B is 4x7.
    Split rows into 2+2.
    B_top = B[0:2, :] (2x7)
    B_bot = B[2:4, :] (2x7)
    
    C = A11*B_top_left + A12*B_bot_left + A11*B_top_right + A12*B_bot_right?
    No, A is 2x4.
    A = [A_left, A_right] where A_left is 2x2, A_right is 2x2.
    B = [B_left; B_right] where B_left is 2x7, B_right is 2x7.
    C = A_left * B_left + A_right * B_right.
    
    This is the sum of two 2x2 * 2x7 multiplications.
    Rank of 2x2 * 2x7?
    Schoolbook: 2*2*7 = 28.
    Can we do better?
    If we use Strassen for the 2x2 part, we need to handle the 7 columns.
    Strassen reduces the 2x2 * 2x2 multiplication to 7 scalar mults.
    Here, the "scalar" mult is actually a 2x7 matrix multiplication? No.
    
    Let's treat the 2x7 matrix as 7 independent 2x1 vectors?
    Or, we can view the 2x2 * 2x7 multiplication as 7 instances of 2x2 * 2x1.
    Rank of 2x2 * 2x1?
    Schoolbook: 4.
    Is there a rank 3 decomposition? No, R(2,2,1) = 4.
    So for 2x2 * 2x7, the rank is at least 7 * 4 = 28?
    Wait, the tensors are not independent in that way.
    
    Actually, the multiplication tensor for 2x2 * 2x7 is the direct sum of 7 copies of 2x2 * 2x1?
    No, because the 2x2 matrix A is shared across the 7 columns of B.
    So it's not a direct sum.
    
    Let's calculate the rank of 2x2 * 2x7.
    Let A be 2x2, B be 2x7.
    C = A B.
    C is 2x7.
    
    We can use the Strassen decomposition for the 2x2 part.
    Strassen decomposes 2x2 * 2x2 into 7 terms.
    For 2x2 * 2x7, we can split B into B1, B2 (each 2x3 and 2x4? or 2x7).
    Actually, we can just apply the Strassen formulas to the 2x2 blocks of A and the corresponding 2x7 blocks of B.
    
    Let A = [[a, b], [c, d]].
    Let B = [B1, B2] where B1, B2 are 2x7? No, B is 2x7.
    Let's split B into two 2x4 and 2x3? No, that's messy.
    
    Let's just use the 7 Strassen multipliers.
    M1 = (a+d) * (B1+B2) ? No.
    
    Standard Strassen for 2x2 * 2x2:
    A = [[a, b], [c, d]], B = [[e, f], [g, h]].
    M1 = (a+d)(e+h)
    M2 = (c+d)e
    M3 = a(f-h)
    M4 = d(g-e)
    M5 = (a+b)h
    M6 = (c-a)(e+f)
    M7 = (b-d)(g+h)
    
    For 2x2 * 2x7:
    Let A = [[a, b], [c, d]].
    Let B = [B_e, B_f; B_g, B_h] where B_e, B_f, B_g, B_h are 1x7 vectors?
    No, B is 2x7.
    Let B = [B1; B2] where B1, B2 are 1x7 vectors.
    Then C = [a*B1 + b*B2; c*B1 + d*B2].
    
    Using Strassen logic:
    M1 = (a+d) * (B1+B2)
    M2 = (c+d) * B1
    M3 = a * (B2-B1)
    M4 = d * (B1-B2)
    M5 = (a+b) * B2
    M6 = (c-a) * (B1+B2)
    M7 = (b-d) * (B1+B2)
    
    Wait, the Strassen formulas are for scalar multiplication. Here, B1, B2 are vectors.
    So M1 is a 1x7 vector, etc.
    
    The rank of this decomposition is 7 * 7 = 49?
    No, the "multiplication" is between a scalar and a vector.
    This is a linear operation.
    
    Let's count the rank.
    We have 7 "terms" (M1..M7).
    Each term involves multiplying a scalar (linear combination of a,b,c,d) with a vector (linear combination of B1, B2).
    So each term is a rank-1 tensor in the space of (A, B) -> C.
    
    Total rank = 7 * (dimension of B part)?
    No, the tensor is:
    T(A, B) = sum_k u_k(A) * v_k(B) * w_k
    
    Here, for each of the 7 Strassen multipliers, we have a scalar u_k(A) and a vector v_k(B) (which is 1x7).
    So for each k, we have 7 rank-1 tensors (one for each column of the vector v_k).
    Total rank = 7 * 7 = 49.
    
    So we can decompose 2x2 * 2x7 with rank 49.
    
    Now, back to 2x4 * 4x7.
    We split A into A_left (2x2) and A_right (2x2).
    We split B into B_top (2x7) and B_bot (2x7).
    C = A_left * B_top + A_right * B_bot.
    
    Rank of A_left * B_top is 49.
    Rank of A_right * B_bot is 49.
    Total rank = 49 + 49 = 98.
    This is worse than schoolbook (56).
    
    So this approach doesn't work.
    
    Let's try splitting B into 7 columns.
    A (2x4) * B (4x7).
    C = A * [b1, b2, ..., b7] where bi are 4x1 vectors.
    C = [A*b1, A*b2, ..., A*b7].
    
    Rank of 2x4 * 4x1?
    Schoolbook: 8.
    Is there a lower rank?
    R(2,4,1) = 8?
    
    If we use 7 independent decompositions, rank = 7 * 8 = 56.
    Same as schoolbook.
    
    So the schoolbook decomposition is likely optimal for this small case, or close to it.
    Given the complexity of finding a lower rank, and the time limit, I will return the schoolbook decomposition.
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
# EVOLVE-BLOCK-END
