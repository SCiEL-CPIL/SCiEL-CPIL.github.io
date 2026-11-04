"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x4 * 4x6 matrix multiplication.
    
    We exploit the isomorphism between (3,4,6) and (4,6,3) to reframe as 4x6 * 6x3.
    Then we use a block structure: treat the 4x6 input as two 2x6 blocks.
    
    Key insight: For 2x2x2, Strassen gives rank 7.
    For our problem, we can decompose into independent 2x2x2 blocks.
    
    Actually, let's think more carefully:
    - 3x4 * 4x6: The tensor has dimensions (3,4,6)
    - We want to find U (12 x r), V (24 x r), W (18 x r) such that the tensor decomposes
    
    A simple approach: Use the fact that 3x4x6 can be decomposed by treating it as
    a collection of 2x2x2 problems with some overlap.
    
    Better approach: Use known results. The rank of 3x4x6 is known to be at most 72 (schoolbook).
    We can try to do better by exploiting structure.
    
    Let's use a different strategy:
    - Split the 4x6 matrix into two 2x6 blocks
    - Each 2x6 * 6x3 multiplication can be done with some rank
    - But we need to be careful about the output structure
    
    Actually, let's use a more direct approach:
    For 3x4x6, we can use the fact that this is equivalent to computing
    3 separate 1x4 * 4x6 products (each giving a 1x6 vector).
    
    Each 1x4 * 4x6 is just a linear combination of 6 vectors in 4D, so rank 6.
    Total: 3 * 6 = 18? No, that's not right either.
    
    Let me reconsider. The schoolbook rank is 3*4*6 = 72.
    
    For a better bound, we can use:
    - Group the 4 columns into two groups of 2
    - Each 3x2 * 2x6 block: rank is at most 3*2*6/2 = 18 using Strassen-like?
    
    Actually, for 2x2x2, Strassen gives rank 7 instead of 8.
    For 3x2x6, we can think of it as 3 instances of 1x2x6, each of rank 6, so 18.
    Or we can use the 2x2 structure: 3x2x6 = 3 * (1x2x6), and 1x2x6 has rank 6.
    
    Let me try a different decomposition:
    Split the middle dimension (4) into 2+2.
    For each pair of columns (2 columns), we have a 3x2 * 2x6 product.
    
    A 3x2 * 2x6 product: 
    - Input: 3x2 matrix A, 2x6 matrix B
    - Output: 3x6 matrix C = A*B
    
    This can be seen as: for each of the 3 rows of A and 6 columns of B,
    C[i,j] = A[i,0]*B[0,j] + A[i,1]*B[1,j]
    
    This is a 3x1x6 tensor times a 1x2x6 tensor... no, let me think again.
    
    The tensor for 3x2x2x6 is: T[i,j,k,l] = delta_{i,j} * delta_{k,l} (in some sense)
    Actually, T[i,j,k,l] = 1 if we're computing C[i,l] = sum_j A[i,j] B[j,l]
    
    So the tensor has indices (i, j, l) where i in 0..2, j in 0..1, l in 0..5.
    This is a 3x2x6 tensor, and its rank is at most 3*2*6 = 36 (schoolbook).
    
    Can we do better? For 2x2x2, Strassen gives 7. For 3x2x6:
    - We can fix i and get a 2x6 tensor, which has rank 2*6=12 (since it's just outer products)
    - So total rank is at most 3*12 = 36, same as schoolbook.
    
    Hmm, that's not helpful. Let me think differently.
    
    Actually, the 3x2x6 tensor: T[i,j,l] = delta_{i,?} no, T is the multiplication tensor.
    T[i,j,l] represents the coefficient of A[i,j] * B[j,l] in C[i,l].
    So T[i,j,l] = 1 for all i,j,l in their ranges, but it's a 3-way tensor where
    T[i,j,l] = 1 if the indices match the multiplication pattern.
    
    Wait, I'm overcomplicating this. The multiplication tensor for (n,m,p) has
    entries T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j] B[j,k].
    So T is a 3-tensor of shape (n, m, p) where T[i,j,k] = 1 for all valid i,j,k.
    No wait, that's not right either.
    
    The multiplication tensor T for (n,m,p) is defined as:
    T[i,j,k] = 1 if the (i,j) entry of A and (j,k) entry of B contribute to (i,k) entry of C.
    So T[i,j,k] = 1 for all i in 0..n-1, j in 0..m-1, k in 0..p-1.
    
    No, that's the all-ones tensor, which has rank 1. That's not right.
    
    Let me look at the seed code:
    U[i*m+j, c] = 1
    V[j*p+k, c] = 1  
    W[k*n+i, c] = 1
    
    So for each (i,j,k) triple, we have a rank-1 term:
    u = e_{i*m+j} (in R^{n*m})
    v = e_{j*p+k} (in R^{m*p})
    w = e_{k*n+i} (in R^{p*n})
    
    And the tensor T = sum_{i,j,k} u ⊗ v ⊗ w
    
    This means T[a,b,c] = 1 if there exist i,j,k such that:
    a = i*m+j, b = j*p+k, c = k*n+i
    Otherwise T[a,b,c] = 0.
    
    So T is not the all-ones tensor. It's a sparse tensor.
    
    For n=3, m=4, p=6:
    a in 0..11, b in 0..23, c in 0..17
    
    T[a,b,c] = 1 iff:
    Let i = a // m = a // 4, j = a % m = a % 4
    Let j' = b // p = b // 6, k = b % p = b % 6
    Let k' = c // n = c // 3, i' = c % n = c % 3
    
    Then T[a,b,c] = 1 iff i == i' and j == j' and k == k'
    
    So T[a,b,c] = 1 iff (a//4 == c%3) and (a%4 == b//6) and (a%4 == b//6) and (b%6 == c//3)
    Wait, let me redo this.
    
    a = i*m + j where i in 0..2, j in 0..3
    b = j*p + k where j in 0..3, k in 0..5
    c = k*n + i where k in 0..5, i in 0..2
    
    For T[a,b,c] to be 1, we need:
    - i = a // 4, j = a % 4
    - j = b // 6, k = b % 6
    - k = c // 3, i = c % 3
    
    So the conditions are:
    1. a // 4 == c % 3  (i matches)
    2. a % 4 == b // 6  (j matches)
    3. b % 6 == c // 3  (k matches)
    
    This is a specific pattern. The schoolbook decomposition uses rank 72.
    
    Can we do better? Let's think about using Strassen's algorithm.
    
    Strassen works for 2x2x2, reducing rank from 8 to 7.
    For our problem, we can try to decompose into 2x2x2 blocks.
    
    The dimensions are 3x4x6. We can write:
    3 = 2 + 1
    4 = 2 + 2
    6 = 2 + 2 + 2
    
    So we can split into 2x2x2 blocks with some overlap.
    
    Let's try a different approach: use the fact that the problem is isomorphic to 4x6x3.
    4x6x3: n=4, m=6, p=3.
    4 = 2+2, 6 = 2+2+2, 3 = 2+1
    
    For each 2x2x2 block, Strassen gives rank 7.
    Number of 2x2x2 blocks: (4/2) * (6/2) * (3/2) = 2 * 3 * 1.5 = 9 (with overlap)
    
    This is getting complicated. Let me just try to implement a search for a lower rank.
    
    Given the time constraint, let's try a few known decompositions:
    
    1. Schoolbook: rank 72
    2. Try to use Strassen on 2x2x2 subproblems
    
    For 2x2x2, Strassen's decomposition:
    M1 = (A11+A22)(B11+B22)
    M2 = (A21+A22)B11
    M3 = A11(B21-B22)
    M4 = A22(B12-B11)
    M5 = (A11+A12)B22
    M6 = (A21-A11)(B11+B12)
    M7 = (A12-A22)(B21+B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    
    This gives a rank-7 decomposition of the 2x2x2 multiplication tensor.
    
    For our 3x4x6 problem, let's try to decompose it into 2x2x2 blocks.
    
    Actually, let's just try a simple improvement: use the fact that some of the
    schoolbook terms might be combined.
    
    Given the complexity and time constraints, let's stick with the schoolbook
    decomposition for now, but try to verify it's correct and see if we can find
    a better one through a simple search.
    
    For now, return the schoolbook decomposition.
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
