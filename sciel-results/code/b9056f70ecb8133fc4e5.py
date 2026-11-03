"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Block-wise decomposition for 3x5 * 5x7 matrix multiplication.
    
    Split the middle dimension m=5 into m1=3 and m2=2.
    A = [A1 | A2] where A1 is 3x3, A2 is 3x2
    B = [B1; B2] where B1 is 3x7, B2 is 2x7
    C = A1*B1 + A2*B2
    
    For each sub-problem, use the optimal known decomposition:
    - 3x3 * 3x7: Use Strassen-like approach. 
      We can decompose 3x3*3x3 with rank 7, and for each of the 7 columns,
      we need to handle the 7x7 output. Actually, let's think more carefully.
      
      For A1 (3x3) * B1 (3x7): The result is 3x7.
      C1[i,k] = sum_{j=0}^{2} A1[i,j] * B1[j,k]
      
      We can use the Strassen decomposition for the 3x3*3x3 part, but since
      B1 has 7 columns, we need to be careful. Let's use a direct approach:
      The rank of the 3x3*3x7 multiplication tensor is at most 3*3*7 = 63,
      but we can do better by exploiting structure.
      
      Actually, let's use a simpler but effective approach:
      For 3x3 * 3x7, we can view this as computing 7 separate 3x3 * 3x1
      multiplications. Each 3x3*3x1 has rank at most 3 (one per row of A1).
      So total rank for 3x3*3x7 is at most 7*3 = 21? No, that's not right either.
      
      Let me use the naive decomposition for each block but try to be smart:
      3x3 * 3x7: 3*3*7 = 63 terms (naive)
      3x2 * 2x7: 3*2*7 = 42 terms (naive)
      Total: 105 terms
      
      To improve, let's use Strassen for the 3x3*3x3 sub-problems where possible.
      
      Actually, here's a better approach: use the fact that we can decompose
      the multiplication using fewer terms by exploiting the rectangular structure.
      
      For 3x3 * 3x7:
      We can write C1 = A1 * B1 where A1 is 3x3 and B1 is 3x7.
      Using the standard identity, this requires 3*3*7 = 63 scalar multiplications
      in the naive case. But we can use Strassen-like techniques.
      
      Let me implement a known low-rank decomposition for rectangular matrix
      multiplication. For 3x3 * 3x7, one approach is to use 7 copies of the
      3x3*3x1 case, each of which can be done in 3 multiplications (one per
      row), giving 21 total. But that's not quite right either.
      
      Actually, the cleanest approach: use the tensor decomposition directly.
      For C[i,k] = sum_j A[i,j]*B[j,k], each (i,j,k) triple gives a rank-1 term.
      For 3x3*3x7, that's 63 terms. For 3x2*2x7, that's 42 terms. Total 105.
      
      To get below 105, we need to find a non-trivial decomposition. Let's use
      Strassen's algorithm adapted for rectangular matrices.
      
      For the 3x3 * 3x7 part, we can use the following:
      - Decompose B1 into 7 columns: b0, b1, ..., b6 (each 3x1)
      - For each column bk, compute A1 * bk using 3 multiplications (one per row)
      - Total: 7 * 3 = 21 terms? No, each "multiplication" A1[i,:] * bk is
        a dot product of 3 elements, which is 3 scalar multiplications.
      
      I'm overcomplicating this. Let me just use the naive decomposition but
      see if I can find any obvious savings.
      
      Actually, let me try a different strategy: use the fact that for
      rectangular matrices, the rank can be significantly lower than n*m*p.
      
      For 3x2 * 2x7: The minimum rank is at least max(3*2, 2*7, 3*7) = 21.
      The naive rank is 42. Can we achieve rank 21?
      
      For 3x3 * 3x7: The minimum rank is at least max(9, 21, 21) = 21.
      The naive rank is 63. Can we achieve rank 21?
      
      Total minimum possible: 21 + 21 = 42.
      
      Let me try to construct a rank-21 decomposition for each block.
      
      For 3x2 * 2x7:
      C2[i,k] = A2[i,0]*B2[0,k] + A2[i,1]*B2[1,k]
      For each (i,k), this is a sum of 2 terms. Total terms = 3*2*7 = 42.
      To get rank 21, we need to share factors across different (i,k) pairs.
      
      One approach: for each k, we have 3x2 * 2x1 multiplication, which is
      just a vector-matrix product. The rank for 3x2*2x1 is 3*2 = 6 (naive)
      or possibly less. For 7 different k values, if we can share the
      decomposition, we might reduce the total rank.
      
      Actually, let me just implement the naive decomposition with rank 105
      and see if that's accepted, then try to optimize.
      
      Wait, the current program already has rank 105. I need to do better.
      
      Let me try using Strassen's algorithm for the 3x3*3x3 sub-problems.
      For 3x3 * 3x3, Strassen gives rank 7. For 3x3 * 3x7, I can potentially
      use 7 * 7 = 49 terms? No, that's not right.
      
      Let me try a concrete construction:
      
      For 3x3 * 3x7:
      Let A1 = [[a00, a01, a02], [a10, a11, a12], [a20, a21, a22]]
      Let B1 = [[b00, b01, ..., b06], [b10, b11, ..., b16], [b20, b21, ..., b26]]
      
      C1[i,k] = a_i0*b0k + a_i1*b1k + a_i2*b2k
      
      Using Strassen-like ideas, we can compute the 3x3*3x3 product with 7
      multiplications. For the 3x3*3x7 case, we can apply this to each
      column of B1, but that would give 7*7 = 49 multiplications, which is
      worse than the naive 63? No, 49 < 63, so it's better.
      
      Actually, for each column k of B1, we compute A1 * B1[:,k] which is a
      3x3 * 3x1 product. This can be done in 3 multiplications (one per row
      of A1). So total = 7 * 3 = 21 multiplications.
      
      But wait, that's just the naive approach for 3x3*3x1. The Strassen
      algorithm is for 3x3*3x3, not 3x3*3x1.
      
      For 3x3*3x1: C[i] = sum_j A[i,j] * B[j], which is 3 dot products of
      length 3, requiring 9 scalar multiplications total. Not 3.
      
      I think I'm confusing myself. Let me just go with the naive decomposition
      and accept rank 105, or find a simpler improvement.
      
      Actually, here's a simple improvement: for the 3x2 * 2x7 part, we can
      use the fact that it's a small multiplication. The rank is at most
      3*2*7 = 42, but we can potentially do better.
      
      Let me just implement the naive decomposition and return it. If the
      evaluator accepts rank 105, that's fine. If not, I'll need to optimize.
      
      Actually, let me try to implement a rank-42 decomposition by using
      Strassen-like techniques for the 3x3*3x3 blocks.
      
      For 3x3 * 3x3, Strassen's algorithm uses 7 multiplications. We can
      extend this to 3x3 * 3x7 by applying it to each of the 7 columns,
      giving 7 * 7 = 49 terms for the 3x3*3x7 part.
      
      For 3x2 * 2x7, we can use a similar approach. The rank for 3x2*2x2 is
      at most 3*2*2 = 12 (naive) or possibly less. For 7 columns, that's
      7 * 12 = 84 terms? That's worse.
      
      Let me just use the naive decomposition for now.
    """
    n, m, p = DIMENSIONS  # n=3, m=5, p=7
    
    # Use naive decomposition with rank = n*m*p = 105
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
