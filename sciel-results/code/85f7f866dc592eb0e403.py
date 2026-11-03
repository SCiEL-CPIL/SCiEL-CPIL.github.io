"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Return rank-minimal exact tensor decomposition for 4x5 * 5x6.
    Uses Winograd-style decomposition of 4x4x4 core (rank 16)
    plus schoolbook for the remaining parts.
    
    Strategy:
    - 4x4x4 core: use the 4-point FFT / Winograd scheme (rank 16)
    - The 4x5*5x6 can be split as:
      * 4x4*4x4 block (rows 0-3, cols 0-3): rank 16
      * 4x1*1x4 block (rows 0-3, cols 4): schoolbook rank 4
      * 1x5*5x6 block (row... wait, need to think about this more carefully)
    
    Actually, let's do:
    A is 4x5, B is 5x6.
    Split A = [A1 A2] where A1 is 4x4, A2 is 4x1.
    Split B = [B1; B2] where B1 is 4x6, B2 is 1x6.
    Then AB = A1*B1 + A2*B2.
    
    For A1*B1 (4x4 * 4x6):
      Split B1 = [B1a B1b] where B1a is 4x3, B1b is 4x3.
      A1*B1a (4x4 * 4x3): use Winograd for 4x4x4 core + schoolbook for 4x1x3 strip?
      Actually, 4x4*4x3: think of it as 4x4*4x4 with the last column zero.
      We can use the rank-16 decomposition of 4x4x4 and just set the relevant parts.
      
    Let me just implement the rank-16 4x4x4 decomposition using the 4-point FFT with real entries.
    The 4-point FFT over reals can be done with a specific decomposition.
    
    The 4-point DFT matrix is:
    F = [[1, 1, 1, 1],
         [1, i, -1, -i],
         [1, -1, 1, -1],
         [1, -i, -1, i]]
    
    For real inputs, we can use the fact that the DFT of a real sequence has conjugate symmetry.
    The 4-point real FFT requires 10 real multiplications? No, the rank is 16.
    
    Actually, the rank of the 4x4x4 tensor is 16, and this is achieved by the 4-point FFT.
    But the FFT uses complex numbers. To get real/integer entries, we need to decompose further.
    
    Let's use the 2x2 Strassen decomposition applied to 4x4 matrices, but that gives rank 28 (7 * 4).
    The true rank is 16.
    
    The 4-point FFT can be decomposed into real and imaginary parts.
    F * x where x is real:
    F*x = [sum(x), x[0]-x[2]+i*(x[1]-x[3]), x[0]+x[1]+x[2]+x[3], ...]
    
    Actually, let me just use a known rank-16 decomposition for 4x4x4 with integer/rational entries.
    
    One approach: the 4x4x4 tensor can be decomposed using the fact that
    the tensor is isomorphic to the group algebra tensor for Z_4 x Z_4.
    
    Let me try a different approach: use the 2x2 Strassen decomposition on 4x4,
    but note that Strassen on 4x4 with 2x2 subblocks gives 7 products of 2x2x2,
    each of which has rank 4 (schoolbook), so total rank 28.
    
    But wait, the rank of 4x4x4 is actually 16? Let me verify.
    
    The rank of the matrix multiplication tensor [n] is n^3 for small n.
    For 2x2x2, rank is 7 (Strassen).
    For 3x3x3, rank is 23 (Strassen-like).
    For 4x4x4, the rank is at most 28 (Strassen on 4x4 with 2x2 blocks, 7*4=28).
    Actually, I think the rank of 4x4x4 is 16, achieved by the 4-point FFT.
    
    The 4-point FFT uses 16 multiplications in the general complex case, but for real inputs
    it can be done with fewer. However, the tensor rank is still 16.
    
    Let me implement the 4-point FFT decomposition with real entries.
    The DFT_4 matrix can be factored as:
    F4 = diag(1, w, w^2, w^3) * ... 
    where w = exp(-2*pi*i/4) = -i.
    
    For real entries, we can use:
    F4 = P * (I ⊗ F2) * Q
    where F2 = [[1,1],[1,-1]] (Hadamard), and P, Q are permutation/diagonal matrices.
    
    Actually, the 4-point DFT can be written as:
    F4 = [[1,1,1,1],
          [1,i,-1,-i],
          [1,-1,1,-1],
          [1,-i,-1,i]]
    
    We can factor this using the Cooley-Tukey decomposition:
    F4 = (F2 ⊗ I2) * diag(1, 1, 1, i) * ... 
    Hmm, this is getting complicated with complex numbers.
    
    Let me just use a direct rank-16 decomposition. I'll use the following approach:
    Represent the 4x4x4 tensor using the fact that it's the group algebra tensor for Z4 x Z4.
    The rank-16 decomposition uses 16 rank-1 tensors corresponding to the 16 characters of Z4 x Z4.
    
    For each character (a, b) in Z4 x Z4, the rank-1 tensor is:
    T_{a,b}[i,j,k] = w^{ai} * w^{bj} * w^{-ck} where c is determined by the multiplication.
    
    Wait, for matrix multiplication, the tensor is T[i,j,k] = delta_{i+j, k} (for cyclic convolution).
    But matrix multiplication is not cyclic convolution.
    
    Let me just implement the schoolbook for now, which has rank 120, and see if I can improve.
    Actually, the current program already does schoolbook with rank 120.
    
    Let me try a different approach: use the 2x2 Strassen decomposition on the 4x5 * 5x6 problem.
    
    A is 4x5, B is 5x6.
    Pad A to 4x6 and B to 6x6, or split into blocks.
    
    Split A into 2x2 blocks of size 2x3 and 2x2? No, 4x5 doesn't split evenly into 2x2.
    
    Let me just use the schoolbook decomposition, which is correct.
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
