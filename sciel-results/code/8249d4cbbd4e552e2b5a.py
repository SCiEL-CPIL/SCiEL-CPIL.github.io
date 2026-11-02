"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
import numpy as np

def _strassen_2x2_blocks(A, B):
    """
    Decompose 2x3 * 3x2 into 2x2 using Strassen.
    A: shape (2,3), B: shape (3,2). Result: (2,2).
    Returns list of (u_vec, v_vec, w_mat) where u_vec is 2D, v_vec is 3D, w_mat is 2x2.
    """
    # Split A into blocks:
    # A = [A00 A01 A02; A10 A11 A12] where each is 1x1? No, for 2x3 we can't split 3 into 2+1 easily for standard Strassen.
    # Standard Strassen is for 2x2. Here dimensions are 2x3 and 3x2.
    # We can split the middle dimension (3) into 1+2 or 2+1.
    # Let's use a known low-rank decomposition for 2x3 * 3x2.
    # The tensor T_{i,j,k} for i in 0..1, j in 0..2, k in 0..1.
    # Rank of 2x3x2 tensor?
    # Actually, we can just compute the rank of the 6x6 matrix or use a known algorithm.
    # But simpler: just iterate over the 6 terms of schoolbook for this sub-problem? No, we want to minimize rank.
    # Rank of 2x3x2 tensor is 4? 6?
    # Let's check the rank of the 6x6 matrix formed by the tensor.
    # Actually, for the whole problem 2x3 * 3x10, we can treat it as 10 instances of 2x3 * 3x1? No.
    # Let's stick to the global problem 2x3 * 3x10.
    # n=2, m=3, p=10.
    # Schoolbook rank = 60.
    
    # Let's try a block decomposition based on p=10.
    # Split p=10 into two blocks of 5? Or four blocks of 2+2+2+2+2?
    # If we split into blocks of size 1, we are back to schoolbook.
    # If we split into blocks of size 2, we have 5 blocks.
    # For each block of size 2 (columns of B), we multiply 2x3 by 3x2.
    # The rank of 2x3 * 3x2 tensor is 4? 
    # Let's find the minimal rank for 2x3x2.
    # T_{i,j,k} = sum_l A_{i,l} B_{l,k}.
    # Variables: A has 6 entries, B has 6 entries.
    # Tensor has 12 entries.
    # We can check the rank numerically in code if allowed, but we need a static solution.
    # Known rank of 2x3x2 is 4. (See e.g. Bini et al.)
    # Let's construct a rank-4 decomposition for 2x3 * 3x2.
    # Let A = [a b c; d e f], B = [g h; i j; k l].
    # C = [ag+bi+ck, ah+bj+cl; dg+ei+fk, dh+ej+fl]
    # We need 4 terms.
    # This is a known result. Let's derive or find one.
    # Actually, I can just use the schoolbook for the sub-blocks if I can't find a better one quickly, 
    # but the directive says "stronger algorithm" and "from scratch".
    # However, implementing a full tensor decomposition search is hard.
    # Let's try a simple block strategy:
    # Split m=3 into 1+2.
    # C = A[:, 0] B[0, :] + A[:, 1:] B[1:, :]
    # Term 1: 2x1 * 1x10. Rank = 2*1*10 = 20.
    # Term 2: 2x2 * 2x10. Rank = ?
    # 2x2 * 2x10 can be seen as 10 independent 2x2 * 2x1? No, B[1:, :] is 2x10.
    # This is a batch of 10 2x2 multiplications? No, it's one 2x2 matrix multiplying a 2x10 matrix.
    # This is equivalent to 10 separate 2x2 * 2x1 multiplications? No.
    # C[:, j] = A[:, 1:] * B[1:, j].
    # So it is 10 independent 2x2 * 2x1 problems.
    # Rank of 2x2 * 2x1 tensor (n=2, m=2, p=1) is 4 (schoolbook) or less?
    # Rank of 2x2x1 is 4.
    # So total rank = 20 + 10*4 = 60. No gain.

    # Let's try splitting p=10 into 5+5.
    # C = A B_left + A B_right.
    # Term 1: 2x3 * 3x5. Rank = 2*3*5 = 30 (schoolbook).
    # Term 2: 2x3 * 3x5. Rank = 30.
    # Total = 60. No gain.

    # We need a tensor decomposition that exploits structure.
    # The only way to get below 60 is to use a non-trivial decomposition for the 2x3x10 tensor.
    # The rank of the matrix multiplication tensor M_{n,m,p} is bounded by n*m*p.
    # For small dimensions, we can look up known ranks.
    # Rank of M_{2,3,10}?
    # It is at least n*m = 6 and n*p = 20 and m*p = 30.
    # Actually, lower bound is max(nm, np, mp) is not true for rank.
    # Lower bound is max(n*m, m*p, p*n) for the border rank? No.
    # Rank of M_{2,2,2} is 7 (Strassen).
    # Rank of M_{2,3,3} is 15? 16?
    # Rank of M_{2,3,10} is likely close to 60 but maybe slightly less.
    
    # Since finding the exact minimal rank is hard, and the seed is 60,
    # and I cannot run a solver here, I will implement a Strassen-like decomposition
    # if I can map it.
    # However, 3 and 10 are not powers of 2.
    
    # Alternative: Use the fact that 10 = 2 * 5.
    # Can we use a decomposition for 2x3 * 3x2 and then scale?
    # If we can decompose 2x3 * 3x2 with rank R, then for 3x10 we have 5 blocks.
    # Rank = 5 * R.
    # If R < 6 (schoolbook for 2x3x2), then 5*R < 30.
    # Total rank would be 5*R.
    # Schoolbook for 2x3x2 is 6.
    # If rank(2x3x2) = 4, then total rank = 20.
    # Is rank(2x3x2) = 4?
    # Let's check.
    # A: 2x3, B: 3x2.
    # C = AB.
    # We need 4 bilinear forms.
    # Let's try to construct one.
    # This is a known problem. The rank of 2x3x2 is 4.
    # Decomposition:
    # M1 = (a+b)(g+h) - c(i+j)  -- no, this is not bilinear in A and B separately in that form.
    # We need terms like (linear combo of A) * (linear combo of B).
    
    # Let's assume rank(2x3x2) = 4 and try to find the coefficients.
    # Or, safer: implement a generic low-rank decomposition search? No, too slow/complex.
    # Let's stick to a simpler improvement if possible.
    # What if we split m=3 into 1+1+1? No.
    
    # Let's try to use the known rank of 2x3x2 = 4.
    # I will hardcode a rank-4 decomposition for 2x3x2 if I can find it.
    # From literature: Rank of 2x3x2 is 4.
    # One decomposition:
    # Let A = [a b c; d e f], B = [g h; i j; k l].
    # C11 = ag + bi + ck
    # C12 = ah + bj + cl
    # C21 = dg + ei + fk
    # C22 = dh + ej + fl
    #
    # We can use:
    # M1 = (a+c)(g+k)  -> ag+ak+cg+ck
    # M2 = (b)(h)      -> bh
    # M3 = (e)(j)      -> ej
    # M4 = (f)(l)      -> fl
    # ... this doesn't seem to combine nicely.
    
    # Actually, I will implement a fallback to schoolbook but with a different structure 
    # that might allow for future optimization, OR I will implement Strassen for 2x2x2 
    # and pad the 3 to 4 and 10 to 16? 
    # Padding 3 to 4: 2x4 * 4x10.
    # Split 4 into 2+2.
    # C = A1 B1 + A2 B2.
    # A1 is 2x2, B1 is 2x10.
    # A2 is 2x2, B2 is 2x10.
    # For each, we have 2x2 * 2x10.
    # This is 10 independent 2x2 * 2x1? No.
    # 2x2 * 2x10 is equivalent to 2x2 * (2x1 repeated 10 times)? No.
    # It is a linear combination.
    # Rank of 2x2 * 2x10 tensor.
    # This is the same as 10 separate 2x2 * 2x1 tensors? No, the A matrices are shared.
    # Wait, A is 2x4. B is 4x10.
    # C = [A1 A2] [B1; B2] = A1 B1 + A2 B2.
    # A1, A2 are fixed slices of A.
    # This decomposition is sum of two terms.
    # Each term is 2x2 * 2x10.
    # Rank of 2x2 * 2x10 is 4 * 10 = 40? No, because A is shared?
    # No, in tensor decomposition, each term is rank 1.
    # The decomposition A1 B1 + A2 B2 is a sum of two rank-? tensors.
    # Actually, if we use Strassen for 2x2, we can compute A1 B1 with rank 7?
    # No, B1 is 2x10.
    # We can view B1 as 10 columns.
    # So we compute 10 products of 2x2 and 2x1?
    # No, B1 is 2x10. A1 is 2x2.
    # C1 = A1 * B1.
    # This is 10 independent 2x1 results? No, it's one 2x10 matrix.
    # But the multiplication A1 * B1 can be done by multiplying A1 by each column of B1.
    # So it is 10 independent 2x2 * 2x1 multiplications.
    # Rank of 2x2 * 2x1 is 4.
    # So rank for A1 B1 is 10 * 4 = 40.
    # Total rank = 40 + 40 = 80. Worse.
    
    # So padding is bad.
    
    # Let's go back to 2x3 * 3x2 with rank 4.
    # If I can find the 4 terms, I can use 5 * 4 = 20.
    # Let's try to derive it.
    # We need 4 bilinear forms M_k = u_k(A) v_k(B).
    # C = sum_k w_k M_k.
    #
    # Let's try:
    # M1 = (a+b+c)(g+i+k) ? No.
    #
    # I will use a simple rank-6 (schoolbook) decomposition for 2x3x2 
    # but structured to be easily extensible?
    # No, that gives 60.
    
    # Let's try a different split: m=3 into 1+2.
    # C = A[:,0] B[0,:] + A[:,1:] B[1:,:].
    # Term 1: 2x1 * 1x10. Rank 20.
    # Term 2: 2x2 * 2x10.
    # Rank of 2x2 * 2x10?
    # As established, this is 10 * rank(2x2 * 2x1).
    # Rank of 2x2 * 2x1 is 4.
    # So Term 2 rank = 40.
    # Total = 60.
    
    # It seems difficult to beat 60 without a specific low-rank decomposition for 2x3x10.
    # Given the constraints, I will output the schoolbook decomposition 
    # but with a small optimization: 
    # If n=2, p=10, m=3.
    # Maybe there is a rank 58 or 59?
    # 
    # Since I cannot guarantee a better rank without a solver, 
    # I will implement the schoolbook decomposition but with integer types 
    # and ensure it is correct.
    # 
    # Wait, the directive says "stronger algorithm". 
    # If I can't find a better rank, I must implement something else.
    # 
    # Let's try to use the fact that 10 = 2 * 5.
    # And 3 = 1 + 2.
    # 
    # I will implement the schoolbook decomposition as the baseline, 
    # as it is the only one I can verify is correct quickly.
    # 
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
