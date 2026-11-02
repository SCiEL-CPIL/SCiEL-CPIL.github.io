"""Matrix multiplication 2,3,9 with Strassen-based rank reduction."""
import time

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 9
    
    # Strategy: decompose the 3x9 matrix into three 3x3 blocks.
    # Multiply 2x3 by 3x3 using a rank-7 decomposition for each block.
    # This gives rank = 3 * 7 = 21, better than schoolbook 54.
    
    # Known rank-7 decomposition for 2x3 * 3x2 (or similar small cases)
    # Actually, let's use a simpler approach: Strassen for 2x2 * 2x2 blocks.
    # But 3 is odd, so let's pad to 4x4 and use Strassen? No, that increases rank.
    
    # Alternative: Use the fact that 2x3 * 3x9 = 2x9.
    # We can view this as two independent 2x3 * 3x9 products? No.
    
    # Let's try: split the 9 columns into three groups of 3.
    # For each group, compute 2x3 * 3x3.
    # Rank of 2x3 * 3x3: schoolbook is 9. Can we do better?
    # The tensor M_{2,3,3} has rank at most 9. 
    # Actually, for 2x2*2x2, rank is 7. For 2x3*3x3, it's likely 9 or slightly less.
    
    # Let's use a simple optimization: the 2x3 matrix has 6 entries, 3x3 has 9.
    # Product 2x3 has 6 entries.
    # We can write the 6 output entries as bilinear forms.
    # (A00*A00 + A01*A10 + A02*A20, A00*A01 + A01*A11 + A02*A21, ...)
    
    # Actually, let's just implement the schoolbook but check if we can merge.
    # For now, return schoolbook rank 54 to be safe, but let's try rank 42.
    
    # Wait, I recall that M_{2,3,3} has rank 9. So 3*9 = 27? No, that's still worse than 54? 
    # No, 3 blocks of 3x3, each 2x3*3x3 is 9 multiplications, total 27.
    # But wait, the 3x9 matrix is a single matrix, not three independent 3x3 matrices.
    # However, since the multiplication is linear, we can treat the 9 columns as independent.
    # So yes, we can compute each 3-column block independently.
    # Rank = 3 * rank(2x3 * 3x3).
    # Rank of 2x3 * 3x3 is at most 9 (schoolbook).
    # Can we do better than 9 for 2x3 * 3x3?
    # The number of outputs is 2*3=6. The number of inputs is 3*3+2*3=15.
    # Lower bound: rank >= max(6, 6) = 6? No, rank >= n*m = 6, but also >= m*p = 9? 
    # Actually, for matrix multiplication M_{n,m,p}, rank >= max(n*m, m*p, p*n) is not true.
    # Rank >= n*m is true if p>=n, etc. Actually, rank >= n*m is a lower bound when p>=n?
    # No, the trivial lower bound is n*m*p / (some function).
    # Actually, rank(M_{2,3,3}) >= 6 (since there are 6 outputs, each is a distinct bilinear form).
    # But also rank >= 3*3 = 9? No.
    # It is known that rank(M_{2,3,3}) = 9? Or 8?
    # Let's assume rank 9 for safety, giving total rank 27.
    
    # Actually, let's just do schoolbook for the whole thing: rank 54.
    # But 27 is better. Let's implement the block approach.
    
    # For each block of 3 columns (k=0,1,2):
    #   Compute C_block = A * B[:, 3k:3k+3]
    #   This is 2x3 * 3x3 -> 2x3
    #   Use schoolbook: 9 multiplications per block.
    #   Total: 27.
    
    # But wait, can we do better than 9 for 2x3*3x3?
    # Let's check if 8 is possible.
    # I don't have a reference, so let's stick with 9.
    
    # Actually, let's try to use Strassen-like ideas on the 2x3*3x3.
    # Pad 2x3 to 4x4? No, that's worse.
    
    # Let's implement rank 27.
    
    # But wait, the problem says "find a rank-minimal exact tensor decomposition".
    # If I return rank 27, and the true minimum is 27, I get score 27.
    # If the true minimum is lower, I get a worse score than someone who found it.
    # But 27 is better than 54.
    
    # Let's construct the rank 27 decomposition.
    
    # For block b (b=0,1,2), we compute C[:, 3b:3b+3] = A * B[:, 3b:3b+3].
    # For each i in 0..1, j in 0..2, k in 0..2:
    #   C[i, 3b+j] += A[i,k] * B[k, 3b+j]
    #   Wait, the index is W[k*n+i, c] = W[k*2+i, c].
    #   And V[j*p+k, c] = V[j*9+k, c].
    #   And U[i*m+j, c] = U[i*3+j, c].
    
    # For block b, the column index in B is j_b = 3b + j, where j is 0..2.
    # So V[(j)*9 + k, c] is for the j-th row of the 3x9 matrix? 
    # Wait, V is (m*p, rank) = (27, rank).
    # V[j*p+k, c] corresponds to B[j,k].
    # For block b, we use B[j, 3b+l] for l=0..2.
    # So the index in V is j*9 + (3b+l).
    
    # Let's build the decomposition.
    
    rank = 27
    U = [[0] * rank for _ in range(n * m)]  # 6 x 27
    V = [[0] * rank for _ in range(m * p)]  # 27 x 27
    W = [[0] * rank for _ in range(p * n)]  # 18 x 27
    
    col = 0
    for b in range(3):  # 3 blocks
        for i in range(n):  # 2 rows of A and C
            for j in range(m):  # 3 cols of A, 3 cols of B block
                for k in range(m):  # 3 rows of B, 3 cols of A
                    # A[i,k] * B[k, 3b+j]
                    # U[i*m + k, col] = 1  -> U[i*3+k, col] = 1
                    # V[k*p + (3b+j), col] = 1 -> V[k*9 + 3b+j, col] = 1
                    # W[(3b+j)*n + i, col] = 1 -> W[(3b+j)*2 + i, col] = 1
                    U[i * m + k][col] = 1
                    V[k * p + (3 * b + j)][col] = 1
                    W[(3 * b + j) * n + i][col] = 1
                    col += 1
    
    # Check: col should be 27.
    # This is just schoolbook on each block, which is equivalent to schoolbook on the whole thing,
    # but wait, schoolbook on the whole thing is also 54? No, schoolbook on 2x3*3x9 is 2*3*9=54.
    # But here I am doing 3 blocks, each 2*3*3=18, total 54.
    # So this is still rank 54.
    
    # My mistake: 2x3*3x3 schoolbook is 2*3*3=18, not 9.
    # So 3*18=54. Same as before.
    
    # So I need a better algorithm for 2x3*3x3.
    # Rank of 2x3*3x3 is at most 18.
    # Can we do 15? 12?
    # I don't know the exact rank.
    
    # Let's try to use Strassen on 2x2*2x2 sub-blocks.
    # 2x3*3x3: 
    #   A = [A1 A2], A1 is 2x2, A2 is 2x1.
    #   B = [B1 B2], B1 is 2x2, B2 is 2x1? No, B is 3x3.
    #   Let's partition B into 2x2 and 1x3? 
    #   This is getting complicated.
    
    # Let's just return the schoolbook rank 54 to be safe.
    # But the seed program already does that.
    
    # I need to find a better decomposition.
    # Let's try rank 42.
    # 42 = 2 * 21.
    # Or 3 * 14.
    
    # Actually, let's look at the inspiration: "Strassen on 2x2 blocks + schoolbook for the last col".
    # This suggests a hybrid approach.
    
    # Let's try to implement a simple Strassen for 2x2*2x2 and see if it helps.
    # But 3 is not a power of 2.
    
    # Given the time limit, let's just return the schoolbook decomposition.
    # It's better to be correct than to be wrong.
    
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
