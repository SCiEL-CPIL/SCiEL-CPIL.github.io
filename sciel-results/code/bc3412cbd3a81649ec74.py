import numpy as np
import time

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # Part 1: 4x4 * 4x4 (first 4 columns of output) using Strassen rank 49
    # Part 2: 4x4 * 4x3 (last 3 columns of output) using Schoolbook rank 48
    
    rank_total = 49 + 48
    U = [[0] * rank_total for _ in range(n * m)]
    V = [[0] * rank_total for _ in range(m * p)]
    W = [[0] * rank_total for _ in range(p * n)]
    
    # --- Part 1: Strassen 4x4x4x4 ---
    # Recursive Strassen for 2x2x2x2 (rank 7) applied to 4x4x4x4
    # We generate the U, V, W for the 4x4x4x4 case
    
    # Strassen 2x2x2x2 terms (indices 0..6)
    # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b12-b22)
    # M4 = a22(b21-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For 4x4x4x4, we apply this to the 2x2 blocks.
    # This creates a tensor of rank 7*7=49.
    
    # We will explicitly build the 49 terms.
    # This is tedious to hardcode. Instead, we can use a helper to generate the 2x2 Strassen
    # and then combine them.
    
    # Helper: Generate U, V, W for 2x2x2x2 Strassen
    def strassen_2x2():
        u = [[0]*7 for _ in range(4)]
        v = [[0]*7 for _ in range(4)]
        w = [[0]*7 for _ in range(4)]
        
        # M1
        u[0][0] = 1; u[3][0] = 1
        v[0][0] = 1; v[3][0] = 1
        w[0][0] = 1; w[1][0] = -1; w[2][0] = -1; w[3][0] = 1 # C11, C12, C21, C22 coeffs for M1?
        # C11 = M1..., C12 = ... - M5, C21 = ..., C22 = M1...
        # W[k*n+i, c] where n=2, m=2, p=2.
        # Index in W: k*n + i.
        # C11 is i=0, k=0 -> idx 0.
        # C12 is i=0, k=1 -> idx 1.
        # C21 is i=1, k=0 -> idx 2.
        # C22 is i=1, k=1 -> idx 3.
        
        # M1: C11+=1, C22+=1. C12, C21 unaffected?
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # M1
        w[0][0] = 1; w[3][0] = 1
        # M2
        u[2][1] = 1; u[3][1] = 1
        v[0][1] = 1
        w[2][1] = 1; w[3][1] = -1
        # M3
        u[0][2] = 1
        v[1][2] = 1; v[3][2] = -1
        w[1][2] = 1; w[3][2] = 1
        # M4
        u[3][3] = 1
        v[2][3] = 1; v[0][3] = -1
        w[0][3] = 1; w[2][3] = 1
        # M5
        u[0][4] = 1; u[1][4] = 1
        v[3][4] = 1
        w[0][4] = -1; w[1][4] = 1
        # M6
        u[2][5] = 1; u[0][5] = -1
        v[0][5] = 1; v[1][5] = 1
        w[3][5] = 1
        # M7
        u[1][6] = 1; u[3][6] = -1
        v[2][6] = 1; v[3][6] = 1
        w[0][6] = 1
        
        return u, v, w

    u2, v2, w2 = strassen_2x2()
    
    # Now build 4x4x4x4 from 2x2x2x2
    # A is 4x4, split into 2x2 blocks A11, A12, A21, A22.
    # B is 4x4, split into B11, B12, B21, B22.
    # C is 4x4, split into C11, C12, C21, C22.
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    
    # We need to map the 2x2 Strassen terms to the 4x4 indices.
    # This is a standard Kronecker product of the 2x2 Strassen with itself?
    # Yes, the rank is 7*7=49.
    
    u4 = [[0]*49 for _ in range(16)]
    v4 = [[0]*49 for _ in range(16)]
    w4 = [[0]*49 for _ in range(16)]
    
    for i in range(7):
        for j in range(7):
            col = i*7 + j
            # A indices for term i in block (bi, bj)
            # A block (bi, bj) corresponds to rows 2*bi, 2*bi+1 and cols 2*bj, 2*bj+1
            # In U (16x49), row index is a_flat = a_row*4 + a_col
            for a_row in range(2):
                for a_col in range(2):
                    # Local index in 2x2 A
                    la_row = a_row
                    la_col = a_col
                    val = u2[la_row*2 + la_col][i]
                    if val != 0:
                        # Global row in A
                        g_a_row = 2*bi + a_row
                        g_a_col = 2*bj + a_col
                        # Wait, bi, bj are the block indices of A.
                        # We are iterating i (term in A block) and j (term in B block)?
                        # No, the 4x4 Strassen is not simply Kronecker of 2x2 Strassen for A and B separately.
                        # It is the Kronecker product of the 2x2 Strassen algorithm applied to the blocks.
                        # The 49 terms are products of the 7 terms of A-blocks and 7 terms of B-blocks.
                        # This is correct.
                        
                        # But we need to know which block of A and B the term i and j correspond to.
                        # In the Kronecker product, term (i,j) involves A-block (bi, bj) and B-block (bi, bj)?
                        # No, C11 = A11*B11 + A12*B21.
                        # This requires mixing blocks.
                        
                        # Actually, the rank-49 decomposition is obtained by applying Strassen's algorithm to the 2x2 blocks of the 4x4 matrix.
                        # The 7 intermediate matrices M_k are 2x2 matrices.
                        # Then we apply Strassen again? No.
                        # The standard way to get rank 49 for 4x4x4x4 is to use the rank-7 algorithm on the 2x2 blocks.
                        # The 7 M's are 2x2 matrices.
                        # Then we compute the 4x4 C from the 2x2 M's.
                        # This is not a direct sum of 49 rank-1 terms in the scalar sense unless we expand the 2x2 M's.
                        
                        # Let's just use the schoolbook for the 4x4x4x4 part? Rank 64.
                        # 64 + 48 = 112. No improvement.
                        
                        # We need a better 4x4x4x4 decomposition.
                        # Let's use the rank-49 decomposition from the literature.
                        # It is known that rank(4x4x4x4) = 49.
                        # We can construct it by applying the rank-7 Strassen to the 2x2 blocks.
                        # The 7 intermediate 2x2 matrices are linear combinations of the 16 A and B entries.
                        # Then the 16 C entries are linear combinations of the 28 M entries.
                        # This is a bilinear map.
                        # The rank is the number of scalar multiplications.
                        # Each of the 7 M's is a 2x2 matrix, so 7*4=28 scalar multiplications?
                        # No, Strassen uses 7 multiplications of 2x2 matrices.
                        # Each 2x2 matrix multiplication has rank 4 (schoolbook) or 7 (Strassen).
                        # If we use schoolbook for the 2x2 matrix multiplications inside Strassen, the total rank is 7*4 = 28?
                        # No, the 7 M's are formed by additions. The multiplications are M_k = (linear comb A) * (linear comb B).
                        # Each M_k is a 2x2 matrix.
                        # So we need to compute 7 products of 2x2 matrices.
                        # If we use schoolbook for each, rank is 7*4 = 28.
                        # But the lower bound for 4x4x4x4 is 28? No, it's 49?
                        # Actually, the rank of 4x4x4x4 is 49.
                        # This implies we cannot just use schoolbook for the inner 2x2s.
                        # We must use Strassen for the inner 2x2s as well?
                        # If we use Strassen for the inner 2x2s, the rank is 7*7 = 49.
                        
                        # So, we apply Strassen (rank 7) to compute each of the 7 M_k (which are 2x2 matrices).
                        # This requires 7*7 = 49 scalar multiplications.
                        
                        # So, we need to generate the 49 terms.
                        # Term (i, j) where i is the Strassen term for A-blocks, j is the Strassen term for B-blocks?
                        # No, the structure is:
                        # 1. Compute 7 intermediate 2x2 matrices M_k using Strassen on the 2x2 blocks of A and B.
                        #    This uses 7 multiplications of 2x2 matrices.
                        # 2. Each 2x2 matrix multiplication uses 7 scalar multiplications (Strassen).
                        #    So total 49 scalar multiplications.
                        
                        # We will implement this.
                        
                        pass # Placeholder

    # This is getting too complex to generate correctly in the fly.
    # Let's fall back to a simpler improvement: Rank 56 for 4x4x4x4? No.
    # Let's use the rank-49 decomposition if we can find a pre-computed table.
    # Since I cannot look it up, I will use the rank-64 schoolbook for 4x4x4x4 and rank-48 for 4x4x4x3.
    # Total 112. No improvement.
    
    # Alternative: Use the rank-10 decomposition for 2x2x2x3.
    # 4x4x4x7 can be split into 4 independent 2x2x2x3 multiplications?
    # If we ignore cross terms, yes. But we can't.
    
    # Let's try to use the rank-7 Strassen for 2x2x2x2 and apply it to the 4x4x4x7 problem by padding B to 4x8.
    # Rank of 4x4x4x8 is 49*2 = 98?
    # Then we project out the last 4 columns.
    # This gives rank 98.
    # 98 < 112.
    
    # Let's implement the rank-98 solution.
    # Pad B to 4x8. Use Strassen for 4x4x4x8.
    # Strassen for 4x4x4x8:
    # Split 4x8 into two 4x4.
    # Use rank-49 for each? No, Strassen is for square.
    # We can use the rank-49 decomposition for 4x4x4x4 and apply it to the two 4x4 blocks of B.
    # Total rank 49 + 49 = 98.
    
    # This is a valid improvement.
    
    # We will generate the U, V, W for the 4x4x4x4 Strassen (rank 49) and use it twice.
    # Since I don't have the 49 terms hardcoded, I will use the schoolbook for 4x4x4x4 (rank 64) and 4x4x4x3 (rank 48).
    # Total 112.
    
    # I will simply return the schoolbook decomposition but with a smaller rank by using the rank-10 2x2x2x3 algorithm for the 4x4x4x3 part?
    # 4x4x4x3: Split A into 2x2 blocks, B into 2x3 blocks.
    # This is not straightforward.
    
    # I will return the schoolbook decomposition. It is safe.
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
