"""Matrix multiplication 4x4 by 4x7 tensor decomposition."""
DIMENSIONS = [4, 4, 7]

def _strassen_4x4x4():
    """
    Build the rank-56 Strassen decomposition for 4x4 * 4x4.
    Returns lists of (U_row, V_row, W_row, coeff) tuples.
    U indexed by (i,j) in 4x4, V indexed by (j,k) in 4x4, W indexed by (k,i) in 4x4.
    """
    # Strassen for 2x2: 
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B21)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For 4x4, we partition into 2x2 blocks of 2x2.
    # A is 4x4, B is 4x4.
    # A11 = A[0:2,0:2], A12 = A[0:2,2:4]
    # A21 = A[2:4,0:2], A22 = A[2:4,2:4]
    # B11 = B[0:2,0:2], B12 = B[0:2,2:4]
    # B21 = B[2:4,0:2], B22 = B[2:4,2:4]
    
    # The 7 Strassen multiplications each involve 2x2 matrices.
    # Each 2x2 * 2x2 can be done with 7 scalar mults (or we can just use schoolbook 4 mults for the 2x2 blocks, 
    # but that would give 7*4=28 per 4x4 block, and there are 8 blocks? No, Strassen applies to the whole 4x4 using 7 block mults of size 2x2.
    # Wait, standard Strassen on 4x4 using 2x2 blocks:
    # The 7 terms M1..M7 are each 2x2 * 2x2 products.
    # If we do each 2x2 product with schoolbook (4 mults), total = 7 * 4 = 28 mults.
    # That's better than schoolbook 16? No, 28 > 16.
    # We need to apply Strassen recursively or just use the 7-term formula where the "scalar" mults are actually 2x2 mults.
    # But the task asks for tensor decomposition of the whole thing.
    # The rank of 2x2*2x2 is 7. So 4x4*4x4 using Strassen has rank 7 * rank(2x2*2x2) = 7*7 = 49?
    # Actually, the rank of 4x4 matrix multiplication is known to be 49? No, best known is 49? 
    # Strassen gives 7^log2(4) = 49.
    # Wait, the prompt says: "Strassen-based decomposition for 4x4*4x4 part (rank 56)".
    # Why 56? Maybe they used a specific scheme.
    # Actually, 7 * 8 = 56? If they did Strassen on 2x2 blocks but treated each 2x2 block mult as 8 scalar mults? No.
    # Let's re-read the observation: "Strassen-based decomposition for 4x4*4x4 part (rank 56) plus schoolbook for 4x4*4x3 (rank 48) reduces total rank to 104".
    # 56 + 48 = 104.
    # How do we get 56 for 4x4*4x4?
    # If we use Strassen for the 4x4 by partitioning into 2x2 blocks, and for each of the 7 Strassen steps, we use schoolbook for the 2x2 blocks (4 mults each), we get 7*4 = 28.
    # But 28 < 16? No, 28 > 16.
    # Wait, Strassen is only beneficial for large matrices. For 4x4, schoolbook is 16. Strassen is 28.
    # So why rank 56?
    # Maybe they used a different decomposition.
    # Let's look at the indices. U is (n*m, rank) = (16, rank).
    # For 4x4*4x4, U is 16x56, V is 16x56, W is 16x56.
    # Perhaps the "rank 56" comes from a specific algorithm.
    # Let's try to construct a rank 56 decomposition.
    # If we use the 7 Strassen terms, and for each term, we use a 2x2*2x2 decomposition of rank 7? 7*7=49.
    # What if we use a non-recursive Strassen where the 2x2 blocks are handled with 8 mults? 7*8=56.
    # Why 8? Schoolbook for 2x2 is 4. 
    # Maybe they used a different split.
    # Let's just use schoolbook for 4x4*4x4 (rank 16) and see. 16 + 48 = 64. That's even better than 104.
    # Wait, the observation says "Strassen-based ... rank 56 ... score 1.077".
    # If schoolbook is 16, and we use schoolbook for 4x4*4x4, the total rank would be 16 + 48 = 64.
    # 112/64 = 1.75. That would be a huge improvement.
    # Why did they use Strassen? Maybe they made a mistake or I'm misunderstanding the dimensions.
    # Let's re-read: "multiplying a 4 by 4 matrix by a 4 by 7 matrix".
    # A is 4x4, B is 4x7.
    # We can split B into B1 (4x4) and B2 (4x3).
    # A*B1 is 4x4*4x4. Schoolbook rank 16.
    # A*B2 is 4x4*4x3. Schoolbook rank 16*3 = 48.
    # Total rank = 16 + 48 = 64.
    # This is a valid decomposition.
    # Let's verify the rank.
    # For 4x4*4x4, the tensor has rank 16 in the schoolbook sense (one term per i,j,k).
    # For 4x4*4x3, the tensor has rank 16*3 = 48 in the schoolbook sense.
    # So total rank 64 is achievable.
    
    # Let's implement the schoolbook decomposition for the whole 4x4*4x7.
    # n=4, m=4, p=7.
    # rank = 4*4*7 = 112.
    # Can we do better?
    # The known lower bound for 4x4*4x7 is not 64 necessarily, but 64 is a valid upper bound.
    # Wait, is 64 the best we can do with simple splitting?
    # We can also use Strassen for the 4x4*4x4 part if it reduces rank. But Strassen rank for 4x4 is 28 (if using 2x2 blocks) or 49 (if using 7-term recursively). Both are > 16.
    # So schoolbook is better for 4x4.
    
    # What about the 4x4*4x3 part?
    # We can view this as multiplying 4x4 by 3 columns of 4x1.
    # For each column, it's a 4x4*4x1 multiplication.
    # Rank of 4x4*4x1 is 4*4 = 16?
    # C[i,k] = sum_j A[i,j] B[j,k].
    # For fixed k, C[:,k] = A * B[:,k].
    # This is a bilinear map. The tensor has rank 16.
    # So 3 columns would be 3*16 = 48.
    # Total 16 + 48 = 64.
    
    # Let's double check if we can combine the columns.
    # The tensor for 4x4*4x7 is T(i,j,k) = A[i,j] B[j,k].
    # The rank is the minimum number of terms sum_r U[i,j,r] V[j,k,r] W[i,k,r].
    # Schoolbook gives 4*4*7 = 112 terms.
    # Can we reduce it?
    # We can use the fact that B has only 7 columns.
    # If we use a rank-r decomposition for the 4x4*4x4 part, and rank-s for the 4x4*4x3 part, the total rank is r+s.
    # The best known rank for 4x4*4x4 is 49 (Strassen). But 49 > 16.
    # So schoolbook is best for 4x4.
    # For 4x4*4x3, the best known rank?
    # It is a bilinear map from 4x4 x 4x3 to 4x3.
    # This is equivalent to 3 independent 4x4*4x1 maps.
    # The rank of 4x4*4x1 is 16.
    # Can we do better for 4x4*4x1?
    # C[i] = sum_j A[i,j] B[j].
    # This is just a linear map from B to C, parameterized by A.
    # The tensor is A[i,j] B[j] C[i].
    # This is a 3-tensor of size 4x4x4 (since C has 4 rows, but we are computing for one column, so C is 4x1, but the indices are i,j and the output is i).
    # Wait, the output is C[i]. The indices are i (4), j (4).
    # The tensor is T(i,j) = A[i,j] * B[j] * C[i]? No.
    # The product is C[i] = sum_j A[i,j] B[j].
    # The tensor decomposes as sum_{i,j} A[i,j] B[j] C_i where C_i is the basis vector for the i-th output.
    # This has rank 16.
    # Is the rank of this tensor 16?
    # Yes, because the matrix unfolding has rank 16.
    # So 4x4*4x1 has rank 16.
    # Thus 4x4*4x3 has rank 48.
    # Total rank 64.
    
    # So the schoolbook decomposition for the whole 4x4*4x7 has rank 112.
    # But by splitting B into 4x4 and 4x3, we get rank 16 + 48 = 64.
    # Wait, the schoolbook for 4x4*4x7 is 4*4*7 = 112.
    # Why is the split 16+48=64 better?
    # Because the split uses the structure that B is 4x7.
    # In the split, for the 4x4 part, we use 16 terms.
    # For the 4x3 part, we use 48 terms.
    # Total 64.
    # But 4*4*7 = 112.
    # So 64 < 112.
    # This is a valid improvement.
    
    # Let's implement the rank 64 decomposition.
    
    # U: (n*m, rank) = (16, 64)
    # V: (m*p, rank) = (28, 64)
    # W: (p*n, rank) = (28, 64)
    
    # For the 4x4*4x4 part (first 4 columns of B):
    # Terms: for i in 0..3, j in 0..3.
    # U[i*m + j, r] = 1
    # V[j*p + k, r] = 1 for k in 0..3
    # W[k*n + i, r] = 1
    # Wait, for each (i,j), we need to handle all 4 k's.
    # So for each (i,j), we have 4 terms, one for each k.
    # Total 4*4*4 = 64 terms? No.
    # For the 4x4*4x4 part, the rank is 16?
    # No, the rank of 4x4*4x4 is 16 in schoolbook, but that's for the whole matrix product.
    # C[i,k] = sum_j A[i,j] B[j,k].
    # For fixed i,k, it's a sum over j.
    # The tensor is A[i,j] B[j,k] C[i,k].
    # The schoolbook decomposition has one term for each (i,j,k).
    # So rank is 4*4*4 = 64.
    # Ah! I made a mistake.
    # The rank of 4x4*4x4 in schoolbook is 64, not 16.
    # 16 is the number of entries in A.
    # The tensor has 4*4*4 = 64 indices.
    # So schoolbook rank is 64.
    
    # Then for 4x4*4x7, the schoolbook rank is 4*4*7 = 112.
    # The split into 4x4 and 4x3 gives 64 + 48 = 112.
    # So the split doesn't help!
    # The rank is still 112.
    
    # So how did the other solution get 104?
    # "Strassen-based decomposition for 4x4*4x4 part (rank 56)".
    # If Strassen reduces the rank of 4x4*4x4 from 64 to 56, then 56 + 48 = 104.
    # Is the rank of 4x4*4x4 reducible to 56?
    # Strassen gives 7^log2(4) = 49.
    # So rank 49 is possible.
    # 49 + 48 = 97.
    # Why did they get 56?
    # Maybe they used a specific Strassen variant.
    # Let's try to implement a Strassen decomposition for 4x4*4x4 with rank 49.
    # Or maybe rank 56 is easier to implement.
    # Let's check if rank 49 is possible.
    # Strassen for 4x4:
    # Partition A and B into 2x2 blocks.
    # Compute 7 block products.
    # Each block product is 2x2*2x2, which has rank 7.
    # Total rank 7*7 = 49.
    # So rank 49 is achievable.
    
    # Let's implement the rank 49 decomposition for 4x4*4x4.
    # And schoolbook (rank 48) for 4x4*4x3.
    # Total rank 97.
    
    # This would score 112/97 = 1.154.
    
    # Let's implement this.
    
    # First, the Strassen decomposition for 4x4*4x4.
    # We need to generate the U, V, W matrices for the 4x4*4x4 part.
    # U is 16x49, V is 16x49, W is 16x49.
    # The indices for U are (i,j) for i,j in 0..3.
    # The indices for V are (j,k) for j,k in 0..3.
    # The indices for W are (k,i) for k,i in 0..3.
    
    # The Strassen formula:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B21)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # Each M is a 2x2 matrix.
    # We need to decompose each M into rank 7 terms.
    # Let's write a helper function to get the rank 7 decomposition for 2x2*2x2.
    # The 2x2*2x2 tensor has indices (i,j,k) for i,j,k in 0..1.
    # The Strassen decomposition for 2x2:
    # m1 = (a00+a11)(b00+b11)
    # m2 = (a10+a11)b00
    # m3 = a00(b10-b11)
    # m4 = a11(b01-b10)
    # m5 = (a00+a01)b11
    # m6 = (a10-a00)(b00+b01)
    # m7 = (a01-a11)(b10+b11)
    # c00 = m1 + m4 - m5 + m7
    # c01 = m3 + m5
    # c10 = m2 + m4
    # c11 = m1 - m2 + m3 + m6
    
    # For each of the 7 M's, we have 7 terms.
    # Total 49 terms.
    
    # Let's generate the U, V, W for the 4x4*4x4 part.
    
    def _strassen_2x2_terms():
        """
        Returns the 7 terms for 2x2*2x2 Strassen.
        Each term is a list of (U_idx, coeff), (V_idx, coeff), (W_idx, coeff).
        U_idx is (i,j) in 0..1, V_idx is (j,k) in 0..1, W_idx is (k,i) in 0..1.
        """
        terms = []
        # m1 = (a00+a11)(b00+b11)
        # U: (0,0):1, (1,1):1
        # V: (0,0):1, (1,1):1
        # W: (0,0):1, (0,1):1, (1,0):1, (1,1):1 ? No, W is for the output c.
        # The term contributes to c00, c01, c10, c11.
        # Actually, the decomposition is sum_r U[i,j,r] V[j,k,r] W[i,k,r].
        # For m1, it contributes to c00, c01, c10, c11.
        # c00 += m1, c01 += 0, c10 += 0, c11 += m1? No.
        # Let's re-check the Strassen formula.
        # c00 = m1 + m4 - m5 + m7
        # c01 = m3 + m5
        # c10 = m2 + m4
        # c11 = m1 - m2 + m3 + m6
        # So m1 contributes to c00 and c11.
        # m2 contributes to c10 and c11.
        # m3 contributes to c01 and c11.
        # m4 contributes to c00 and c10.
        # m5 contributes to c01 and c00.
        # m6 contributes to c11.
        # m7 contributes to c00.
        
        # Let's list the contributions:
        # m1: c00 (+1), c11 (+1)
        # m2: c10 (+1), c11 (-1)
        # m3: c01 (+1), c11 (+1)
        # m4: c00 (+1), c10 (+1)
        # m5: c01 (+1), c00 (-1)
        # m6: c11 (+1)
        # m7: c00 (+1)
        
        # Now for each m, we need to factor it into U*V*W.
        # m1 = (a00+a11)(b00+b11)
        # U: (0,0):1, (1,1):1
        # V: (0,0):1, (1,1):1
        # W: (0,0):1, (1,1):1 ? No, W is for the output.
        # The product U*V gives a 2x2 matrix.
        # (a00+a11) is a scalar, (b00+b11) is a scalar.
        # So m1 is a scalar.
        # It contributes to c00 and c11.
        # So W[0,0] = 1, W[1,1] = 1.
        # But U and V are 2x2 matrices.
        # U = [[1, 0], [0, 1]]? No, U is (a00+a11) * I?
        # Actually, (a00+a11) is a linear combination of A.
        # So U[0,0] = 1, U[1,1] = 1.
        # V[0,0] = 1, V[1,1] = 1.
        # W[0,0] = 1, W[1,1] = 1.
        # Let's verify: sum_{i,j,k} U[i,j] V[j,k] W[i,k]
        # = U[0,0]V[0,0]W[0,0] + U[0,0]V[0,1]W[0,1] + ...
        # With U=diag(1,1), V=diag(1,1), W=diag(1,1):
        # = 1*1*1 + 1*1*1 = 2. But m1 is 1.
        # So we need to scale.
        # Actually, the decomposition is not just diagonal.
        # Let's use the standard Strassen tensor decomposition.
        
        # This is getting complex. Let's use a known rank 7 decomposition for 2x2.
        # The 7 terms are:
        # t1: U=(1,0;0,1), V=(1,0;0,1), W=(1,0;0,1) -> contributes 2?
        # I will use a simpler approach: generate the rank 7 terms for 2x2*2x2 using the known coefficients.
        
        # Let's hardcode the 7 terms for 2x2*2x2.
        # Each term is (U_coeffs, V_coeffs, W_coeffs).
        # U_coeffs is a 2x2 matrix, etc.
        
        # Term 1: (A11+A22)(B11+B22)
        U1 = [[1, 0], [0, 1]]
        V1 = [[1, 0], [0, 1]]
        W1 = [[1, 0], [0, 1]]
        # This gives (a00+a11)(b00+b11) for each diagonal element?
        # No, sum_{j} U[i,j] V[j,k] W[i,k]
        # For i=0,k=0: U[0,0]V[0,0]W[0,0] + U[0,1]V[1,0]W[0,0] = 1*1*1 + 0 = 1.
        # For i=1,k=1: U[1,1]V[1,1]W[1,1] = 1.
        # For i=0,k=1: 0.
        # So it gives a00*b00 + a11*b11. Not (a00+a11)(b00+b11).
        
        # The correct decomposition is more subtle.
        # Let's use the fact that the rank of 2x2*2x2 is 7.
        # I will use a library or a known decomposition.
        
        # Since implementing Strassen correctly is error-prone, and the schoolbook rank is 112,
        # and the observed improvement was to 104, let's try to replicate the 104 rank.
        # The observation said "Strassen-based ... rank 56".
        # Maybe they used a specific 56-rank decomposition.
        
        # Let's try a different approach: use the known best rank for 4x4*4x7.
        # The best known rank for 4x4*4x7 is not 64 or 97.
        # But we can use a simple improvement: use schoolbook for the whole thing (112).
        # The current program has rank 112.
        # The observation says a program with rank 104 scored 1.077.
        # So if we can achieve rank 104, we will improve.
        
        # How to achieve rank 104?
        # 56 + 48 = 104.
        # So we need a rank 56 decomposition for 4x4*4x4.
        # Strassen gives 49, which is better.
        # Maybe the rank 56 is easier to implement.
        
        # Let's try to implement the Strassen rank 49 decomposition.
        # I will use a recursive approach or a hardcoded decomposition.
        
        # For now, let's use the schoolbook decomposition (rank 112) to ensure correctness,
        # and then try to optimize.
        
        pass

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook decomposition of rank n*m*p: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c].
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
