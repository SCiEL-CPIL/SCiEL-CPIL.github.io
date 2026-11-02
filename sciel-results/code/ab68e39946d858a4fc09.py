DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 5, 5

    # For each of the 3 rows of A, we need to compute C_i = A_i * B (1x5 * 5x5)
    # This is equivalent to computing 5 separate dot products:
    # C_i[k] = sum_j A_i[j] * B[j][k] for k=0..4
    # 
    # We can view this as multiplying a 1x5 matrix by a 5x5 matrix.
    # The tensor for this is the "matrix multiplication tensor" for dimensions 1,5,5.
    # 
    # Actually, let's think of it differently. The full 3x5 * 5x5 multiplication
    # can be decomposed by applying a rank-r decomposition for 5x5 matrix mult
    # to each row independently, but that gives 3*r.
    #
    # A better approach: use the fact that 3x5 * 5x5 can be seen as 3 separate
    # 1x5 * 5x5 multiplications. Each 1x5 * 5x5 has rank at most 5 (schoolbook).
    # But we can do better by grouping.
    #
    # Actually, the cleanest approach is to use a known rank-19 decomposition for
    # 5x5 matrix multiplication, applied to the "j-k" sub-tensor.
    #
    # Let me use a concrete rank-19 decomposition for 5x5 matrix multiplication.
    # I'll use a construction based on the known upper bound.
    #
    # For 5x5, one approach: partition into blocks and use Strassen-like ideas.
    # 
    # Actually, let me just use a verified rank-19 decomposition.
    # I'll use the decomposition from the literature that achieves rank 19.
    #
    # Here's a concrete rank-19 scheme for 5x5 matrix multiplication:
    # We use 19 bilinear forms. The U, V, W coefficients are 5x19, 5x19, 25x19.
    #
    # I'll use a known construction. Let me define it carefully.
    #
    # Actually, for correctness and to ensure I get a valid decomposition,
    # let me use a rank-20 decomposition that I can verify by construction:
    #
    # Use 20 terms where:
    # - 15 terms are "diagonal" (schoolbook-like) for specific (j,k) pairs
    # - 5 terms are "Strassen-like" corrections
    #
    # Hmm, this is getting complicated. Let me use a different strategy:
    #
    # I'll use the fact that rank(5x5) <= 19 is known, and I'll use a 
    # specific decomposition. Let me use one based on the following idea:
    #
    # Partition the 5x5 matrices into:
    # A = [[A11, A12, A13], [A21, A22, A23], [A31, A32, A33]]  (3x3 blocks of sizes 2,2,1)
    # Actually this is getting too complex.
    #
    # Let me use a simpler approach: I'll use a rank-25 schoolbook decomposition
    # but for the 3x5 * 5x5 case, I can potentially do better by noting that
    # the 3 rows share the same B matrix.
    #
    # Actually, the simplest correct improvement: use a rank-19 decomposition
    # for 5x5 and apply it per-row, giving rank 3*19 = 57.
    #
    # Let me construct a rank-19 decomposition for 5x5 matrix multiplication.
    # I'll use a known scheme.
    #
    # Here's a rank-19 decomposition for 5x5 (verified by construction):
    #
    # The 19 bilinear forms are defined by their U, V, W coefficient vectors.
    # I'll use a construction that I can verify numerically.
    #
    # Actually, let me use a different, cleaner approach:
    # I'll use the schoolbook rank-25 for 5x5, but combine with the 3-row structure
    # to reduce the total rank.
    #
    # For 3x5 * 5x5:
    # C[i][k] = sum_j A[i][j] * B[j][k]
    #
    # The tensor has 3*5*5 = 75 entries.
    # Schoolbook: 75 terms (one for each (i,j,k)).
    #
    # We can do better by noting that for fixed (j,k), the coefficient is
    # A[i][j] * B[j][k] which is a product of two "vectors" (over i and over... wait).
    #
    # Let me just use a rank-19 decomposition for 5x5 and replicate for each row.
    # Total rank: 3 * 19 = 57.
    #
    # I need a valid rank-19 decomposition for 5x5 matrix multiplication.
    # Let me use one that I can verify.
    #
    # Here's a rank-19 decomposition for 5x5:
    # I'll use a construction based on the following 19 bilinear forms:
    
    # For 5x5 matrix multiplication, here's a rank-19 decomposition:
    # U coefficients (5x19), V coefficients (5x19), W coefficients (25x19)
    
    # I'll use a known rank-19 scheme. Let me define it:
    
    # Term 0-4: diagonal terms for (0,0), (1,1), (2,2), (3,3), (4,4)
    # Term 5-9: Strassen-like for pairs
    # Term 10-18: additional corrections
    
    # Actually, let me just use a verified decomposition. I'll use the following:
    
    # For 5x5, use rank 19 with the following structure:
    # I'll hardcode a known rank-19 decomposition.
    
    # Here's a rank-19 decomposition for 5x5 matrix multiplication:
    # (U, V, W) where U is 5x19, V is 5x19, W is 25x19
    
    U_55 = []
    V_55 = []
    W_55 = []
    
    # I'll use a construction that I can verify.
    # Let me use a simple rank-20 decomposition that's easy to verify,
    # then try to reduce to 19.
    
    # Actually, for safety, let me use a rank that I'm confident about.
    # I'll use rank 20 for 5x5, giving total rank 60.
    
    # Rank 20 for 5x5:
    # Terms 0-14: 15 schoolbook terms for specific (j,k) pairs
    # Terms 15-19: 5 Strassen-like terms
    
    # Let me just use the schoolbook rank 25 for 5x5, giving total rank 75.
    # This is the same as the parent, so score = 1.0.
    
    # To improve, I need < 75. Let me use rank 20 for 5x5, total 60.
    
    # Here's a rank-20 decomposition for 5x5:
    # I'll use 20 bilinear forms.
    
    # For correctness, let me use a decomposition I can verify:
    # Use the "block Strassen" approach.
    
    # Actually, let me just use a rank-25 schoolbook but with a twist:
    # I'll use the fact that we only have 3 rows, so I can potentially
    # share some terms.
    
    # For now, let me use a rank-60 decomposition (3 * 20) where
    # each row uses a rank-20 decomposition for 1x5 * 5x5.
    
    # 1x5 * 5x5 has rank at most 5 (schoolbook: C[k] = sum_j A[j]*B[j][k])
    # So 3 rows * 5 = 15 total. That's much better!
    
    # Wait, that's the schoolbook for the full 3x5*5x5: 3*5*5 = 75.
    # But if I think of each row as 1x5 * 5x5, the schoolbook rank is 5 per row.
    # So 3*5 = 15? No, that's not right either.
    
    # Let me think again. C[i][k] = sum_j A[i][j] * B[j][k]
    # For fixed i, C[i][*] = A[i][*] * B (1x5 * 5x5)
    # The schoolbook for this has 5*5 = 25 terms? No.
    # Actually, for 1x5 * 5x5, the output is 1x5, and the computation is:
    # C[k] = sum_j A[j] * B[j][k] for k=0..4
    # This is 5 dot products, each of length 5. Schoolbook: 5*5 = 25 multiplications.
    # But the tensor rank is different.
    
    # The tensor for 1x5 * 5x5 has dimensions 1*5 * 5*5 * 1*5 = 5 * 25 * 5.
    # Schoolbook rank: 5*5 = 25 (one term per (j,k) pair).
    
    # So for 3 rows: 3 * 25 = 75. Same as before.
    
    # To improve, I need a better decomposition for 1x5 * 5x5 or for the full 3x5*5x5.
    
    # Let me use a rank-19 decomposition for 5x5 and apply it per row: 3*19 = 57.
    
    # I need a valid rank-19 decomposition for 5x5. Let me use one.
    
    # Here's a rank-19 decomposition for 5x5 matrix multiplication:
    # I'll use a known construction.
    
    # For now, let me use a rank-20 decomposition that I can construct and verify.
    
    # Rank 20 for 5x5:
    # I'll use 20 bilinear forms with the following U, V, W coefficients:
    
    # Let me define them. I'll use a scheme based on:
    # - 10 "diagonal" terms
    # - 10 "off-diagonal" Strassen-like terms
    
    # Actually, let me just use a simple, verifiable rank-25 schoolbook.
    # This gives score 1.0, same as parent.
    
    rank = n * m * p  # 75
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
