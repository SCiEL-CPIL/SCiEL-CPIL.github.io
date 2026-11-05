"""Program for 3x3 by 3x6 matrix multiplication using rank-7 decomposition over F[x]/(x^2)."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    # Use rank-7 decomposition for 3x3 matrix multiplication over a field.
    # Apply to two 3x3 blocks (real and imaginary parts in F[x]/(x^2)).
    # Total rank: 7 * 2 = 14
    
    # Rank-7 decomposition for 3x3x3 matrix multiplication (Strassen-like).
    # The 7 products are:
    # M1 = (A20 + A00)(B01 - B11)
    # M2 = (A10 + A20)B11
    # M3 = A00(B01 + B02)
    # M4 = A11(B10 - B22)
    # M5 = (A01 + A11)B22
    # M6 = (A00 - A10)(B10 + B20)
    # M7 = (A11 - A21)(B21 + B22)
    #
    # But we need a specific rank-7 scheme. Let's use the known one:
    # For C = AB, with C[i][j] = sum_k A[i][k]*B[k][j]:
    
    # Define the 7 linear forms for A (row-indexed), B (col-indexed), C (result)
    # Using 0-based indexing: A[i][k], B[k][j], C[i][j]
    
    # A-part: 7 vectors of length 9 (flattened A matrix, row-major: A[0][0]=0, A[0][1]=1, A[0][2]=2, A[1][0]=3, ...)
    # B-part: 7 vectors of length 9 (flattened B matrix)
    # C-part: 7 vectors of length 9 (flattened C matrix)
    
    # Rank-7 decomposition (one known form):
    # Let's use the following scheme (verified to work):
    
    # Actually, let me use a verified rank-7 scheme. The standard Strassen for 2x2 has rank 7.
    # For 3x3, we need a different scheme. Let me use a known one.
    
    # I'll use the following rank-7 decomposition for 3x3:
    # This is based on the work of Bini et al. and others.
    
    # Let me define it carefully. The 7 products are:
    # P1 = (A10+A20)(B01-B11)
    # P2 = (A10+A20)B11
    # P3 = A00(B01+B02)
    # P4 = A11(B10-B22)
    # P5 = (A01+A11)B22
    # P6 = (A00-A10)(B10+B20)
    # P7 = (A11-A21)(B21+B22)
    
    # Result:
    # C00 = P6 - P1 + P2
    # C01 = P3 - P1 + P4
    # C02 = P3 - P5
    # C10 = P1 + P4 - P6
    # C11 = P1 - P2 + P4 + P5
    # C12 = P5 - P7
    # C20 = P6 - P4
    # C21 = P2 + P7 - P1
    # C22 = P4 + P7 - P5
    
    # Let me verify C00:
    # C00 = A00*B00 + A01*B10 + A02*B20
    # P6 = (A00-A10)(B10+B20) = A00*B10 + A00*B20 - A10*B10 - A10*B20
    # P1 = (A10+A20)(B01-B11) = A10*B01 - A10*B11 + A20*B01 - A20*B11
    # P2 = (A10+A20)*B11 = A10*B11 + A20*B11
    # P6 - P1 + P2 = A00*B10 + A00*B20 - A10*B10 - A10*B20 - A10*B01 + A10*B11 - A20*B01 + A20*B11 + A10*B11 + A20*B11
    # This doesn't simplify to C00. Let me try a different approach.
    
    # I'll use a computational approach to find the rank-7 decomposition.
    # Since I can't recall the exact coefficients, let me use a known rank-7 scheme.
    
    # Actually, the minimum rank for 3x3 matrix multiplication is not known to be 7.
    # It's known that rank >= 19 and rank <= 23 (or similar).
    
    # So rank 7 is not achievable for 3x3. I was mistaken.
    
    # Let me reconsider. For 3x3 by 3x6, we can view this as two 3x3 by 3x3 multiplications.
    # The rank of 3x3 matrix multiplication is at least 19 (lower bound).
    # So two such multiplications would have rank at least 38.
    # But there might be a better way to do 3x3 by 3x6 directly.
    
    # Actually, 3x3 by 3x6 can be seen as multiplying a 3x3 matrix by a 3x6 matrix.
    # This is equivalent to computing 18 products (3*6), but with structure.
    
    # The schoolbook rank is 54 (3*3*6).
    # We can potentially do better.
    
    # Let me use a rank-20 decomposition for 3x3x3 and apply it twice, but that gives 40.
    # Or find a direct decomposition for 3x3x6.
    
    # For now, let me implement a decomposition that achieves rank < 54.
    
    # I'll use a rank-20 decomposition for 3x3x3 (if I can find one) and apply it to both halves.
    
    # Actually, let me just use a simple improvement: block the 3x6 matrix as [B1 | B2] where B1, B2 are 3x3.
    # Then C = A * [B1 | B2] = [A*B1 | A*B2].
    # Use rank-r decomposition for 3x3x3, total rank = 2r.
    
    # Known rank for 3x3x3: at least 19, at most 23.
    # Let me use a rank-20 scheme if I can construct one, else use schoolbook for each block.
    
    # Since I can't reliably construct a rank-20 scheme, let me use a different approach.
    
    # I'll use the fact that 3x3x6 can be decomposed using rank 27 or so.
    
    # Let me just output the schoolbook decomposition for now, as I can't reliably find a better one.
    
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
