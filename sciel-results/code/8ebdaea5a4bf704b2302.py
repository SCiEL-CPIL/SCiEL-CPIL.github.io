"""Tensor decomposition for 4x4x5 matrix multiplication."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    rank = 13
    U = [[0] * rank for _ in range(n * m)]  # 16 x 13
    V = [[0] * rank for _ in range(m * p)]  # 20 x 13
    W = [[0] * rank for _ in range(p * n)]  # 20 x 13
    
    # F(4) for 4x4x4 part: uses columns 0-3 of B (k=0,1,2,3)
    # Schoolbook for 4x4x1 part: uses column 4 of B (k=4)
    
    # F(4) algorithm - rank 9 for 4x4x4
    # Hadamard matrix H4 (up to scaling):
    # H4 = [[1, 1, 1, 1],
    #       [1, -1, 1, -1],
    #       [1, 1, -1, -1],
    #       [1, -1, -1, 1]]
    
    # The F(4) decomposition:
    # For the 4x4x4 part, we use 9 products.
    # The structure is:
    # - 4 products for the "diagonal" blocks using H4 on both sides
    # - 4 products for off-diagonal
    # - 1 product for the remaining
    
    # Actually, let me use a verified rank-9 construction.
    # The 9 terms for F(4) on 4x4x4:
    
    # Term 0: (a00+a01+a02+a03)(b00+b10+b20+b30)
    U[0][0] = 1; U[1][0] = 1; U[2][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[5][0] = 1; V[10][0] = 1; V[15][0] = 1
    W[0][0] = 1; W[1][0] = 1; W[2][0] = 1; W[3][0] = 1
    W[4][0] = 1; W[5][0] = 1; W[6][0] = 1; W[7][0] = 1
    W[8][0] = 1; W[9][0] = 1; W[10][0] = 1; W[11][0] = 1
    W[12][0] = 1; W[13][0] = 1; W[14][0] = 1; W[15][0] = 1
    
    # This is getting too complex to verify by hand. Let me use a different strategy.
    # I'll use the known fact that rank(4,4,5) <= 13 and construct it via:
    # - F(4) for 4x4x4 (rank 9)
    # - 4 additional terms for the 5th column
    
    # For the 5th column (k=4): C[i][4] = sum_j A[i][j] * B[j][4]
    # This is a 4x4x1 problem. The rank of 4x4x1 is 4 (schoolbook).
    # So total rank = 9 + 4 = 13.
    
    # Let me construct the F(4) part properly.
    # I'll use a known rank-9 decomposition for 4x4x4.
    
    # The F(4) algorithm:
    # Define H = [[1,1,1,1],[1,-1,1,-1],[1,1,-1,-1],[1,-1,-1,1]]
    # 
    # The 9 products are indexed by (r,s) where r,s in {0,1,2,3} but with r+s < 4
    # Actually no. Let me just use a verified construction.
    
    # Known rank-9 for 4x4x4 (from Bini et al. or similar):
    # I'll use the following construction based on polynomial multiplication.
    
    # Let me just use schoolbook for everything to be safe, but that's rank 80.
    # Actually, let me try to implement F(4) correctly.
    
    # Reset
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # F(4) part: 9 terms for 4x4x4
    # Using the Winograd F(4) algorithm
    # H4 transform matrix
    H4 = [[1, 1, 1, 1],
          [1, -1, 1, -1],
          [1, 1, -1, -1],
          [1, -1, -1, 1]]
    
    # The F(4) algorithm for C = A * B where A, B are 4x4:
    # Step 1: A' = H4 * A (transform rows)
    # Step 2: B' = B * H4 (transform cols)  
    # Step 3: C' = A' * B' (element-wise, 16 products)
    # Step 4: C = H4 * C' * H4 (inverse transforms)
    # 
    # But this gives 16 products, not 9. The F(4) algorithm is different.
    # 
    # The actual F(4) for matrix multiplication uses the fact that:
    # C[i,j] = sum_k A[i,k] B[k,j]
    # 
    # We can group: for each pair (i,j), C[i,j] = dot(A[i,:], B[:,j])
    # 
    # The F(4) algorithm exploits that we can compute 4x4 = 16 entries with 9 products
    # by using the structure of the Hadamard transform.
    
    # Let me use a different, verified rank-9 construction.
    # From the literature, the 9 products for 4x4x4 are:
    
    # m0 = (a00+a01+a02+a03)(b00+b10+b20+b30)
    # m1 = (a00-a01+a02-a03)(b00-b10+b20-b30)  
    # m2 = (a00+a01-a02-a03)(b00+b10-b20-b30)
    # m3 = (a00-a01-a02+a03)(b00-b10-b20+b30)
    # m4 = (a10+a11+a12+a13)(b01+b11+b21+b31)
    # m5 = (a10-a11+a12-a13)(b01-b11+b21-b31)
    # m6 = (a10+a11-a12-a13)(b01+b11-b21-b31)
    # m7 = (a10-a11-a12+a13)(b01-b11-b21+b31)
    # m8 = (a20+a21+a22+a23+a30+a31+a32+a33)(b02+b12+b22+b32+b03+b13+b23+b33)
    
    # No, m8 is wrong. Let me think differently.
    
    # I'll use the schoolbook for the entire 4x4x5 = 80 rank decomposition.
    # This is guaranteed correct but not optimal.
    
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
