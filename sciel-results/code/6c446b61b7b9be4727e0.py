"""Matrix multiplication 2x4 by 4x8 with rank-minimal exact decomposition."""

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=8
    # Rank: 2 * (4*8) = 64 is the naive rank.
    # We use a block-Strassen-like approach.
    # The matrix A is 2x4, B is 4x8.
    # We can view this as computing C = A * B where C is 2x8.
    # 
    # Let's think of it as:
    # A = [A0 A1] where A0, A1 are 2x2 blocks
    # B = [B0; B1; B2; B3] where B0,B1,B2,B3 are 2x2 blocks (rows 0-1, 2-3, 4-5, 6-7)
    # C = A0*B0 + A1*B1 + A1*B2 + A1*B3  ... wait, that's not right.
    #
    # Actually A is 2x4, so A = [A_left | A_right] where each is 2x2.
    # B is 4x8, so B = [B1; B2; B3; B4] where each is 2x8.
    # C = A_left * B1 + A_right * B2 + A_right * B3 + A_right * B4? No.
    #
    # Let me re-index. A[i][j] for i in 0..1, j in 0..3.
    # B[j][k] for j in 0..3, k in 0..7.
    # C[i][k] = sum_j A[i][j] * B[j][k].
    #
    # C[0][k] = A[0][0]*B[0][k] + A[0][1]*B[1][k] + A[0][2]*B[2][k] + A[0][3]*B[3][k]
    # C[1][k] = A[1][0]*B[0][k] + A[1][1]*B[1][k] + A[1][2]*B[2][k] + A[1][3]*B[3][k]
    #
    # The standard rank is n*m*p = 2*4*8 = 64.
    #
    # Let's try a simple factorization approach. We know that for matrix multiplication
    # of sizes (n,m) x (m,p), the tensor rank is at most n*m*p.
    # For 2x4 * 4x8, we can potentially do better.
    #
    # One approach: group the columns of C.
    # C is 2x8. We can compute it as:
    # C[:, 0:4] = A[:, 0:2] * B[0:2, 0:4] + A[:, 2:4] * B[2:4, 0:4]
    # C[:, 4:8] = A[:, 0:2] * B[0:2, 4:8] + A[:, 2:4] * B[2:4, 4:8]
    #
    # Each 2x2 * 2x4 product has rank at most 2*2*4 = 16.
    # So 4 such products give rank 64 total. Same as naive.
    #
    # Let me try a different grouping. Use the fact that we have a 4-dimensional middle.
    # 
    # Actually, let's just use the straightforward decomposition but see if we can find
    # a lower rank. The known result for 2x4 * 4x8:
    # 
    # The rank of the (2,4,8) matrix multiplication tensor.
    # 
    # Let's use a simple but potentially non-optimal approach first, then optimize.
    # 
    # Approach: Use the identity that we can compute the product using rank n*m*p = 64.
    # 
    # Let me try to see if rank 32 works. That would be the rank of a 2x4 * 4x8 
    # if we could factor it as (2x4)*(4x8) -> 2*8 with rank 16 per output element... no.
    #
    # Let me just implement the standard rank-64 decomposition and see if I can improve.
    
    # Standard decomposition: rank = n*m*p = 64
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]  # 8 x 64
    V = [[0] * rank for _ in range(m * p)]  # 32 x 64
    W = [[0] * rank for _ in range(p * n)]  # 16 x 64
    
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
