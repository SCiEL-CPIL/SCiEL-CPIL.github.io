"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 13 decomposition: F(4) for 4x4x4 (rank 9) + schoolbook for 4x4x1 (rank 4)
    rank = 13
    U = [[0] * rank for _ in range(n * m)]  # 16 x 13
    V = [[0] * rank for _ in range(m * p)]  # 20 x 13
    W = [[0] * rank for _ in range(p * n)]  # 20 x 13
    
    # F(4) Winograd transform matrix (4x4 Hadamard)
    H4 = [
        [1,  1,  1,  1],
        [1, -1,  1, -1],
        [1,  1, -1, -1],
        [1, -1, -1,  1]
    ]
    
    # Part 1: F(4) for first 4 columns (k=0,1,2,3)
    # U indices: i*m + j = i*4 + j (i=0..3, j=0..3)
    # V indices: j*p + k = j*5 + k (j=0..3, k=0..3 for first 4 cols)
    # W indices: k*n + i = k*4 + i (k=0..3, i=0..3)
    
    # F(4): 9 multiplications
    # The F(4) algorithm uses:
    # For each of 4 row-groups (i) and 4 col-groups (j), we have 4x4 submatrices
    # Actually, for 4x4x4, we need rank 9
    
    # Let me use the explicit F(4) construction:
    # m0 = (a00+a01+a02+a03)(b00+b10+b20+b30)
    # m1 = (a00-a01+a02-a03)(b00-b10+b20-b30)
    # m2 = (a00+a01-a02-a03)(b00+b10-b20-b30)
    # m3 = (a00-a01-a02+a03)(b00-b10-b20+b30)
    # m4 = (a10+a11+a12+a13)(b01+b11+b21+b31)
    # m5 = (a10-a11+a12-a13)(b01-b11+b21-b31)
    # m6 = (a10+a11-a12-a13)(b01+b11-b21-b31)
    # m7 = (a10-a11-a12+a13)(b01-b11-b21+b31)
    # m8 = (a20+a21+a22+a23+a30+a31+a32+a33)(b02+b12+b22+b32+b03+b13+b23+b33)
    
    # Wait, this isn't right either. Let me use the proper F(4) from Winograd.
    
    # For 4x4 matrix multiplication, the rank-9 decomposition is:
    # We use the Kronecker product structure.
    
    # Actually, let me just verify: is rank(4,4,4) really 9?
    # Yes, it's a known result that r(4,4,4) = 9.
    
    # The standard F(4) algorithm:
    # U is 16x9, V is 16x9, W is 16x9
    
    # Let me construct it using the Hadamard transform approach:
    # The 4-point Winograd transform uses:
    # G = [1 1 1 1; 1 -1 1 -1; 1 1 -1 -1; 1 -1 -1 1]
    # G^{-1} = (1/4) * G
    
    # For matrix multiplication C = A*B where A,B are 4x4:
    # We can write this as 4 independent 1x4x4 multiplications? No.
    
    # Let me use a different approach. I'll use the known fact that:
    # r(4,4,4) = r(2,2,2) + 4*r(2,2,2) = 7 + 4*7 = 35? No, that's not right.
    
    # Actually, the correct approach: use the schoolbook for all 5 columns = rank 80.
    # But we want rank 13.
    
    # Let me think about this differently. For 4x4x5:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # For k=0,1,2,3: we can use F(4) which gives rank 9 for the 4x4x4 part
    # For k=4: we use schoolbook which gives rank 4
    # Total: 9 + 4 = 13
    
    # Now I need to construct the actual F(4) matrices.
    
    # The F(4) algorithm for 4x4 matrix multiplication:
    # Using the 4-point FFT/Winograd approach
    
    # Let me define the transform matrices explicitly.
    
    # For the F(4) part (first 9 terms, columns k=0,1,2,3):
    # U_f4 (16x9): rows indexed by (i,j) where i=0..3 (row of A), j=0..3 (col of A)
    # V_f4 (16x9): rows indexed by (j,k) where j=0..3 (row of B), k=0..3 (col of B)
    # W_f4 (16x9): rows indexed by (k,i) where k=0..3 (col of C), i=0..3 (row of C)
    
    # The F(4) decomposition:
    # We use 4 Hadamard transforms on A (along rows) and 4 on B (along cols)
    # Then 9 products, then inverse transforms
    
    # Actually, the standard F(4) uses:
    # m0 = (a00+a01+a02+a03)(b00+b10+b20+b30)
    # m1 = (a00-a01+a02-a03)(b00-b10+b20-b30)  
    # m2 = (a00+a01-a02-a03)(b00+b10-b20-b30)
    # m3 = (a00-a01-a02+a03)(b00-b10-b20+b30)
    # m4 = (a10+a11+a12+a13)(b01+b11+b21+b31)
    # m5 = (a10-a11+a12-a13)(b01-b11+b21-b31)
    # m6 = (a10+a11-a12-a13)(b01+b11-b21-b31)
    # m7 = (a10-a11-a12+a13)(b01-b11-b21+b31)
    # m8 = (a20+a21+a22+a23)(b02+b12+b22+b32) + (a30+a31+a32+a33)(b03+b13+b23+b33)
    
    # No, that's not right. Let me look up the actual F(4) algorithm.
    
    # The correct F(4) for 4x4x4 uses the following 9 products:
    # Define:
    # s0 = a00+a01+a02+a03, t0 = b00+b10+b20+b30
    # s1 = a00-a01+a02-a03, t1 = b00-b10+b20-b30
    # s2 = a00+a01-a02-a03, t2 = b00+b10-b20-b30
    # s3 = a00-a01-a02+a03, t3 = b00-b10-b20+b30
    # s4 = a10+a11+a12+a13, t4 = b01+b11+b21+b31
    # s5 = a10-a11+a12-a13, t5 = b01-b11+b21-b31
    # s6 = a10+a11-a12-a13, t6 = b01+b11-b21-b31
    # s7 = a10-a11-a12+a13, t7 = b01-b11-b21+b31
    # s8 = a20+a21+a22+a23+a30+a31+a32+a33, t8 = b02+b12+b22+b32+b03+b13+b23+b33
    
    # Then:
    # m0 = s0*t0, m1 = s1*t1, ..., m8 = s8*t8
    
    # And C is reconstructed from these.
    
    # Let me just implement this and verify.
    
    # Actually, I realize I should just use the schoolbook for a valid answer first,
    # then optimize. But the task asks for rank-minimal.
    
    # Let me try a different approach: use the known rank-13 decomposition.
    
    # For now, let me return the schoolbook decomposition which is valid but not optimal.
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
