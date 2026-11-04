"""Matrix multiplication 3x4x4 - improved decomposition via block structure."""

def construct():
    n, m, p = 3, 4, 4
    
    # We decompose C = A @ B into three parts:
    # A = [A_1 (3x3) | A_2 (3x1)]
    # B = [B_1 (3x4); B_2 (1x4)]
    # C = A_1 @ B_1 + A_2 @ B_2
    # Further split B_1 = [B_11 (3x3) | B_12 (3x1)]
    # C = A_1 @ B_11 + A_1 @ B_12 + A_2 @ B_2
    
    # Part 1: A_1 (3x3) @ B_11 (3x3) -> C_11 (3x3)
    # Indices: i in 0..2, j in 0..2, k in 0..2
    # Rank 9 (Schoolbook)
    
    # Part 2: A_1 (3x3) @ B_12 (3x1) -> C_12 (3x1)
    # Indices: i in 0..2, j in 0..2, k = 3
    # Rank 9 (Schoolbook)
    
    # Part 3: A_2 (3x1) @ B_2 (1x4) -> C_2 (3x4)
    # Indices: i in 0..2, j = 3, k in 0..3
    # Rank 1 (Outer product)
    
    total_rank = 9 + 9 + 1
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: i in 0..2, j in 0..2, k in 0..2
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    # Part 2: i in 0..2, j in 0..2, k = 3
    for i in range(3):
        for j in range(3):
            k = 3
            U[i * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            
    # Part 3: i in 0..2, j = 3, k in 0..3
    # This is a single rank-1 term: A[i,3] * B[3,k]
    # We need to sum over k? No, the tensor entry T[i,3,k] = A[i,3] B[3,k].
    # The decomposition is sum_r U[i,3,r] V[3,k,r] W[k,i,r].
    # If we set R=1 for this part, we can't have different V for each k if U is fixed?
    # Wait. T[i,3,k] = A[i,3] * B[3,k].
    # If we use rank 1:
    # U[i,3,0] = A[i,3]? No, U must be constant coefficients.
    # The decomposition is linear.
    # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r].
    # For Part 3:
    # T[i,3,k] = A[i,3] B[3,k].
    # We can set:
    # U[i,3,r] = 1 for all i, r?
    # V[3,k,r] = 1 for all k, r?
    # W[k,i,r] = 1?
    # Then sum_r 1*1*1 = R. This gives R * A[i,3] B[3,k].
    # We want 1 * A[i,3] B[3,k].
    # So we can set R=1, and coefficients 1.
    # But we have 4 values of k.
    # The term is A[i,3] * B[3,k].
    # This is a single product of two vectors?
    # No, it's a matrix of products.
    # C_2 = A_2 B_2.
    # A_2 is 3x1. B_2 is 1x4.
    # C_2[i,k] = A_2[i] B_2[k].
    # This is an outer product.
    # The tensor T_2[i,k] (with j fixed at 3) has rank 1.
    # So we can represent it with 1 multiplication.
    # U[i,3,0] = 1
    # V[3,k,0] = 1
    # W[k,i,0] = 1
    # Then sum_0 U V W = 1.
    # This matches T[i,3,k] = A[i,3] B[3,k] if the "multiplication" is A[i,3] * B[3,k].
    # Yes.
    
    for i in range(3):
        for k in range(4):
            # We only need ONE column for this part because it's rank 1.
            # But we have already allocated columns?
            # No, we are at col=18.
            # We need to fill col 18.
            pass
            
    # Wait, the loop above for Part 3 was just a placeholder.
    # We need to fill U, V, W for col=18.
    # U[i, 3, 18] = 1 for i in 0..2
    # V[3, k, 18] = 1 for k in 0..3
    # W[k, i, 18] = 1 for k in 0..3, i in 0..2
    
    for i in range(3):
        U[i * m + 3][col] = 1
    for k in range(4):
        V[3 * p + k][col] = 1
    for i in range(3):
        for k in range(4):
            W[k * n + i][col] = 1
            
    col += 1
    
    return {"U": U, "V": V, "W": W}
