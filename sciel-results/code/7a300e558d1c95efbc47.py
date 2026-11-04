"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Using Winograd-style / block decomposition for 4x4 * 4x8
    # We can split the 4x8 matrix into two 4x4 blocks
    # Each 4x4 * 4x4 multiplication can potentially be done with rank < 64
    # But for now, let's use a structured approach that might have lower rank
    
    # Actually, let's try a direct approach with shared subcomputation
    # For matrix multiplication C = A * B where A is 4x4, B is 4x8
    
    # The standard schoolbook gives rank = 4*4*8 = 128
    # But we can do better by recognizing that for each output position (i,k),
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # Let's use a factorization based on the fact that we can share computations
    # across different k values for the same i
    
    # One approach: use the Strassen-like idea of reducing the number of multiplications
    # For 4x4 matrices, the minimum known rank is 49 (Bürgisser et al.)
    # But implementing that is complex
    
    # For now, let's implement a simple block decomposition that's at least correct
    # and potentially better than pure schoolbook if we share structure
    
    # Actually, the safest approach is to just do schoolbook correctly
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
