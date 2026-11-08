"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 5, 5
    
    # Try to construct a lower-rank decomposition
    # Strategy: Use block structure and Winograd-like identities
    
    # For 4x5 * 5x5, we can use the following approach:
    # Split the 4 rows into groups and use efficient 2x5 * 5x5 multiplications
    
    # Known: R(2,5,5) can be done in 2*5*5 - 5 - 5 + 1 = 41 using Winograd
    # But let's try a more direct construction
    
    # Alternative: Use the fact that we can share computations
    # Let's try a rank-75 construction based on block diagonal + corrections
    
    # Actually, let me try a systematic search over small structured decompositions
    # Given time constraints, I'll use a known construction pattern
    
    # Construction: Use 15 "super-rows" of length 5, each computed with 5 multiplications
    # 15 * 5 = 75 multiplications
    
    # For each output position (i,k) where i in 0..3, k in 0..4:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # We can group the 4 rows: rows 0,1 and rows 2,3
    # For rows 0,1: use 2x5 * 5x5 with rank 41
    # For rows 2,3: use 2x5 * 5x5 with rank 41
    # Total: 82
    
    # Better: Use a unified construction
    
    # Let me try: rank = 75 using the following pattern
    # Each of the 20 output entries requires 5 multiplications in schoolbook
    # But we can share: group outputs that share the same B row
    
    # Actually, let's use a different approach:
    # The tensor T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
    # We want to decompose T = sum_r U_r * V_r * W_r
    
    # For now, return schoolbook as fallback, but try to improve
    
    # Let's implement a rank-80 construction using block methods
    # Split 4x5 into 2x(2x5), use 2x2 * 5x5 blocks
    
    # 2x2 * 5x5: R(2,2,5) <= 2*2*5 - 2 - 5 + 1 = 12 (Winograd bound)
    # We have two such blocks (for the two 2x2 row groups)
    # But we need to be careful with the column dimension
    
    # Let me just return a verified low-rank construction
    # I'll use a precomputed rank-75 decomposition if I can find one
    
    # For now, use schoolbook to ensure correctness, then optimize
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
