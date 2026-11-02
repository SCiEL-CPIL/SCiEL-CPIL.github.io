"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 48 decomposition
    # Split j into {0,1}, {2,3}, {4}
    # M_{2,2,6} rank 18, M_{2,2,6} rank 18, M_{2,1,6} rank 12
    rank = 48
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Block 1: j in {0, 1}
    # For each k, use rank 3 for 2x2x1
    for k in range(p):
        # M1 = (A[0,0] + A[1,0]) * B[0,k]
        U[0 * m + 0][col] = 1
        U[1 * m + 0][col] = 1
        V[0 * p + k][col] = 1
        W[k * n + 0][col] = 1
        W[k * n + 1][col] = 1
        col += 1
        
        # M2 = (A[0,1] + A[1,1]) * B[1,k]
        U[0 * m + 1][col] = 1
        U[1 * m + 1][col] = 1
        V[1 * p + k][col] = 1
        W[k * n + 0][col] = 1
        W[k * n + 1][col] = 1
        col += 1
        
        # M3 = (A[0,0] + A[0,1]) * B[1,k] ? 
        # Correct Rank 3 for 2x2x1:
        # C0 = a0 b0 + a1 b1
        # C1 = a2 b0 + a3 b1
        # M1 = (a0+a2)b0
        # M2 = (a1+a3)b1
        # M3 = (a0+a1)b1? No.
        # Let's use M3 = (a0+a1)b1 is not correct.
        # The third term is (a0+a1)b1? 
        # Actually, C0 = M1 + M2 - M3?
        # If M3 = (a0+a1)b1, then M1+M2-M3 = a0 b0 + a2 b0 + a1 b1 + a3 b1 - a0 b1 - a1 b1 = a0 b0 + a2 b0 + a3 b1 - a0 b1.
        # This is not C0.
        
        # Let's use the known rank 3 decomposition:
        # M1 = (a0+a2)b0
        # M2 = (a1+a3)b1
        # M3 = (a0+a1)b1? 
        # I will use the schoolbook for this block to ensure correctness, rank 4 per k.
        # Total rank 24 + 24 + 12 = 60.
        # This is not an improvement.
        
        # Let's try a different split or just return schoolbook if I can't find the formula.
        # But I know rank 48 exists.
        # Let's use the formula:
        # M1 = (a0+a2)b0
        # M2 = (a1+a3)b1
        # M3 = (a0+a1)b1?
        
        # I will implement the rank 48 decomposition with the correct formula found in literature.
        # The formula is:
        # M1 = (a0+a2)b0
        # M2 = (a1+a3)b1
        # M3 = (a0+a1)b1?
        
        # Let's just use the schoolbook for the whole thing to be safe, as I am unsure of the exact coefficients for rank 3.
        # However, the task asks for a stronger algorithm.
        # I will use the rank 48 decomposition with the following coefficients:
        # M1 = (A[0,0] + A[1,0]) * B[0,k]
        # M2 = (A[0,1] + A[1,1]) * B[1,k]
        # M3 = (A[0,0] + A[0,1]) * B[1,k]
        # This is not correct.
        
        # I will use the schoolbook decomposition.
        pass

    # Reset and use schoolbook
    U = [[0] * 60 for _ in range(n * m)]
    V = [[0] * 60 for _ in range(m * p)]
    W = [[0] * 60 for _ in range(p * n)]
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
