"""Matrix multiplication 4x5x6 decomposition via Strassen blocks."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 5, 6
    # We decompose into blocks:
    # A: 4x5, B: 5x6, C: 4x6
    # Split A into A_L (4x2) and A_R (4x3)
    # Split B into B_T (2x6) and B_B (3x6)
    # Split C into C_L (4x2) and C_R (4x4)
    # C = A_L * B_T + A_R * B_B
    # C_L = A_L * B_T[:, 0:2]  (4x2 * 2x2 = 4x2)
    # C_R = A_L * B_T[:, 2:6] + A_R * B_B  (4x2 * 2x4 + 4x3 * 3x4 = 4x4)
    
    # Block 1: (4,2,2) -> rank 32 schoolbook, or use Strassen? 
    # Actually for 4x2x2, schoolbook is 16. Let's use schoolbook for simplicity.
    # Wait, let me reconsider the block structure.
    
    # Better approach: use the known decomposition from the inspiration.
    # The inspiration mentions: "Replaces the schoolbook 4x4x2 sub-block (rank 32) with a Strassen-based 4x4x2 decomposition (rank 28)"
    
    # Let me use a cleaner block decomposition:
    # Split columns of B and rows of C:
    # C = A * B where A is 4x5, B is 5x6
    # 
    # Block 1: A[:, 0:4] * B[0:4, :] -> this is 4x4 * 4x6
    #   This can be split as:
    #   A1 * B1 where A1 is 4x4, B1 is 4x6
    #   Further split B1 into B1[:, 0:2] and B1[:, 2:6]
    #   A1 * B1[:, 0:2]: 4x4 * 4x2 = 4x2, rank 32 schoolbook
    #   A1 * B1[:, 2:6]: 4x4 * 4x4 = 4x4, rank 64 schoolbook
    #
    # Block 2: A[:, 4:5] * B[4:5, :] -> 4x1 * 1x6 = 4x6, rank 24 schoolbook
    #
    # Total schoolbook: 32 + 64 + 24 = 120
    # 
    # With Strassen on 4x4x4: rank 28 instead of 64, saving 36
    # New total: 32 + 28 + 24 = 84
    #
    # Wait, the inspiration says 101. Let me re-examine.
    
    # Actually, let me just implement a valid decomposition.
    # Start with schoolbook rank = 4*5*6 = 120
    
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
    
    assert col == rank
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
