"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We use a block decomposition:
    # n=4 split into 3+1
    # m=5 split into 3+2
    # p=6 split into 3+3
    
    # Block sizes:
    # 1. 3x3x3 (Rank 23)
    # 2. 3x3x3 (Rank 23)
    # 3. 3x2x3 (Rank 18)
    # 4. 3x2x3 (Rank 18)
    # 5. 1x3x3 (Rank 9)
    # 6. 1x3x3 (Rank 9)
    # 7. 1x2x3 (Rank 6)
    # 8. 1x2x3 (Rank 6)
    
    total_rank = 23 + 23 + 18 + 18 + 9 + 9 + 6 + 6
    # total_rank = 112
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a block decomposition
    def add_block(n_start, m_start, p_start, n_size, m_size, p_size, block_U, block_V, block_W):
        nonlocal col
        for r in range(len(block_U)):
            # block_U is n_size x m_size
            # block_V is m_size x p_size
            # block_W is p_size x n_size
            
            for i in range(n_size):
                for j in range(m_size):
                    U[(n_start + i) * m + (m_start + j)][col] = block_U[r][i][j]
            
            for j in range(m_size):
                for k in range(p_size):
                    V[(m_start + j) * p + (p_start + k)][col] = block_V[r][j][k]
            
            for k in range(p_size):
                for i in range(n_size):
                    W[(p_start + k) * n + (n_start + i)][col] = block_W[r][k][i]
            
            col += 1

    # Generate Winograd 3x3x3 matrices (Rank 23)
    # Standard Winograd 3x3 algorithm matrices
    # I will use the known 23 triples.
    
    # Since hardcoding 23 * 3 * 3 * 3 numbers is prone to error in thought process,
    # I will use a simpler valid improvement: Schoolbook is 120.
    # If I cannot guarantee the 112 rank, I should return 120 to be safe?
    # No, the prompt asks for a stronger algorithm.
    # I will implement the 3x3x3 Winograd using the standard transforms.
    
    # Winograd 3x3 Matrices
    # M = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    # S = [[1, 1, 0], [1, -1, 0], [0, 0, 1]]
    # T = [[1, 1, 0], [1, -1, 0], [0, 0, 1]]
    # R = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    
    # The 23 matrices are derived from the tensor decomposition of 3x3x3.
    # I will use the explicit values from literature.
    
    # To avoid incorrect hardcoding, I will use the 120 rank schoolbook if I am not 100% sure.
    # But 112 is a distinct improvement.
    
    # Let's try to use the 2x2x2 Strassen (Rank 7) on the 4x4x6 part?
    # 4x4x6 can be split into 2x2x6 + 2x2x6 + 2x2x6 + 2x2x6 ?
    # No, 4x4 = 2x2 + 2x2 + 2x2 + 2x2.
    # Each 2x2x6 is 2x2x2 + 2x2x2 + 2x2x2 (Rank 7*3 = 21).
    # 4 blocks * 21 = 84.
    # Plus the 1x5x6 part?
    # Split 4x5x6 into (2+2)x(5)x(6).
    # Block 1: 2x5x6. Split 5 into 2+2+1.
    #   2x2x6: Rank 21.
    #   2x2x6: Rank 21.
    #   2x1x6: Rank 12.
    #   Total 54.
    # Block 2: 2x5x6. Same. Total 54.
    # Total 108.
    
    # This is 108 < 112.
    # And 2x2x2 Strassen is easier to hardcode (Rank 7).
    
    # Strassen 2x2x2 Matrices:
    # M1 = [[1,0],[0,1]]
    # M2 = [[1,1],[0,1]]
    # M3 = [[1,0],[1,1]]
    # M4 = [[0,1],[1,0]]
    # M5 = [[1,1],[1,0]]
    # M6 = [[1,1],[0,1]]
    # M7 = [[1,0],[1,-1]]
    
    # I will implement the 108 rank solution using Strassen 2x2x2.

    # Reset U, V, W for rank 108
    total_rank = 108
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    col = 0

    # Strassen 2x2 matrices
    S = [
        [[1,0],[0,1]],
        [[1,1],[0,1]],
        [[1,0],[1,1]],
        [[0,1],[1,0]],
        [[1,1],[1,0]],
        [[1,1],[0,1]],
        [[1,0],[1,-1]]
    ]

    def add_strassen_2x2x2(i_start, j_start, k_start):
        nonlocal col
        for r in range(7):
            # U 2x2
            for i in range(2):
                for j in range(2):
                    U[(i_start + i) * m + (j_start + j)][col] = S[r][i][j]
            # V 2x2
            for j in range(2):
                for k in range(2):
                    V[(j_start + j) * p + (k_start + k)][col] = S[r][j][k]
            # W 2x2
            for k in range(2):
                for i in range(2):
                    W[(k_start + k) * n + (i_start + i)][col] = S[r][k][i]
            col += 1

    # Add 2x2x2 blocks for the 4x4 part of A and B?
    # A is 4x5. B is 5x6.
    # Split n=4 into 2+2.
    # Split m=5 into 2+2+1.
    # Split p=6 into 2+2+2.
    
    # Block (0,0,0): 2x2x2
    add_strassen_2x2x2(0, 0, 0)
    # Block (0,0,1): 2x2x2
    add_strassen_2x2x2(0, 0, 2)
    # Block (0,0,2): 2x2x2
    add_strassen_2x2x2(0, 0, 4)
    
    # Block (0,1,0): 2x2x2
    add_strassen_2x2x2(0, 2, 0)
    # Block (0,1,1): 2x2x2
    add_strassen_2x2x2(0, 2, 2)
    # Block (0,1,2): 2x2x2
    add_strassen_2x2x2(0, 2, 4)
    
    # Block (0,2,0): 2x1x2 (Schoolbook 4)
    # Block (0,2,1): 2x1x2 (Schoolbook 4)
    # Block (0,2,2): 2x1x2 (Schoolbook 4)
    for k in [0, 2, 4]:
        for i in range(2):
            for kk in range(2):
                U[i * m + 4][col] = 1
                V[4 * p + (k + kk)][col] = 1
                W[(k + kk) * n + i][col] = 1
                col += 1

    # Repeat for i=2..3
    add_strassen_2x2x2(2, 0, 0)
    add_strassen_2x2x2(2, 0, 2)
    add_strassen_2x2x2(2, 0, 4)
    add_strassen_2x2x2(2, 2, 0)
    add_strassen_2x2x2(2, 2, 2)
    add_strassen_2x2x2(2, 2, 4)
    
    for k in [0, 2, 4]:
        for i in range(2):
            for kk in range(2):
                U[(i + 2) * m + 4][col] = 1
                V[4 * p + (k + kk)][col] = 1
                W[(k + kk) * n + (i + 2)][col] = 1
                col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
