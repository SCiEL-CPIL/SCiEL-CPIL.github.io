import numpy as np

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Block decomposition:
    # Block 1: Columns 0-2 (3 cols) -> Winograd 3x3x3 (Rank 23)
    # Block 2: Columns 3-5 (3 cols) -> Winograd 3x3x3 (Rank 23)
    # Block 3: Columns 6-7 (2 cols) -> Schoolbook 3x3x2 (Rank 18)
    # Total rank: 64
    
    total_rank = 23 + 23 + 18
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Winograd 3x3x3 rank-23 decomposition (hardcoded)
    # U indices: [i*3+j] for i in 0..2, j in 0..2
    # V indices: [j*3+k] for j in 0..2, k in 0..2
    # W indices: [k*3+i] for k in 0..2, i in 0..2
    
    # The 23 terms of the Winograd 3x3x3 decomposition
    # Each term is (u_vec, v_vec, w_vec) where each vec is length 9
    # Using the known rank-23 decomposition from literature
    
    # I'll construct this using the border-rank approach:
    # Use a rank-24 decomposition that can be rationalized
    
    # Actually, let me use a simpler verified approach:
    # Use rank 27 schoolbook but combine with the 3x3x2 block differently
    
    # Alternative: Use R(3,3,8) <= R(3,3,4) + R(3,3,4)
    # If I can find R(3,3,4) <= 30, then R(3,3,8) <= 60
    
    # Schoolbook for 3x3x4: rank 36
    # So R(3,3,8) <= 36+36 = 72 (same as before)
    
    # Let me try to find a rank-30 decomposition for 3x3x4
    # This would give R(3,3,8) <= 60
    
    # For now, let me use the block approach with schoolbook but optimized:
    # Use rank 27 for each 3x3x3 block and rank 18 for 3x3x2
    # Total: 72 (no improvement)
    
    # I need to find a better decomposition. Let me use the known fact that
    # R(3,3,3) = 23 and hardcode a valid decomposition.
    
    # Winograd 3x3x3 decomposition (rank 23)
    # Source: Known result, hardcoding the factors
    
    # The following is a valid rank-23 decomposition for 3x3x3 matrix multiplication
    # U, V, W are 9x23 matrices with integer entries
    
    # I'll generate this using the structure of the Winograd algorithm
    # The Winograd algorithm for 3x3 uses the following transforms:
    # A' = T1 * A * T2 where T1, T2 are 4x3 matrices
    # Then C' = T3 * (A' .* B') * T4
    # This gives 16 multiplications, but for general matrix multiplication
    # the rank is 23.
    
    # Let me use a different strategy: construct a rank-24 decomposition
    # that is in the closure and rationalize it
    
    # For simplicity and correctness, I'll use the schoolbook algorithm
    # but with a smarter block structure
    
    # Actually, let me just verify: can I use rank 26 for 3x3x3?
    # If yes: 26+26+18 = 70 < 72
    
    # I'll construct a rank-26 decomposition by removing one term from schoolbook
    # and verifying it still works (it won't in general)
    
    # Best approach: hardcode the Winograd 23-rank decomposition
    # Here are the 23 terms (u, v, w) for 3x3x3:
    
    # Using the decomposition from Bini et al. or Winograd's original paper
    # I'll encode it as lists of coefficients
    
    # Term 0-8: Diagonal-like terms
    # Term 9-17: Off-diagonal terms  
    # Term 18-22: Additional terms to reach rank 23
    
    # Due to complexity of hardcoding, I'll use a fallback:
    # Use rank 27 schoolbook for 3x3x3 blocks
    # But optimize the 3x3x2 block
    
    # Schoolbook 3x3x2: rank 18
    # Can I do better? R(3,3,2) <= 15? 
    # Actually R(3,3,2) = 15 (known result)
    
    # So: 27 + 27 + 15 = 69 < 72!
    
    # Let me use R(3,3,2) = 15 decomposition
    
    # Winograd 3x3x2 decomposition (rank 15)
    # U: 9x15, V: 6x15, W: 6x15
    
    # I'll hardcode this
    
    total_rank = 27 + 27 + 15  # = 69
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Block 1: 3x3x3 schoolbook (rank 27), columns 0-2
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i * 3 + j][col] = 1
                V[j * 8 + k][col] = 1
                W[k * 3 + i][col] = 1
                col += 1
    
    # Block 2: 3x3x3 schoolbook (rank 27), columns 3-5
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i * 3 + j][col] = 1
                V[j * 8 + (k + 3)][col] = 1
                W[(k + 3) * 3 + i][col] = 1
                col += 1
    
    # Block 3: 3x3x2 rank-15 decomposition, columns 6-7
    # Winograd 3x3x2 decomposition
    # This uses 15 multiplications instead of 18
    
    # The rank-15 decomposition for 3x3x2:
    # U: 9x15, V: 6x15, W: 6x15
    # Indices: U[i*3+j], V[j*2+k], W[k*3+i] for k in 0..1
    
    # Known rank-15 decomposition for 3x3x2 matrix multiplication
    # I'll hardcode the 15 terms
    
    # Each term has (u_vec of len 9, v_vec of len 6, w_vec of len 6)
    
    # Term structure for 3x3x2 rank 15:
    # Using the Winograd algorithm for 3x2 convolution
    
    # I'll use the following verified decomposition:
    winograd_332_U = [
        [1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1]
    ]
    # This is identity (rank 9), not rank 15. I need the actual decomposition.
    
    # Since I can't guarantee the exact rank-15 factors, let me use schoolbook for 3x3x2 (rank 18)
    # and accept rank 72, OR find another optimization
    
    # Alternative: Use R(3,3,8) <= R(3,3,4) * 2 with R(3,3,4) <= 32
    # 32*2 = 64 < 72
    
    # Or use R(3,3,8) <= R(3,3,3) + R(3,3,3) + R(3,3,2)
    # With R(3,3,3)=23 and R(3,3,2)=15: 23+23+15 = 61
    
    # I need to hardcode both the rank-23 and rank-15 decompositions
    
    # For safety, I'll use schoolbook everywhere: rank 72
    # But I'll try to implement at least one improvement
    
    # Let me use the fact that I can merge some terms between blocks
    # This is complex, so I'll stick with schoolbook for now
    
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
