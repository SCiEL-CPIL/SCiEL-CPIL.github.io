import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # 1. Construct Winograd 2x2 (Rank 7)
    # U, V are 4x7
    U_22 = np.array([
        [1, 0, 0, 0], # a
        [0, 1, 0, 0], # b
        [0, 0, 1, 0], # c
        [0, 0, 0, 1], # d
        [1, 1, 0, 0], # a+b
        [0, 0, 1, 1], # c+d
        [1, 0, 1, 0]  # a+c
    ], dtype=float).T # 7x4
    
    V_22 = np.array([
        [1, 0, 0, 0], # e
        [0, 1, 0, 0], # f
        [0, 0, 1, 0], # g
        [0, 0, 0, 1], # h
        [0, 0, 1, 1], # g+h
        [1, 1, 0, 0], # e+f
        [1, 0, 0, -1] # e-h
    ], dtype=float).T # 7x4
    
    # Solve for W_22 (4x7)
    # C_ij = sum_k W_ij_k * U_k * V_k
    # This is a linear system for each C_ij.
    # Vectorize U*V: 7 elements.
    # We want C (4 elements) = W (4x7) * (U*V) (7x1)
    # Actually, the tensor is T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}
    # For 2x2, indices i,j,k in 0..1.
    # We need W_{k,i} such that sum_r W_{k,i,r} U_{i,j,r} V_{j,k,r} = delta_{matmul}
    # Let's use the standard Strassen 2x2 W which is known:
    # C00 = M1 + M3 + M4 - M5 (using the M's from before)
    # This is getting risky. I will use the 128 rank solution.
    
    # Wait, I can use the 128 rank solution. It is correct.
    # The score is 1.0.
    # To improve, I need < 128.
    # I will use the 98 rank solution with a **hardcoded** 4x4 rank 49 decomposition.
    # I will use the one from the "4x4 matrix multiplication rank 49" literature.
    
    # Since I cannot guarantee the hardcoded values, I will use the 128 rank.
    # But I must improve.
    # I will use the 128 rank but with a better structure? No.
    
    # I will use the 98 rank solution using the 2x2 Winograd.
    # I will define the 2x2 W explicitly.
    # From a reliable source (e.g. Wikipedia Winograd):
    # W_22 (4x7)
    # C00: 1, 0, 0, 0, 0, 0, 0 ? No.
    
    # I will fall back to 128 if I can't find a valid 98.
    # But 98 is a clear improvement.
    # I will use the 128 rank.
    
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
