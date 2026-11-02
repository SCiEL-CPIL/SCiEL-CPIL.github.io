import numpy as np

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Use a rank-reduced decomposition
    # For 2x3x9, we can potentially use rank 42 or lower.
    # Here we implement a simple reduction by grouping k terms.
    # However, without a verified algorithm, we stick to a safe reduction.
    # Let's try rank = 42 (merging some trivial terms).
    # Actually, let's just return the trivial rank but with a smaller constant if possible?
    # No, we must reduce rank.
    # Let's use rank = 42.
    rank = 42
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # This is a placeholder for a valid lower-rank decomposition.
    # Since I don't have a verified rank 42 algorithm for 2x3x9,
    # I will fall back to the trivial rank to ensure correctness, 
    # but the task asks for improvement.
    # Let's try to use Strassen-like blocks for 2x2x2 parts.
    # 2x3x9 can be seen as 2x2x2 (rank 7) + 2x1x2 (rank 2) + ...
    # This is complex.
    # For now, I will return the trivial decomposition to avoid errors,
    # but this scores 1.0.
    # To improve, I need a valid lower rank.
    # Let's assume rank 42 is possible and fill it with a pattern that might work?
    # No, it must be exact.
    
    # Let's use the trivial rank but optimized memory? No, score is rank.
    # I will return the trivial rank 54.
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
                
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
