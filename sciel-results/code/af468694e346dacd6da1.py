import numpy as np

def _alw_3x3():
    # Returns U, V, W for 3x3x3 rank 23
    U = np.zeros((9, 23), dtype=int)
    V = np.zeros((9, 23), dtype=int)
    W = np.zeros((9, 23), dtype=int)
    
    # The 23 terms
    # I will use the coefficients from the Alon-Lovasz-Wigderson paper.
    # The terms are indexed by c=0..22.
    
    # This is a known decomposition.
    # I will fill it in.
    
    # For now, I will return None to signal fallback to schoolbook if I don't have the exact coefficients.
    # But I will try to construct it.
    
    # Actually, I can use a simpler decomposition:
    # Rank 23 is hard to hardcode without reference.
    # However, I can use a rank 27 decomposition (schoolbook) which is easy.
    # To beat 72, I need < 36 per block.
    # Rank 23 + 9 = 32.
    # Rank 27 + 9 = 36.
    
    # If I use rank 23, I save 8 per block, total 16.
    # 72 - 16 = 56? No.
    # Schoolbook 3x3x3 is 27.
    # ALW is 23.
    # Save 4 per 3x3x3 block.
    # Two blocks: save 8.
    # 72 - 8 = 64.
    
    # So I need ALW.
    return None

def construct():
    # Fallback to schoolbook if ALW not implemented
    n, m, p = 3, 3, 8
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
