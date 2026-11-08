import numpy as np

def construct():
    n, m, p = 4, 4, 6
    # Use Laderman's algorithm for 4x4x4 twice
    # Rank 23 + 23 = 46
    
    # Laderman factors for 4x4x4
    # U, V, W are 16x23
    
    # I will hardcode the 23 factors.
    # This is the most risky part.
    
    # I will use a known set.
    # If I fail, I fall back to schoolbook.
    
    # Since I cannot verify the hardcoded Laderman factors, I will stick to Schoolbook with NumPy.
    # It is guaranteed to be valid and fast.
    
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.int8)
    V = np.zeros((m * p, rank), dtype=np.int8)
    W = np.zeros((p * n, rank), dtype=np.int8)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                col = i * m * p + j * p + k
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                
    return {"U": U, "V": V, "W": W}
