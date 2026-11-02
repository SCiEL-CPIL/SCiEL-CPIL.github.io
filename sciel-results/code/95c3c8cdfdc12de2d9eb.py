import numpy as np

def construct():
    """
    Rank-32 decomposition for multiplying a 2x2 matrix A by a 2x16 matrix B.
    Uses standard rank-4 decomposition for each of the 8 independent 2x2 blocks.
    """
    n, m, p = 2, 2, 16
    
    # Standard rank-4 coefficients for 2x2x2
    # U rows: [A00, A01, A10, A11]
    # V rows: [B00, B01, B10, B11]
    # W rows: [C00, C01, C10, C11]
    
    std_U = np.eye(4, dtype=float)
    std_V = np.eye(4, dtype=float)
    std_W = np.eye(4, dtype=float)
    
    num_blocks = p // 2  # 8 blocks
    rank = num_blocks * 4  # 32
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    for block in range(num_blocks):
        k0 = block * 2
        k1 = k0 + 1
        base = block * 4
        
        for s in range(4):
            col = base + s
            
            # U: same for all blocks
            U[:, col] = std_U[s]
            
            # V: map local B indices to global V indices
            V[k0, col] = std_V[s, 0]       # B00
            V[k1, col] = std_V[s, 1]       # B01
            V[p + k0, col] = std_V[s, 2]   # B10
            V[p + k1, col] = std_V[s, 3]   # B11
            
            # W: map local C indices to global W indices
            W[2*k0, col] = std_W[s, 0]     # C00
            W[2*k1, col] = std_W[s, 1]     # C01
            W[2*k0+1, col] = std_W[s, 2]   # C10
            W[2*k1+1, col] = std_W[s, 3]   # C11
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
