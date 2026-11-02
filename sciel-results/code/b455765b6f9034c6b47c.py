import numpy as np

def construct():
    # Base Strassen decomposition for 2x2
    # U, V, W are 4x7
    U0 = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    V0 = [
        [1, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, -1, 1, 0, 1]
    ]
    W0 = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    
    # Convert to numpy arrays for easier manipulation
    U0 = np.array(U0)
    V0 = np.array(V0)
    W0 = np.array(W0)
    
    n = 4
    rank = 49
    U = np.zeros((n*n, rank))
    V = np.zeros((n*n, rank))
    W = np.zeros((n*n, rank))
    
    # Coefficients for top-level Strassen
    # alpha[k][block], beta[k][block], gamma[k][block]
    # Blocks: 00, 01, 10, 11
    # k: 0..6 (1..7)
    
    alpha = [
        [1, 0, 0, 1], # k=0 (M1)
        [0, 0, 1, 1], # k=1 (M2)
        [1, 0, 0, 0], # k=2 (M3)
        [0, 0, 0, 1], # k=3 (M4)
        [1, 1, 0, 0], # k=4 (M5)
        [-1, 0, 1, 0], # k=5 (M6)
        [0, 1, 0, -1]  # k=6 (M7)
    ]
    
    beta = [
        [1, 0, 0, 1], # k=0 (M1)
        [1, 0, 0, 0], # k=1 (M2)
        [0, 1, 0, -1], # k=2 (M3)
        [0, 0, 1, -1], # k=3 (M4)
        [0, 0, 0, 1], # k=4 (M5)
        [1, 1, 0, 0], # k=5 (M6)
        [0, 0, 1, 1]  # k=6 (M7)
    ]
    
    gamma = [
        [1, 0, 0, 1], # k=0 (M1) -> C00, C11
        [0, 0, 1, -1], # k=1 (M2) -> C10, C11
        [0, 1, 0, 1], # k=2 (M3) -> C01, C11
        [1, 0, 1, 0], # k=3 (M4) -> C00, C10
        [-1, 1, 0, 0], # k=4 (M5) -> C00, C01
        [0, 0, 0, 1], # k=5 (M6) -> C11
        [1, 0, 0, 0]  # k=6 (M7) -> C00
    ]
    
    # Helper to get block index and local index
    # Global index g (0..15)
    # Block index b (0..3): 0->00, 1->01, 2->10, 3->11
    # Local index l (0..3): 0->00, 1->01, 2->10, 3->11 within block
    def get_block_local(g):
        row = g // n
        col = g % n
        b_row = row // 2
        b_col = col // 2
        l_row = row % 2
        l_col = col % 2
        block = b_row * 2 + b_col
        local = l_row * 2 + l_col
        return block, local
    
    for k in range(7):
        for j in range(7):
            r = k * 7 + j
            for g in range(16):
                bA, lA = get_block_local(g)
                U[g, r] = alpha[k][bA] * U0[lA, j]
                
                bB, lB = get_block_local(g)
                V[g, r] = beta[k][bB] * V0[lB, j]
                
                bC, lC = get_block_local(g)
                W[g, r] = gamma[k][bC] * W0[lC, j]
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
