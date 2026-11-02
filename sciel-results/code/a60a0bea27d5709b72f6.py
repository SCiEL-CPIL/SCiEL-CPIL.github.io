import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 16
    num_blocks = p // 2  # 8 blocks
    rank = 7 * num_blocks  # 56
    
    # Define base Strassen factors for 2x2x2
    # U_base: 4x7
    # V_base: 4x7 (local k=0,1)
    # W_base: 4x7 (local k=0,1)
    
    U_base = np.zeros((4, 7))
    V_base = np.zeros((4, 7))
    W_base = np.zeros((4, 7))
    
    # Fill U_base
    # Row index: i*m + j
    # Col index: s
    
    # M1
    U_base[0, 0] = 1; U_base[3, 0] = 1
    # M2
    U_base[2, 1] = 1; U_base[3, 1] = 1
    # M3
    U_base[0, 2] = 1
    # M4
    U_base[3, 3] = 1
    # M5
    U_base[0, 4] = 1; U_base[1, 4] = 1
    # M6
    U_base[2, 5] = 1; U_base[0, 5] = -1
    # M7
    U_base[1, 6] = 1; U_base[3, 6] = -1
    
    # Fill V_base
    # Row index: j*p_local + k_local (p_local=2) -> j*2 + k
    # M1
    V_base[0, 0] = 1; V_base[3, 0] = 1
    # M2
    V_base[0, 1] = 1
    # M3
    V_base[1, 2] = 1; V_base[3, 2] = -1
    # M4
    V_base[2, 3] = 1; V_base[0, 3] = -1
    # M5
    V_base[3, 4] = 1
    # M6
    V_base[0, 5] = 1; V_base[1, 5] = 1
    # M7
    V_base[2, 6] = 1; V_base[3, 6] = 1
    
    # Fill W_base
    # Row index: k_local*n + i (n=2) -> k*2 + i
    # M1
    W_base[0, 0] = 1; W_base[3, 0] = 1
    # M2
    W_base[2, 1] = 1; W_base[3, 1] = -1
    # M3
    W_base[1, 2] = 1; W_base[3, 2] = 1
    # M4
    W_base[0, 3] = 1; W_base[2, 3] = 1
    # M5
    W_base[0, 4] = -1; W_base[1, 4] = 1
    # M6
    W_base[3, 5] = 1
    # M7
    W_base[0, 6] = 1
    
    # Tile U_base: U is 4 x 56. U_base is 4 x 7.
    # U does not depend on block t, so we just repeat the columns.
    U = np.tile(U_base, (1, num_blocks)).tolist()
    
    # V_base is 4 x 7. We need to expand it to 32 x 56.
    # The rows of V_base correspond to (j, k_local).
    # Global rows are (j, k_global) where k_global = 2t + k_local.
    # We can construct V by placing V_base into the appropriate rows for each block.
    V = np.zeros((32, 56))
    for t in range(num_blocks):
        # Rows for block t: j*16 + (2t + k_local)
        # For j=0: rows 2t, 2t+1
        # For j=1: rows 16+2t, 16+2t+1
        # V_base rows: 0 (j=0,k=0), 1 (j=0,k=1), 2 (j=1,k=0), 3 (j=1,k=1)
        
        # Map V_base rows to V rows
        # V_base[0] -> V[2t]
        # V_base[1] -> V[2t+1]
        # V_base[2] -> V[16+2t]
        # V_base[3] -> V[16+2t+1]
        
        V[2*t, 7*t:7*(t+1)] = V_base[0, :]
        V[2*t+1, 7*t:7*(t+1)] = V_base[1, :]
        V[16+2*t, 7*t:7*(t+1)] = V_base[2, :]
        V[16+2*t+1, 7*t:7*(t+1)] = V_base[3, :]
        
    # W_base is 4 x 7. We need to expand it to 32 x 56.
    # W_base rows correspond to (k_local, i).
    # Global rows are (k_global, i) where k_global = 2t + k_local.
    # Row index in W: k_global * n + i = (2t + k_local) * 2 + i.
    # W_base rows: 0 (k=0,i=0), 1 (k=0,i=1), 2 (k=1,i=0), 3 (k=1,i=1)
    
    W = np.zeros((32, 56))
    for t in range(num_blocks):
        # Map W_base rows to W rows
        # W_base[0] (k=0,i=0) -> W[(2t)*2 + 0] = W[4t]
        # W_base[1] (k=0,i=1) -> W[(2t)*2 + 1] = W[4t+1]
        # W_base[2] (k=1,i=0) -> W[(2t+1)*2 + 0] = W[4t+2]
        # W_base[3] (k=1,i=1) -> W[(2t+1)*2 + 1] = W[4t+3]
        
        W[4*t, 7*t:7*(t+1)] = W_base[0, :]
        W[4*t+1, 7*t:7*(t+1)] = W_base[1, :]
        W[4*t+2, 7*t:7*(t+1)] = W_base[2, :]
        W[4*t+3, 7*t:7*(t+1)] = W_base[3, :]
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
