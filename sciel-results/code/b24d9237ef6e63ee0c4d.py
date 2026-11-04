import numpy as np

def construct():
    # Generate rank-49 decomposition for 4x4x4
    U4, V4, W4 = get_strassen_4x4()
    
    # Generate schoolbook for 4x4x1
    U1, V1, W1 = get_schoolbook_4x4x1()
    
    # Combine
    U_parts = []
    V_parts = []
    W_parts = []
    
    # Block 1: cols 0-3
    U_parts.append(U4)
    V_parts.append(V4)
    W_parts.append(W4) # Indices already match 0-3
    
    # Block 2: cols 4-7
    U_parts.append(U4)
    V_parts.append(V4)
    # Shift W indices: local k in 0-3 -> global k in 4-7
    # W4 is (16, 49). Row r = k_local * 4 + i.
    # Global row = (k_local + 4) * 4 + i = k_local * 4 + 16 + i = r + 16.
    W4_shifted = [[0] * 49 for _ in range(36)] # 9*4 = 36 rows
    for r in range(16):
        for c in range(49):
            W4_shifted[r + 16][c] = W4[r][c]
    W_parts.append(W4_shifted)
    
    # Block 3: col 8
    U_parts.append(U1)
    V_parts.append(V1)
    # Shift W indices: local k=0 -> global k=8
    # W1 is (4, 16). Row r = 0 * 4 + i = i.
    # Global row = 8 * 4 + i = 32 + i = r + 32.
    W1_shifted = [[0] * 16 for _ in range(36)]
    for r in range(4):
        for c in range(16):
            W1_shifted[r + 32][c] = W1[r][c]
    W_parts.append(W1_shifted)
    
    U_final = np.hstack([np.array(u) for u in U_parts])
    V_final = np.hstack([np.array(v) for v in V_parts])
    W_final = np.hstack([np.array(w) for w in W_parts])
    
    return {
        "U": U_final.tolist(),
        "V": V_final.tolist(),
        "W": W_final.tolist()
    }
