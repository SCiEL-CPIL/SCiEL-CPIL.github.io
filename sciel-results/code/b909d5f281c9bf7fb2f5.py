import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    # Strassen factors for 2x2x2
    U_s = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=int)
    
    V_s = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 1],
        [0, 0, -1, 1, 0, 0, 0],
        [1, 0, 0, 0, 1, 0, 0]
    ], dtype=int)
    
    W_s = np.array([
        [1, 0, 0, 1, 1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0, 1, 1]
    ], dtype=int)
    
    # Kronecker products for 4x4x4
    U_44 = np.kron(U_s, U_s)
    V_44 = np.kron(V_s, V_s)
    W_44 = np.kron(W_s, W_s)
    
    # Extend to 4x4x8 by duplicating
    # U is 16x49, need 16x98
    U_98 = np.hstack([U_44, U_44])
    
    # V is 16x49, need 32x98
    # First 16 rows for B1, next 16 for B2
    V_98 = np.vstack([V_44, V_44])
    
    # W is 16x49, need 32x98
    # First 16 rows for C1, next 16 for C2
    W_98 = np.vstack([W_44, W_44])
    
    return {"U": U_98.tolist(), "V": V_98.tolist(), "W": W_98.tolist()}
# EVOLVE-BLOCK-END
