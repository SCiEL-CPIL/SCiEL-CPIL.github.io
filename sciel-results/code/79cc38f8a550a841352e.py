import numpy as np

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    # We use a block decomposition.
    # Pad p=8 to p'=9 (3 blocks of 3).
    # Use schoolbook decomposition for 3x3x3 blocks.
    # Rank = 3 * 9 = 27.
    
    # Schoolbook for 3x3x3
    # U: 9x9, V: 9x9, W: 9x9
    U_block = np.zeros((9, 9))
    V_block = np.zeros((9, 9))
    W_block = np.zeros((9, 9))
    
    for i in range(3):
        for j in range(3):
            c = i * 3 + j
            U_block[i * 3 + j, c] = 1
            for k in range(3):
                V_block[j * 3 + k, c] = 1
                W_block[k * 3 + i, c] = 1
                
    # Now extend to 3x3x9
    # U is 9x27 (3 blocks of 9)
    # V is 24x27 (3 blocks of 9, but V_block is 9x9 for 3x3 B)
    # Wait, V for 3x3x3 is 9x9 (B is 3x3).
    # For 3x3x9, B is 3x9.
    # We split B into 3 blocks of 3x3.
    # V_block is 9x9.
    # V_new is 24x27.
    # For block b (0,1,2), V_new[3*b:3*b+3, 9*b:9*b+9] = V_block?
    # V_block index is j*3+k.
    # V_new index is j*9+k? No, B is 3x9.
    # Index in V_new is j*9 + k.
    # For block b, k is in [3*b, 3*b+3).
    # So V_new[j*9 + 3*b + k', 9*b + c] = V_block[j*3 + k', c].
    
    U_new = np.zeros((9, 27))
    V_new = np.zeros((24, 27))
    W_new = np.zeros((24, 27))
    
    for b in range(3):
        U_new[:, 9*b:9*b+9] = U_block
        for j in range(3):
            for k in range(3):
                V_new[j*9 + 3*b + k, 9*b:9*b+9] = V_block[j*3 + k, :]
        # W is for C (3x9).
        # W index is k*3 + i? No, W is 24x27.
        # C is 3x9. Index is k*3 + i? No, C is 3x9.
        # The task says W[k*n+i, r]. n=3.
        # So W index is k*3 + i.
        # For block b, k is in [3*b, 3*b+3).
        # W_new[3*b + k'*3 + i, 9*b + c] = W_block[k'*3 + i, c].
        for k in range(3):
            for i in range(3):
                W_new[(3*b + k)*3 + i, 9*b:9*b+9] = W_block[k*3 + i, :]
                
    # Convert to lists
    return {
        "U": U_new.tolist(),
        "V": V_new.tolist(),
        "W": W_new.tolist()
    }
