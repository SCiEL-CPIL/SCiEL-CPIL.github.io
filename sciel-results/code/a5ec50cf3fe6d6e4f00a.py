import numpy as np
import time

DIMENSIONS = [2, 2, 16]

def construct():
    """
    Construct the rank-minimal tensor decomposition for 2x2 * 2x16.
    
    Returns a dictionary with keys 'U', 'V', 'W' containing lists of lists.
    The rank is 56, derived from 8 independent Strassen decompositions of 2x2 * 2x2.
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=16
    
    # Strassen decomposition for 2x2 * 2x2
    # U: (4, 7), V: (4, 7), W: (4, 7)
    # Indices for A: 0:a00, 1:a01, 2:a10, 3:a11
    # Indices for B: 0:b00, 1:b01, 2:b10, 3:b11
    # Indices for C: 0:c00, 1:c01, 2:c10, 3:c11
    
    U_core = [
        [1, 0, 1, 0, 1, -1, 0], # a00
        [0, 0, 0, 0, 1,  0, 1], # a01
        [0, 1, 0, 0, 0,  1, 0], # a10
        [1, 1, 0, 1, 0,  0, -1]  # a11
    ]
    
    V_core = [
        [1, 1, 0, -1, 0, 1, 0], # b00
        [0, 0, 1, 0, 0, 1, 0], # b01
        [0, 0, 0, 1, 0, 0, 1], # b10
        [1, 0, -1, 0, 1, 0, 1]  # b11
    ]
    
    W_core = [
        [1, 0, 0, 1, -1, 0, 1], # c00
        [0, 0, 1, 0, 1, 0, 0], # c01
        [0, 1, 0, 1, 0, 0, 0], # c10
        [1, -1, 1, 0, 0, 1, 0]  # c11
    ]
    
    num_blocks = p // 2 # 8
    rank = 7 * num_blocks # 56
    
    U = [[0 for _ in range(rank)] for _ in range(n * m)]
    V = [[0 for _ in range(rank)] for _ in range(m * p)]
    W = [[0 for _ in range(rank)] for _ in range(p * n)]
    
    for t in range(num_blocks):
        col_offset = t * 7
        
        # U is the same for all blocks
        for i in range(4):
            for k in range(7):
                U[i][col_offset + k] = U_core[i][k]
        
        # V mapping:
        # Block t uses B rows 0 and 1, columns 2t and 2t+1.
        # V index is j*p + k.
        # j=0, k=2t   -> 0*p + 2t = 2t
        # j=0, k=2t+1 -> 0*p + 2t+1 = 2t+1
        # j=1, k=2t   -> 1*p + 2t = 16 + 2t
        # j=1, k=2t+1 -> 1*p + 2t+1 = 16 + 2t+1
        
        # V_core rows: 0:b00, 1:b01, 2:b10, 3:b11
        # We assign V_core[0] to row 2t
        # V_core[1] to row 2t+1
        # V_core[2] to row 16+2t
        # V_core[3] to row 16+2t+1
        
        for k in range(7):
            V[2*t][col_offset + k] = V_core[0][k]
            V[2*t+1][col_offset + k] = V_core[1][k]
            V[16+2*t][col_offset + k] = V_core[2][k]
            V[16+2*t+1][col_offset + k] = V_core[3][k]
            
        # W mapping:
        # W index is k*n + i.
        # k=2t, i=0   -> 2t*2 + 0 = 4t
        # k=2t, i=1   -> 2t*2 + 1 = 4t+1
        # k=2t+1, i=0 -> (2t+1)*2 + 0 = 4t+2
        # k=2t+1, i=1 -> (2t+1)*2 + 1 = 4t+3
        
        # W_core rows: 0:c00, 1:c01, 2:c10, 3:c11
        # W_core[0] (c00) goes to C[0, 2t] which is index 4t
        # W_core[1] (c01) goes to C[1, 2t] which is index 4t+2
        # W_core[2] (c10) goes to C[0, 2t+1] which is index 4t+1
        # W_core[3] (c11) goes to C[1, 2t+1] which is index 4t+3
        
        # Wait, let's re-verify W index mapping.
        # Output C is n x p (2 x 16).
        # C[i, k].
        # Index in flattened W is k*n + i.
        # For block t:
        # C[0, 2t]   -> i=0, k=2t   -> idx = 2t*2 + 0 = 4t.   This is c00 of the block.
        # C[0, 2t+1] -> i=0, k=2t+1 -> idx = (2t+1)*2 + 0 = 4t+2. This is c01 of the block.
        # C[1, 2t]   -> i=1, k=2t   -> idx = 2t*2 + 1 = 4t+1.   This is c10 of the block.
        # C[1, 2t+1] -> i=1, k=2t+1 -> idx = (2t+1)*2 + 1 = 4t+3. This is c11 of the block.
        
        # W_core[0] is c00. Should go to 4t.
        # W_core[1] is c01. Should go to 4t+2.
        # W_core[2] is c10. Should go to 4t+1.
        # W_core[3] is c11. Should go to 4t+3.
        
        for k in range(7):
            W[4*t][col_offset + k] = W_core[0][k]
            W[4*t+1][col_offset + k] = W_core[2][k]
            W[4*t+2][col_offset + k] = W_core[1][k]
            W[4*t+3][col_offset + k] = W_core[3][k]

    return {
        "U": U,
        "V": V,
        "W": W
    }

if __name__ == "__main__":
    # Quick validation
    res = construct()
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    
    np.random.seed(123)
    A = np.random.randint(-5, 5, (2, 2))
    B = np.random.randint(-5, 5, (2, 16))
    C_exp = A @ B
    
    C_rec = np.zeros((2, 16), dtype=int)
    for i in range(2):
        for k in range(16):
            s = 0
            for j in range(2):
                for r in range(56):
                    s += U[i*2+j, r] * V[j*16+k, r] * W[k*2+i, r]
            C_rec[i, k] = s
            
    if np.array_equal(C_exp, C_rec):
        print("Validation passed.")
    else:
        print("Validation failed.")
        print(C_exp)
        print(C_rec)
