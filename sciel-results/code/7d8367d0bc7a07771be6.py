import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-33 tensor decomposition for 2x3 by 3x6 matrix multiplication.
    Uses Strassen decomposition for 2x2x2 blocks and outer product for 2x1x2 blocks.
    """
    n, m, p = DIMENSIONS
    # Rank is 3 * (7 + 4) = 33
    rank = 33
    
    U = np.zeros((n * m, rank), dtype=np.int8)
    V = np.zeros((m * p, rank), dtype=np.int8)
    W = np.zeros((p * n, rank), dtype=np.int8)
    
    # Define Strassen factors for 2x2x2
    # U_s, V_s, W_s are 4x7
    U_s = np.array([
        [1, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, -1, -1]
    ], dtype=np.int8).T # Transpose to 7x4? No, U is (n*m, rank). 
    # Let's stick to the definitions derived above.
    # U_s[i, r] where i is index in A (0..3), r is term (0..6)
    U_s = np.array([
        [1, 0, 1, 0, 1, 0, 0], # a00
        [0, 0, 0, 0, 1, 0, 1], # a01
        [0, 1, 0, 0, 0, 1, 0], # a10
        [1, 1, 0, 1, 0, -1, -1] # a11
    ], dtype=np.int8).T # Now shape (7, 4). We need (4, 7) for U part?
    # U is (n*m, rank). So U_s should be (4, 7).
    U_s = U_s.T
    
    V_s = np.array([
        [1, 1, 0, 0, 0, 1, 0], # b00
        [0, 0, 1, 0, 0, 1, 0], # b01
        [0, 0, 0, 1, 0, 0, 1], # b10
        [1, 0, -1, -1, 1, 0, 1] # b11
    ], dtype=np.int8).T
    V_s = V_s.T # Shape (4, 7)
    
    W_s = np.array([
        [1, 0, 0, 1, -1, 0, 1], # c00 (k=0, i=0)
        [0, 1, 0, 1, 0, 0, 0], # c10 (k=0, i=1)
        [0, 0, 1, 0, 1, 0, 0], # c01 (k=1, i=0)
        [1, -1, 1, 0, 0, 1, 0] # c11 (k=1, i=1)
    ], dtype=np.int8).T
    W_s = W_s.T # Shape (4, 7)
    
    # Define Outer Product factors for 2x1x2
    # U_op, V_op, W_op are 2x4, 2x4, 4x4
    U_op = np.array([
        [1, 1, 0, 0], # a0
        [0, 0, 1, 1]  # a1
    ], dtype=np.int8) # Shape (2, 4)
    
    V_op = np.array([
        [1, 0, 1, 0], # b0
        [0, 1, 0, 1]  # b1
    ], dtype=np.int8) # Shape (2, 4)
    
    W_op = np.array([
        [1, 0, 0, 0], # c00
        [0, 1, 0, 0], # c10
        [0, 0, 1, 0], # c01
        [0, 0, 0, 1]  # c11
    ], dtype=np.int8) # Shape (4, 4)
    
    # Fill U, V, W for 3 blocks
    for b in range(3):
        col_offset = b * 11
        
        # Strassen part (cols 0-6 of the block)
        # U: Rows 0-3 (A1), Rows 4-5 (A2) are 0
        U[0:4, col_offset:col_offset+7] = U_s
        # V: Rows for B1 (block b)
        # B is 3x6. B1 is cols 2b, 2b+1.
        # V_s indices 0-3 correspond to b00, b01, b10, b11.
        # Global V indices:
        # b00: 0*6 + 2b
        # b01: 0*6 + 2b + 1
        # b10: 1*6 + 2b
        # b11: 1*6 + 2b + 1
        v_indices = [0*6 + 2*b, 0*6 + 2*b + 1, 1*6 + 2*b, 1*6 + 2*b + 1]
        for k in range(4):
            V[v_indices[k], col_offset:col_offset+7] = V_s[k, :]
            
        # W: Rows for C1 (block b)
        # C is 2x6. C1 is cols 2b, 2b+1.
        # W_s indices 0-3 correspond to c00, c10, c01, c11.
        # Global W indices (k*n + i):
        # c00: 0*2 + 0 = 0 (local k=0, i=0) -> global k=2b, i=0 -> (2b)*2 + 0
        # c10: 0*2 + 1 = 1 (local k=0, i=1) -> global k=2b, i=1 -> (2b)*2 + 1
        # c01: 1*2 + 0 = 2 (local k=1, i=0) -> global k=2b+1, i=0 -> (2b+1)*2 + 0
        # c11: 1*2 + 1 = 3 (local k=1, i=1) -> global k=2b+1, i=1 -> (2b+1)*2 + 1
        w_indices = [(2*b)*2 + 0, (2*b)*2 + 1, (2*b+1)*2 + 0, (2*b+1)*2 + 1]
        for k in range(4):
            W[w_indices[k], col_offset:col_offset+7] = W_s[k, :]
            
        # Outer Product part (cols 7-10 of the block)
        # U: Rows 4-5 (A2)
        U[4:6, col_offset+7:col_offset+11] = U_op
        # V: Rows for B2 (block b)
        # B2 is rows 2 (of A? No, B is 3x6).
        # B is split into B1 (rows 0,1) and B2 (row 2).
        # B2 is 1x2.
        # Global V indices for B2 (block b):
        # b0: 2*6 + 2b
        # b1: 2*6 + 2b + 1
        v_indices_op = [2*6 + 2*b, 2*6 + 2*b + 1]
        for k in range(2):
            V[v_indices_op[k], col_offset+7:col_offset+11] = V_op[k, :]
            
        # W: Rows for C1 (block b) - same indices as Strassen part
        for k in range(4):
            W[w_indices[k], col_offset+7:col_offset+11] = W_op[k, :]
            
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
