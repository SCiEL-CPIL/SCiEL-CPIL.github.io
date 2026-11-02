import numpy as np

DIMENSIONS = [2, 2, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = 21
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Define Strassen vectors for one 2x2 block
    # U indices: 0:11, 1:12, 2:21, 3:22
    # V indices (block): 0:11, 1:12, 2:21, 3:22
    # W indices (block): 0:11, 1:12, 2:21, 3:22
    
    # M1 = (a11+a22)(b11+b22)
    u1 = [1, 0, 0, 1]
    v1 = [1, 0, 0, 1]
    w1 = [1, 0, 0, 1] # c11, c22
    
    # M2 = (a21+a22)b11
    u2 = [0, 0, 1, 1]
    v2 = [1, 0, 0, 0]
    w2 = [0, 0, 1, -1] # c21, c22 (with -1)
    
    # M3 = a11(b12-b22)
    u3 = [1, 0, 0, 0]
    v3 = [0, 1, 0, -1]
    w3 = [0, 1, 0, 1] # c12, c22
    
    # M4 = a22(b21-b11)
    u4 = [0, 0, 0, 1]
    v4 = [-1, 0, 1, 0]
    w4 = [1, 0, 1, 0] # c11, c21
    
    # M5 = (a11+a12)b22
    u5 = [1, 1, 0, 0]
    v5 = [0, 0, 0, 1]
    w5 = [-1, 1, 0, 0] # c11 (-1), c12
    
    # M6 = (a21-a11)(b11+b12)
    u6 = [-1, 0, 1, 0]
    v6 = [1, 1, 0, 0]
    w6 = [0, 0, 0, 1] # c22
    
    # M7 = (a12-a22)(b21+b22)
    u7 = [0, 1, 0, -1]
    v7 = [0, 0, 1, 1]
    w7 = [1, 0, 0, 0] # c11
    
    U_block = np.array([u1, u2, u3, u4, u5, u6, u7]).T
    V_block = np.array([v1, v2, v3, v4, v5, v6, v7]).T
    W_block = np.array([w1, w2, w3, w4, w5, w6, w7]).T
    
    for l in range(3):
        # U is same for all blocks
        U[:, l*7 : (l+1)*7] = U_block
        
        # V mapping
        # Block V indices 0,1,2,3 map to full V indices:
        # j=0, k=2l -> 2l
        # j=0, k=2l+1 -> 2l+1
        # j=1, k=2l -> 6+2l
        # j=1, k=2l+1 -> 7+2l
        v_rows = [2*l, 2*l+1, 6+2*l, 7+2*l]
        for i_block in range(4):
            for c in range(7):
                V[v_rows[i_block], l*7 + c] = V_block[i_block, c]
        
        # W mapping
        # Block W indices 0,1,2,3 map to full W indices:
        # k=2l, i=0 -> 2l
        # k=2l, i=1 -> 2l+1
        # k=2l+1, i=0 -> 2l+2
        # k=2l+1, i=1 -> 2l+3
        w_rows = [2*l, 2*l+1, 2*l+2, 2*l+3]
        for i_block in range(4):
            for c in range(7):
                W[w_rows[i_block], l*7 + c] = W_block[i_block, c]
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
