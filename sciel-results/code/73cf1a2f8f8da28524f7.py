import time
import numpy as np

DIMENSIONS = [2, 2, 4]

def construct():
    n, m, p = DIMENSIONS
    rank = 14
    
    # Initialize U, V, W
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen factors for 2x2x2
    # U_c, V_c, W_c are 2x2 matrices flattened to vectors of length 4
    # Indices: i in {0,1}, j in {0,1}, k in {0,1}
    # U[i*2+j], V[j*2+k], W[k*2+i]
    
    # Factor 1: M1
    u1 = np.array([1, 0, 0, 1]) # [[1,0],[0,1]]
    v1 = np.array([1, 0, 0, 1])
    w1 = np.array([1, 0, 0, 1])
    
    # Factor 2: M2
    u2 = np.array([0, 0, 1, 1]) # [[0,0],[1,1]]
    v2 = np.array([1, 0, 0, 0]) # [[1,0],[0,0]]
    w2 = np.array([0, 0, 1, -1]) # [[0,0],[1,-1]]
    
    # Factor 3: M3
    u3 = np.array([1, 0, 0, 0]) # [[1,0],[0,0]]
    v3 = np.array([0, 1, 0, -1]) # [[0,1],[0,-1]]
    w3 = np.array([0, 1, 0, 1]) # [[0,1],[0,1]]
    
    # Factor 4: M4
    u4 = np.array([0, 0, 0, 1]) # [[0,0],[0,1]]
    v4 = np.array([-1, 0, 1, 0]) # [[-1,0],[1,0]]
    w4 = np.array([1, 0, 1, 0]) # [[1,0],[1,0]]
    
    # Factor 5: M5
    u5 = np.array([1, 1, 0, 0]) # [[1,1],[0,0]]
    v5 = np.array([0, 0, 0, 1]) # [[0,0],[0,1]]
    w5 = np.array([-1, 0, 0, 1]) # [[-1,0],[0,1]]
    
    # Factor 6: M6
    u6 = np.array([-1, 0, 1, 0]) # [[-1,0],[1,0]]
    v6 = np.array([1, 1, 0, 0]) # [[1,1],[0,0]]
    w6 = np.array([0, 0, 0, 1]) # [[0,0],[0,1]]
    
    # Factor 7: M7
    u7 = np.array([0, 1, 0, -1]) # [[0,1],[0,-1]]
    v7 = np.array([0, 0, 1, 1]) # [[0,0],[1,1]]
    w7 = np.array([1, 0, 0, 0]) # [[1,0],[0,0]]
    
    factors_u = [u1, u2, u3, u4, u5, u6, u7]
    factors_v = [v1, v2, v3, v4, v5, v6, v7]
    factors_w = [w1, w2, w3, w4, w5, w6, w7]
    
    # Fill U, V, W for Block 1 (columns 0, 1 of B)
    for c in range(7):
        # U indices: i*2 + j
        U[0:4, c] = factors_u[c]
        
        # V indices: j*4 + k. k is 0 or 1.
        # v_c is [v00, v01, v10, v11] corresponding to j=0,k=0; j=0,k=1; j=1,k=0; j=1,k=1
        # V[j*4 + k]
        V[0*4 + 0, c] = factors_v[c][0] # v00
        V[0*4 + 1, c] = factors_v[c][1] # v01
        V[1*4 + 0, c] = factors_v[c][2] # v10
        V[1*4 + 1, c] = factors_v[c][3] # v11
        
        # W indices: k*2 + i. k is 0 or 1.
        # w_c is [w00, w01, w10, w11] corresponding to k=0,i=0; k=0,i=1; k=1,i=0; k=1,i=1
        # W[k*2 + i]
        W[0*2 + 0, c] = factors_w[c][0] # w00
        W[0*2 + 1, c] = factors_w[c][1] # w01
        W[1*2 + 0, c] = factors_w[c][2] # w10
        W[1*2 + 1, c] = factors_w[c][3] # w11
        
    # Fill U, V, W for Block 2 (columns 2, 3 of B)
    for c in range(7):
        # U indices: i*2 + j. Same as block 1.
        U[0:4, 7 + c] = factors_u[c]
        
        # V indices: j*4 + k. k is 2 or 3.
        # v_c is [v00, v01, v10, v11]
        # We map v00 -> V[0*4 + 2], v01 -> V[0*4 + 3], v10 -> V[1*4 + 2], v11 -> V[1*4 + 3]
        V[0*4 + 2, 7 + c] = factors_v[c][0]
        V[0*4 + 3, 7 + c] = factors_v[c][1]
        V[1*4 + 2, 7 + c] = factors_v[c][2]
        V[1*4 + 3, 7 + c] = factors_v[c][3]
        
        # W indices: k*2 + i. k is 2 or 3.
        # w_c is [w00, w01, w10, w11]
        # We map w00 -> W[2*2 + 0], w01 -> W[2*2 + 1], w10 -> W[3*2 + 0], w11 -> W[3*2 + 1]
        W[2*2 + 0, 7 + c] = factors_w[c][0]
        W[2*2 + 1, 7 + c] = factors_w[c][1]
        W[3*2 + 0, 7 + c] = factors_w[c][2]
        W[3*2 + 1, 7 + c] = factors_w[c][3]

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
