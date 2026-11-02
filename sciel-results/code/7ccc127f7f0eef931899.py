import numpy as np

def construct():
    n, m, p = 4, 4, 4
    rank = 49
    U = np.zeros((16, rank))
    V = np.zeros((16, rank))
    W = np.zeros((16, rank))
    
    # Strassen steps for 2x2
    # u[k] is a 4x1 vector of coefficients for a00, a01, a10, a11
    # v[k] is a 4x1 vector of coefficients for b00, b01, b10, b11
    # w[k] is a 4x1 vector of coefficients for c00, c01, c10, c11
    u_2x2 = [
        [1, 0, 0, 1], # 0
        [0, 0, 1, 1], # 1
        [1, 0, 0, 0], # 2
        [0, 0, 0, 1], # 3
        [1, 1, 0, 0], # 4
        [-1, 0, 1, 0], # 5
        [0, 1, 0, -1]  # 6
    ]
    v_2x2 = [
        [1, 0, 0, 1], # 0
        [1, 0, 0, 0], # 1
        [0, 1, 0, -1], # 2
        [0, 0, 1, -1], # 3
        [0, 0, 0, 1], # 4
        [1, 1, 0, 0], # 5
        [0, 0, 1, 1]  # 6
    ]
    w_2x2 = [
        [1, 0, 0, 1], # 0
        [0, 0, 1, -1], # 1
        [0, 1, 0, 1], # 2
        [1, 0, 1, 0], # 3
        [-1, 1, 0, 0], # 4
        [0, 0, 0, 1], # 5
        [1, 0, 0, 0]  # 6
    ]
    
    # Top-level Strassen steps
    # For each s1, define the linear combinations of blocks for A' and B'
    # A' and B' are 2x2 matrices of blocks
    # We represent them as 4x4 matrices where each element is a linear combination of 4x4 elements
    # Actually, it's easier to just compute the coefficients directly.
    
    # For a given s1, the elements of A' are linear combinations of the 16 elements of A.
    # Let's define a function get_A_coeffs(s1) that returns a 4x16 matrix.
    # Each row i of this matrix gives the coefficients for a'_i (flattened) in terms of A (flattened).
    
    def get_A_coeffs(s1):
        # Returns a 4x16 matrix
        coeffs = np.zeros((4, 16))
        # A is flattened as A[i*4+j]
        # Blocks: A00 = A[0:4], A01 = A[4:8], A10 = A[8:12], A11 = A[12:16]
        # A' is 2x2, so a'_00, a'_01, a'_10, a'_11
        if s1 == 0:
            # A' = A00 + A11
            coeffs[0, 0] = 1; coeffs[0, 12] = 1 # a'_00 = A00_00 + A11_00
            coeffs[0, 1] = 1; coeffs[0, 13] = 1 # a'_01 = A00_01 + A11_01
            coeffs[0, 2] = 1; coeffs[0, 14] = 1 # a'_10 = A00_10 + A11_10
            coeffs[0, 3] = 1; coeffs[0, 15] = 1 # a'_11 = A00_11 + A11_11
        elif s1 == 1:
            # A' = A10 + A11
            coeffs[0, 8] = 1; coeffs[0, 12] = 1
            coeffs[0, 9] = 1; coeffs[0, 13] = 1
            coeffs[0, 10] = 1; coeffs[0, 14] = 1
            coeffs[0, 11] = 1; coeffs[0, 15] = 1
        elif s1 == 2:
            # A' = A00
            coeffs[0, 0] = 1
            coeffs[0, 1] = 1
            coeffs[0, 2] = 1
            coeffs[0, 3] = 1
        elif s1 == 3:
            # A' = A11
            coeffs[0, 12] = 1
            coeffs[0, 13] = 1
            coeffs[0, 14] = 1
            coeffs[0, 15] = 1
        elif s1 == 4:
            # A' = A00 + A01
            coeffs[0, 0] = 1; coeffs[0, 4] = 1
            coeffs[0, 1] = 1; coeffs[0, 5] = 1
            coeffs[0, 2] = 1; coeffs[0, 6] = 1
            coeffs[0, 3] = 1; coeffs[0, 7] = 1
        elif s1 == 5:
            # A' = A10 - A00
            coeffs[0, 8] = 1; coeffs[0, 0] = -1
            coeffs[0, 9] = 1; coeffs[0, 1] = -1
            coeffs[0, 10] = 1; coeffs[0, 2] = -1
            coeffs[0, 11] = 1; coeffs[0, 3] = -1
        elif s1 == 6:
            # A' = A01 - A11
            coeffs[0, 4] = 1; coeffs[0, 12] = -1
            coeffs[0, 5] = 1; coeffs[0, 13] = -1
            coeffs[0, 6] = 1; coeffs[0, 14] = -1
            coeffs[0, 7] = 1; coeffs[0, 15] = -1
        return coeffs
    
    def get_B_coeffs(s1):
        coeffs = np.zeros((4, 16))
        if s1 == 0:
            coeffs[0, 0] = 1; coeffs[0, 12] = 1
            coeffs[0, 1] = 1; coeffs[0, 13] = 1
            coeffs[0, 2] = 1; coeffs[0, 14] = 1
            coeffs[0, 3] = 1; coeffs[0, 15] = 1
        elif s1 == 1:
            coeffs[0, 0] = 1
            coeffs[0, 1] = 1
            coeffs[0, 2] = 1
            coeffs[0, 3] = 1
        elif s1 == 2:
            coeffs[0, 4] = 1; coeffs[0, 12] = -1
            coeffs[0, 5] = 1; coeffs[0, 13] = -1
            coeffs[0, 6] = 1; coeffs[0, 14] = -1
            coeffs[0, 7] = 1; coeffs[0, 15] = -1
        elif s1 == 3:
            coeffs[0, 8] = 1; coeffs[0, 0] = -1
            coeffs[0, 9] = 1; coeffs[0, 1] = -1
            coeffs[0, 10] = 1; coeffs[0, 2] = -1
            coeffs[0, 11] = 1; coeffs[0, 3] = -1
        elif s1 == 4:
            coeffs[0, 12] = 1
            coeffs[0, 13] = 1
            coeffs[0, 14] = 1
            coeffs[0, 15] = 1
        elif s1 == 5:
            coeffs[0, 0] = 1; coeffs[0, 4] = 1
            coeffs[0, 1] = 1; coeffs[0, 5] = 1
            coeffs[0, 2] = 1; coeffs[0, 6] = 1
            coeffs[0, 3] = 1; coeffs[0, 7] = 1
        elif s1 == 6:
            coeffs[0, 8] = 1; coeffs[0, 12] = 1
            coeffs[0, 9] = 1; coeffs[0, 13] = 1
            coeffs[0, 10] = 1; coeffs[0, 14] = 1
            coeffs[0, 11] = 1; coeffs[0, 15] = 1
        return coeffs
    
    def get_W_coeffs(s1):
        # Returns a 4x16 matrix
        # W is flattened as W[k*4+i] for C[i,k]
        # So W index is k*4 + i
        # C is 2x2 blocks: C00, C01, C10, C11
        # C00 is C[0:2, 0:2], C01 is C[0:2, 2:4], etc.
        coeffs = np.zeros((4, 16))
        if s1 == 0:
            # P0 contributes to C00 (+1) and C11 (+1)
            # C00 elements: C[0,0], C[0,1], C[1,0], C[1,1] -> W[0*4+0]=W[0], W[1*4+0]=W[4], W[0*4+1]=W[1], W[1*4+1]=W[5]
            # C11 elements: C[2,2], C[2,3], C[3,2], C[3,3] -> W[2*4+2]=W[10], W[3*4+2]=W[14], W[2*4+3]=W[11], W[3*4+3]=W[15]
            coeffs[0, 0] = 1; coeffs[0, 4] = 1; coeffs[0, 1] = 1; coeffs[0, 5] = 1
            coeffs[0, 10] = 1; coeffs[0, 14] = 1; coeffs[0, 11] = 1; coeffs[0, 15] = 1
        elif s1 == 1:
            # P1 contributes to C10 (+1) and C11 (-1)
            # C10: C[2,0], C[2,1], C[3,0], C[3,1] -> W[0*4+2]=W[2], W[1*4+2]=W[6], W[0*4+3]=W[3], W[1*4+3]=W[7]
            # C11: W[10], W[14], W[11], W[15]
            coeffs[0, 2] = 1; coeffs[0, 6] = 1; coeffs[0, 3] = 1; coeffs[0, 7] = 1
            coeffs[0, 10] = -1; coeffs[0, 14] = -1; coeffs[0, 11] = -1; coeffs[0, 15] = -1
        elif s1 == 2:
            # P2 contributes to C01 (+1) and C11 (+1)
            # C01: C[0,2], C[0,3], C[1,2], C[1,3] -> W[2*4+0]=W[8], W[3*4+0]=W[12], W[2*4+1]=W[9], W[3*4+1]=W[13]
            # C11: W[10], W[14], W[11], W[15]
            coeffs[0, 8] = 1; coeffs[0, 12] = 1; coeffs[0, 9] = 1; coeffs[0, 13] = 1
            coeffs[0, 10] = 1; coeffs[0, 14] = 1; coeffs[0, 11] = 1; coeffs[0, 15] = 1
        elif s1 == 3:
            # P3 contributes to C00 (+1) and C10 (+1)
            # C00: W[0], W[4], W[1], W[5]
            # C10: W[2], W[6], W[3], W[7]
            coeffs[0, 0] = 1; coeffs[0, 4] = 1; coeffs[0, 1] = 1; coeffs[0, 5] = 1
            coeffs[0, 2] = 1; coeffs[0, 6] = 1; coeffs[0, 3] = 1; coeffs[0, 7] = 1
        elif s1 == 4:
            # P4 contributes to C00 (-1) and C01 (+1)
            # C00: W[0], W[4], W[1], W[5]
            # C01: W[8], W[12], W[9], W[13]
            coeffs[0, 0] = -1; coeffs[0, 4] = -1; coeffs[0, 1] = -1; coeffs[0, 5] = -1
            coeffs[0, 8] = 1; coeffs[0, 12] = 1; coeffs[0, 9] = 1; coeffs[0, 13] = 1
        elif s1 == 5:
            # P5 contributes to C11 (+1)
            # C11: W[10], W[14], W[11], W[15]
            coeffs[0, 10] = 1; coeffs[0, 14] = 1; coeffs[0, 11] = 1; coeffs[0, 15] = 1
        elif s1 == 6:
            # P6 contributes to C00 (+1)
            # C00: W[0], W[4], W[1], W[5]
            coeffs[0, 0] = 1; coeffs[0, 4] = 1; coeffs[0, 1] = 1; coeffs[0, 5] = 1
        return coeffs
    
    for s1 in range(7):
        A_coeffs = get_A_coeffs(s1)
        B_coeffs = get_B_coeffs(s1)
        W_coeffs = get_W_coeffs(s1)
        
        for s2 in range(7):
            col = s1 * 7 + s2
            
            # U column is A_coeffs @ u_2x2[s2]
            U[:, col] = A_coeffs @ np.array(u_2x2[s2])
            
            # V column is B_coeffs @ v_2x2[s2]
            V[:, col] = B_coeffs @ np.array(v_2x2[s2])
            
            # W column is W_coeffs @ w_2x2[s2]
            W[:, col] = W_coeffs @ np.array(w_2x2[s2])
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
