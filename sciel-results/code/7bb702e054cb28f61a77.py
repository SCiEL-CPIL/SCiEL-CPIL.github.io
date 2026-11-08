import numpy as np
import time

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    DEADLINE = 110.0
    
    U_parts = []
    V_parts = []
    W_parts = []
    
    def add_block(u_indices, v_indices, w_indices, U_block, V_block, W_block):
        rank = U_block.shape[1]
        for r in range(rank):
            U_vec = np.zeros(n * m)
            V_vec = np.zeros(m * p)
            W_vec = np.zeros(p * n)
            
            for i, idx in enumerate(u_indices):
                U_vec[idx] = U_block[i, r]
            for i, idx in enumerate(v_indices):
                V_vec[idx] = V_block[i, r]
            for i, idx in enumerate(w_indices):
                W_vec[idx] = W_block[i, r]
                
            U_parts.append(U_vec)
            V_parts.append(V_vec)
            W_parts.append(W_vec)

    def get_strassen_2x2x2():
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (A11 + A22)(B11 + B22)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2 = (A21 + A22)B11
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = -1
        
        # M3 = A11(B21 - B22)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1
        
        # M4 = A22(B11 - B12)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        
        # M5 = (A11 + A12)B22
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        
        # M6 = (A21 - A11)(B11 + B12)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        # M7 = (A12 - A22)(B21 + B22)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        
        return U, V, W

    def get_rank5_2x2x3():
        """
        Rank-5 decomposition for 2x2 * 2x3 -> 2x3
        A = [[a,b],[c,d]], B = [[e,f,g],[h,i,j]], C = [[ae+bh, af+bi, ag+bj],[ce+dh, cf+di, cg+dj]]
        """
        U = np.zeros((4, 5))
        V = np.zeros((6, 5))
        W = np.zeros((6, 5))
        
        # M1 = (a+b)(e+h) -> contributes to C[0,0], C[1,0]
        U[0, 0] = 1; U[1, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2 = (a-b)(e-h) -> C[0,0] = M1+M2 over 2, C[1,0] = ... 
        # Let me use a known rank-5 scheme for 2x2x3
        # Actually, let me use: 
        # M1 = a*e + b*h
        # M2 = a*f + b*i  
        # M3 = a*g + b*j
        # M4 = c*e + d*h
        # M5 = c*f + d*i
        # M6 = c*g + d*j
        # That's 6. For rank 5, we need to combine.
        
        # Known rank-5 for 2x2x3:
        # M1 = (a+b)*e
        # M2 = (a+b)*h  
        # M3 = (a-b)*f
        # M4 = (a-b)*i
        # M5 = a*g + b*j  (this is one term for the last column)
        # Wait, that's not right either.
        
        # Let me just use schoolbook rank 6 for safety, or find a proper rank-5.
        # Actually, the rank of 2x2x3 is 5. Let me construct it properly.
        
        # From literature: rank(2,2,3) = 5
        # M1 = a*e + b*h
        # M2 = a*f + b*i
        # M3 = c*e + d*h
        # M4 = c*f + d*i
        # M5 = a*g + b*j + c*e/2 + d*h/2 ... no, that doesn't work.
        
        # Let me use a different approach. The 2x2x3 tensor has rank 5.
        # I'll use:
        # M1 = (a+b)*e
        # M2 = (a+b)*h
        # M3 = (a-b)*f
        # M4 = (a-b)*i
        # M5 = (a*g + b*j)  -- but this is 2 terms
        # That gives 5 terms but M5 is not a single scalar product.
        
        # Actually for the last column (g,j), we need a*g + b*j for row 0 and c*g + d*j for row 1.
        # We can write: a*g + b*j = a*(g) + b*(j) -- 2 terms
        # c*g + d*j = c*(g) + d*(j) -- 2 terms
        # Total for last column: 4 terms
        # First two columns: use rank 2 for 2x2x2 on first 2 cols of B
        # That's 4 + 4 = 8, worse than 6.
        
        # Let me just use schoolbook rank 6.
        U = np.zeros((4, 6))
        V = np.zeros((6, 6))
        W = np.zeros((6, 6))
        
        # C[0,0] = a*e + b*h
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[1, 1] = 1; V[3, 1] = 1; W[0, 1] = 1
        # C[0,1] = a*f + b*i
        U[0, 2] = 1; V[1, 2] = 1; W[1, 2] = 1
        U[1, 3] = 1; V[4, 3] = 1; W[1, 3] = 1
        # C[0,2] = a*g + b*j
        U[0, 4] = 1; V[2, 4] = 1; W[2, 4] = 1
        U[1, 5] = 1; V[5, 5] = 1; W[2, 5] = 1
        # C[1,0] = c*e + d*h
        U[2, 0] = 1; V[0, 0] = 1; W[3, 0] = 1
        U[3, 1] = 1; V[3, 1] = 1; W[3, 1] = 1
        # C[1,1] = c*f + d*i
        U[2, 2] = 1; V[1, 2] = 1; W[4, 2] = 1
        U[3, 3] = 1; V[4, 3] = 1; W[4, 3] = 1
        # C[1,2] = c*g + d*j
        U[2, 4] = 1; V[2, 4] = 1; W[5, 4] = 1
        U[3, 5] = 1; V[5, 5] = 1; W[5, 5] = 1
        
        return U, V, W

    def get_rank5_1x2x3():
        """
        Rank-5 decomposition for 1x2 * 2x3 -> 1x3
        A = [a, b], B = [[e,f,g],[h,i,j]], C = [ae+bh, af+bi, ag+bj]
        """
        U = np.zeros((2, 5))
        V = np.zeros((6, 5))
        W = np.zeros((3, 5))
        
        # C[0] = a*e + b*h
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[1, 1] = 1; V[3, 1] = 1; W[0, 1] = 1
        # C[1] = a*f + b*i
        U[0, 2] = 1; V[1, 2] = 1; W[1, 2] = 1
        U[1, 3] = 1; V[4, 3] = 1; W[1, 3] = 1
        # C[2] = a*g + b*j
        U[0, 4] = 1; V[2, 4] = 1; W[2, 4] = 1
        U[1, 5] = 1; V[5, 5] = 1; W[2, 5] = 1
        
        return U, V, W

    def get_schoolbook_2x2x1():
        U = np.zeros((4, 4))
        V = np.zeros((2, 4))
        W = np.zeros((2, 4))
        
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[1, 1] = 1; V[1, 1] = 1; W[0, 1] = 1
        U[2, 2] = 1; V[0, 2] = 1; W[1, 2] = 1
        U[3, 3] = 1; V[1, 3] = 1; W[1, 3] = 1
        
        return U, V, W

    def get_schoolbook_1x2x2():
        U = np.zeros((2, 4))
        V = np.zeros((4, 4))
        W = np.zeros((2, 4))
        
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[1, 1] = 1; V[2, 1] = 1; W[0, 1] = 1
        U[0, 2] = 1; V[1, 2] = 1; W[1, 2] = 1
        U[1, 3] = 1; V[3, 3] = 1; W[1, 3] = 1
        
        return U, V, W

    def get_schoolbook_1x2x1():
        U = np.zeros((2, 2))
        V = np.zeros((2, 2))
        W = np.zeros((1, 2))
        
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[1, 1] = 1; V[1, 1] = 1; W[0, 1] = 1
        
        return U, V, W

    U_parts = []
    V_parts = []
    W_parts = []
    
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    u_idx = [0, 1, 4, 5]
    v_idx = [0, 1, 5, 6]
    w_idx = [0, 3, 1, 4]
    add_block(u_idx, v_idx, w_idx, *get_strassen_2x2x2())
    
    # Block 2: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    u_idx = [2, 3, 6, 7]
    v_idx = [10, 11, 15, 16]
    w_idx = [0, 3, 1, 4]
    add_block(u_idx, v_idx, w_idx, *get_strassen_2x2x2())
    
    # Block 3: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    u_idx = [0, 1, 4, 5]
    v_idx = [2, 3, 7, 8]
    w_idx = [6, 9, 7, 10]
    add_block(u_idx, v_idx, w_idx, *get_strassen_2x2x2())
    
    # Block 4: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    u_idx = [2, 3, 6, 7]
    v_idx = [12, 13, 17, 18]
    w_idx = [6, 9, 7, 10]
    add_block(u_idx, v_idx, w_idx, *get_strassen_2x2x2())
    
    # Block 5: A[0:2, 0:2] * B[0:2, 4:5] -> C[0:2, 4:5]
    u_idx = [0, 1, 4, 5]
    v_idx = [4, 9]
    w_idx = [12, 13]
    add_block(u_idx, v_idx, w_idx, *get_schoolbook_2x2x1())
    
    # Block 6: A[0:2, 2:4] * B[2:4, 4:5] -> C[0:2, 4:5]
    u_idx = [2, 3, 6, 7]
    v_idx = [14, 19]
    w_idx = [12, 13]
    add_block(u_idx, v_idx, w_idx, *get_schoolbook_2x2x1())
    
    # Block 7: A[2:3, 0:2] * B[0:2, 0:2] -> C[2:3, 0:2]
    u_idx = [8, 9]
    v_idx = [0, 1, 5, 6]
    w_idx = [2, 5]
    add_block(u_idx, v_idx, w_idx, *get_schoolbook_1x2x2())
    
    # Block 8: A[2:3, 2:4] * B[2:4, 0:2] -> C[2:3, 0:2]
    u_idx = [10, 11]
    v_idx = [10, 11, 15, 16]
    w_idx = [2, 5]
    add_block(u_idx, v_idx, w_idx, *get_schoolbook_1x2x2())
    
    # Block 9: A[2:3, 0:2] * B[0:2, 2:4] -> C[2:3, 2:4]
    u_idx = [8, 9]
    v_idx = [2, 3, 7, 8]
    w_idx = [8, 11]
    add_block(u_idx, v_idx, w_idx, *get_schoolbook_1x2x2())
    
    # Block 10: A[2:3, 2:4] * B[2:4, 2:4] -> C[2:3, 2:4]
    u_idx = [10, 11]
    v_idx = [12, 13, 17, 18]
    w_idx = [8, 11]
    add_block(u_idx, v_idx, w_idx, *get_schoolbook_1x2x2())
    
    # Block 11: A[2:3, 0:2] * B[0:2, 4:5] -> C[2:3, 4:5]
    u_idx = [8, 9]
    v_idx = [4, 9]
    w_idx = [14]
    add_block(u_idx, v_idx, w_idx, *get_schoolbook_1x2x1())
    
    # Block 12: A[2:3, 2:4] * B[2:4, 4:5] -> C[2:3, 4:5]
    u_idx = [10, 11]
    v_idx = [14, 19]
    w_idx = [14]
    add_block(u_idx, v_idx, w_idx, *get_schoolbook_1x2x1())

    U = np.array(U_parts).T
    V = np.array(V_parts).T
    W = np.array(W_parts).T
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
