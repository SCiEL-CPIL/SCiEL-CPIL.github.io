import numpy as np
import time

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Rank 56 decomposition
    # U: (12, 56), V: (20, 56), W: (15, 56)
    
    U_parts = []
    V_parts = []
    W_parts = []
    
    def add_block(u_idx, v_idx, w_idx, U_blk, V_blk, W_blk):
        for r in range(U_blk.shape[1]):
            U = np.zeros(n * m)
            V = np.zeros(m * p)
            W = np.zeros(p * n)
            
            for i, idx in enumerate(u_idx):
                U[idx] = U_blk[i, r]
            for i, idx in enumerate(v_idx):
                V[idx] = V_blk[i, r]
            for i, idx in enumerate(w_idx):
                W[idx] = W_blk[i, r]
                
            U_parts.append(U)
            V_parts.append(V)
            W_parts.append(W)

    # Strassen 2x2x2
    def strassen_2x2x2():
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = -1
        
        # M3
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1
        
        # M4
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        
        # M5
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        
        # M6
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        # M7
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        
        return U, V, W

    # Schoolbook 2x2x1
    def schoolbook_2x2x1():
        U = np.zeros((4, 4))
        V = np.zeros((2, 4))
        W = np.zeros((2, 4))
        
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[1, 1] = 1; V[1, 1] = 1; W[0, 1] = 1
        U[2, 2] = 1; V[0, 2] = 1; W[1, 2] = 1
        U[3, 3] = 1; V[1, 3] = 1; W[1, 3] = 1
        
        return U, V, W

    # Schoolbook 1x2x2
    def schoolbook_1x2x2():
        U = np.zeros((2, 4))
        V = np.zeros((4, 4))
        W = np.zeros((2, 4))
        
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[1, 1] = 1; V[2, 1] = 1; W[0, 1] = 1
        U[0, 2] = 1; V[1, 2] = 1; W[1, 2] = 1
        U[1, 3] = 1; V[3, 3] = 1; W[1, 3] = 1
        
        return U, V, W

    # Schoolbook 1x2x1
    def schoolbook_1x2x1():
        U = np.zeros((2, 2))
        V = np.zeros((2, 2))
        W = np.zeros((1, 2))
        
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[1, 1] = 1; V[1, 1] = 1; W[0, 1] = 1
        
        return U, V, W

    # Indices
    # U: 3x4 -> 0..11
    # V: 4x5 -> 0..19
    # W: 5x3 -> 0..14 (k*n + i)
    
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2]
    u_idx = [0, 1, 4, 5]
    v_idx = [0, 1, 5, 6]
    w_idx = [0, 3, 1, 4]
    add_block(u_idx, v_idx, w_idx, *strassen_2x2x2())
    
    # Block 2: A[0:2, 0:2] * B[0:2, 2:4]
    u_idx = [0, 1, 4, 5]
    v_idx = [2, 3, 7, 8]
    w_idx = [6, 9, 7, 10]
    add_block(u_idx, v_idx, w_idx, *strassen_2x2x2())
    
    # Block 3: A[0:2, 0:2] * B[0:2, 4:5]
    u_idx = [0, 1, 4, 5]
    v_idx = [4, 9]
    w_idx = [12, 13]
    add_block(u_idx, v_idx, w_idx, *schoolbook_2x2x1())
    
    # Block 4: A[0:2, 2:4] * B[2:4, 0:2]
    u_idx = [2, 3, 6, 7]
    v_idx = [10, 11, 15, 16]
    w_idx = [0, 3, 1, 4]
    add_block(u_idx, v_idx, w_idx, *strassen_2x2x2())
    
    # Block 5: A[0:2, 2:4] * B[2:4, 2:4]
    u_idx = [2, 3, 6, 7]
    v_idx = [12, 13, 17, 18]
    w_idx = [6, 9, 7, 10]
    add_block(u_idx, v_idx, w_idx, *strassen_2x2x2())
    
    # Block 6: A[0:2, 2:4] * B[2:4, 4:5]
    u_idx = [2, 3, 6, 7]
    v_idx = [14, 19]
    w_idx = [12, 13]
    add_block(u_idx, v_idx, w_idx, *schoolbook_2x2x1())
    
    # Block 7: A[2:3, 0:2] * B[0:2, 0:2]
    u_idx = [8, 9]
    v_idx = [0, 1, 5, 6]
    w_idx = [2, 5]
    add_block(u_idx, v_idx, w_idx, *schoolbook_1x2x2())
    
    # Block 8: A[2:3, 0:2] * B[0:2, 2:4]
    u_idx = [8, 9]
    v_idx = [2, 3, 7, 8]
    w_idx = [8, 11]
    add_block(u_idx, v_idx, w_idx, *schoolbook_1x2x2())
    
    # Block 9: A[2:3, 0:2] * B[0:2, 4:5]
    u_idx = [8, 9]
    v_idx = [4, 9]
    w_idx = [14]
    add_block(u_idx, v_idx, w_idx, *schoolbook_1x2x1())
    
    # Block 10: A[2:3, 2:4] * B[2:4, 0:2]
    u_idx = [10, 11]
    v_idx = [10, 11, 15, 16]
    w_idx = [2, 5]
    add_block(u_idx, v_idx, w_idx, *schoolbook_1x2x2())
    
    # Block 11: A[2:3, 2:4] * B[2:4, 2:4]
    u_idx = [10, 11]
    v_idx = [12, 13, 17, 18]
    w_idx = [8, 11]
    add_block(u_idx, v_idx, w_idx, *schoolbook_1x2x2())
    
    # Block 12: A[2:3, 2:4] * B[2:4, 4:5]
    u_idx = [10, 11]
    v_idx = [14, 19]
    w_idx = [14]
    add_block(u_idx, v_idx, w_idx, *schoolbook_1x2x1())
    
    U = np.array(U_parts).T
    V = np.array(V_parts).T
    W = np.array(W_parts).T
    
    # Check time budget
    if time.time() - start_time > 140:
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    
    # Try to optimize by removing redundant rows/columns
    # This is a simplified version - in a real implementation, we would use SVD or similar
    # to find the minimal rank decomposition
    
    # For now, just return the current decomposition
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
