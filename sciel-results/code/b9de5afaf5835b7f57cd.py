import numpy as np
import time

def construct():
    """
    Constructs an exact tensor decomposition for matrix multiplication of size 3x4x5.
    Rank 56.
    """
    n, m, p = 3, 4, 5
    
    U_parts = []
    V_parts = []
    W_parts = []
    
    def add_block(u_indices, v_indices, w_indices, U_blk, V_blk, W_blk):
        rank = U_blk.shape[1]
        for r in range(rank):
            u_vec = np.zeros(n * m)
            v_vec = np.zeros(m * p)
            w_vec = np.zeros(p * n)
            
            for i, idx in enumerate(u_indices):
                u_vec[idx] = U_blk[i, r]
            for i, idx in enumerate(v_indices):
                v_vec[idx] = V_blk[i, r]
            for i, idx in enumerate(w_indices):
                w_vec[idx] = W_blk[i, r]
                
            U_parts.append(u_vec)
            V_parts.append(v_vec)
            W_parts.append(w_vec)

    def strassen_2x2x2():
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = -1
        
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1
        
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        
        return U, V, W

    def schoolbook_2x2x1():
        U = np.zeros((4, 4))
        V = np.zeros((2, 4))
        W = np.zeros((2, 4))
        
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[1, 1] = 1; V[1, 1] = 1; W[0, 1] = 1
        U[2, 2] = 1; V[0, 2] = 1; W[1, 2] = 1
        U[3, 3] = 1; V[1, 3] = 1; W[1, 3] = 1
        
        return U, V, W

    def schoolbook_1x2x2():
        U = np.zeros((2, 4))
        V = np.zeros((4, 4))
        W = np.zeros((2, 4))
        
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[1, 1] = 1; V[2, 1] = 1; W[0, 1] = 1
        U[0, 2] = 1; V[1, 2] = 1; W[1, 2] = 1
        U[1, 3] = 1; V[3, 3] = 1; W[1, 3] = 1
        
        return U, V, W

    def schoolbook_1x2x1():
        U = np.zeros((2, 2))
        V = np.zeros((2, 2))
        W = np.zeros((1, 2))
        
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[1, 1] = 1; V[1, 1] = 1; W[0, 1] = 1
        
        return U, V, W

    U_b, V_b, W_b = strassen_2x2x2()
    add_block([0, 1, 4, 5], [0, 1, 5, 6], [0, 3, 1, 4], U_b, V_b, W_b)
    
    U_b, V_b, W_b = strassen_2x2x2()
    add_block([0, 1, 4, 5], [2, 3, 7, 8], [6, 9, 7, 10], U_b, V_b, W_b)
    
    U_b, V_b, W_b = schoolbook_2x2x1()
    add_block([0, 1, 4, 5], [4, 9], [12, 13], U_b, V_b, W_b)
    
    U_b, V_b, W_b = strassen_2x2x2()
    add_block([2, 3, 6, 7], [10, 11, 15, 16], [0, 3, 1, 4], U_b, V_b, W_b)
    
    U_b, V_b, W_b = strassen_2x2x2()
    add_block([2, 3, 6, 7], [12, 13, 17, 18], [6, 9, 7, 10], U_b, V_b, W_b)
    
    U_b, V_b, W_b = schoolbook_2x2x1()
    add_block([2, 3, 6, 7], [14, 19], [12, 13], U_b, V_b, W_b)
    
    U_b, V_b, W_b = schoolbook_1x2x2()
    add_block([8, 9], [0, 1, 5, 6], [2, 5], U_b, V_b, W_b)
    
    U_b, V_b, W_b = schoolbook_1x2x2()
    add_block([8, 9], [2, 3, 7, 8], [8, 11], U_b, V_b, W_b)
    
    U_b, V_b, W_b = schoolbook_1x2x1()
    add_block([8, 9], [4, 9], [14], U_b, V_b, W_b)
    
    U_b, V_b, W_b = schoolbook_1x2x2()
    add_block([10, 11], [10, 11, 15, 16], [2, 5], U_b, V_b, W_b)
    
    U_b, V_b, W_b = schoolbook_1x2x2()
    add_block([10, 11], [12, 13, 17, 18], [8, 11], U_b, V_b, W_b)
    
    U_b, V_b, W_b = schoolbook_1x2x1()
    add_block([10, 11], [14, 19], [14], U_b, V_b, W_b)
    
    U = np.array(U_parts).T
    V = np.array(V_parts).T
    W = np.array(W_parts).T
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
