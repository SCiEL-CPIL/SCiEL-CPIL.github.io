import numpy as np
import time
import itertools

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    
    start_time = time.time()
    DEADLINE = 15.0
    
    def time_left():
        return DEADLINE - (time.time() - start_time)
    
    def get_strassen_2x2x2():
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (A00 + A11)(B00 + B11)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2 = (A10 + A11)B00
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = -1
        
        # M3 = A00(B01 - B11)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1
        
        # M4 = A11(B10 - B00)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        
        # M5 = (A00 + A01)B11
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        
        # M6 = (A10 - A00)(B00 + B01)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        # M7 = (A01 - A11)(B10 + B11)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        
        return U, V, W

    def get_schoolbook_2x2x1():
        U = np.zeros((4, 4))
        V = np.zeros((2, 4))
        W = np.zeros((2, 4))
        
        # C0 = A00*B0 + A01*B1
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[1, 1] = 1; V[1, 1] = 1; W[0, 1] = 1
        # C1 = A10*B0 + A11*B1
        U[2, 2] = 1; V[0, 2] = 1; W[1, 2] = 1
        U[3, 3] = 1; V[1, 3] = 1; W[1, 3] = 1
        
        return U, V, W

    def get_schoolbook_1x2x2():
        U = np.zeros((2, 4))
        V = np.zeros((4, 4))
        W = np.zeros((2, 4))
        
        # C0 = A0*B00 + A1*B10
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[1, 1] = 1; V[2, 1] = 1; W[0, 1] = 1
        # C1 = A0*B01 + A1*B11
        U[0, 2] = 1; V[1, 2] = 1; W[1, 2] = 1
        U[1, 3] = 1; V[3, 3] = 1; W[1, 3] = 1
        
        return U, V, W

    def get_schoolbook_1x2x1():
        U = np.zeros((2, 2))
        V = np.zeros((2, 2))
        W = np.zeros((1, 2))
        
        # C0 = A0*B0 + A1*B1
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[1, 1] = 1; V[1, 1] = 1; W[0, 1] = 1
        
        return U, V, W

    # --- Block Construction ---
    
    # Top 2 rows of A (2x4) times B (4x5) -> C (2x5)
    # Split B columns into 2, 2, 1
    
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    u_idx = [0, 1, 4, 5]
    v_idx = [0, 1, 5, 6]
    w_idx = [0, 3, 1, 4]
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
            
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    add_block([0, 1, 4, 5], [0, 1, 5, 6], [0, 3, 1, 4], *get_strassen_2x2x2())
    
    # Block 2: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    add_block([2, 3, 6, 7], [10, 11, 15, 16], [0, 3, 1, 4], *get_strassen_2x2x2())
    
    # Block 3: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    add_block([0, 1, 4, 5], [2, 3, 7, 8], [6, 9, 7, 10], *get_strassen_2x2x2())
    
    # Block 4: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    add_block([2, 3, 6, 7], [12, 13, 17, 18], [6, 9, 7, 10], *get_strassen_2x2x2())
    
    # Block 5: A[0:2, 0:2] * B[0:2, 4:5] -> C[0:2, 4:5]
    add_block([0, 1, 4, 5], [4, 9], [12, 13], *get_schoolbook_2x2x1())
    
    # Block 6: A[0:2, 2:4] * B[2:4, 4:5] -> C[0:2, 4:5]
    add_block([2, 3, 6, 7], [14, 19], [12, 13], *get_schoolbook_2x2x1())
    
    # Bottom 1 row of A (1x4) times B (4x5) -> C (1x5)
    # Split B columns into 2, 2, 1
    
    # Block 7: A[2:3, 0:2] * B[0:2, 0:2] -> C[2:3, 0:2]
    add_block([8, 9], [0, 1, 5, 6], [2, 5], *get_schoolbook_1x2x2())
    
    # Block 8: A[2:3, 2:4] * B[2:4, 0:2] -> C[2:3, 0:2]
    add_block([10, 11], [10, 11, 15, 16], [2, 5], *get_schoolbook_1x2x2())
    
    # Block 9: A[2:3, 0:2] * B[0:2, 2:4] -> C[2:3, 2:4]
    add_block([8, 9], [2, 3, 7, 8], [8, 11], *get_schoolbook_1x2x2())
    
    # Block 10: A[2:3, 2:4] * B[2:4, 2:4] -> C[2:3, 2:4]
    add_block([10, 11], [12, 13, 17, 18], [8, 11], *get_schoolbook_1x2x2())
    
    # Block 11: A[2:3, 0:2] * B[0:2, 4:5] -> C[2:3, 4:5]
    add_block([8, 9], [4, 9], [14], *get_schoolbook_1x2x1())
    
    # Block 12: A[2:3, 2:4] * B[2:4, 4:5] -> C[2:3, 4:5]
    add_block([10, 11], [14, 19], [14], *get_schoolbook_1x2x1())

    U = np.array(U_parts).T
    V = np.array(V_parts).T
    W = np.array(W_parts).T
    
    # Try to reduce rank by finding linear dependencies
    if time_left() > 5.0:
        # Check if any columns of U are linearly dependent
        U_mat = np.array(U_parts).T
        V_mat = np.array(V_parts).T
        W_mat = np.array(W_parts).T
        
        # Find null space of U
        U_rank = np.linalg.matrix_rank(U_mat)
        if U_rank < U_mat.shape[1]:
            # There are dependencies
            null_U = np.linalg.matrix_rank(np.hstack([U_mat, np.eye(U_mat.shape[1])]))
            
            # For now, just return the current solution
            # A full rank reduction would require careful analysis
            
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
