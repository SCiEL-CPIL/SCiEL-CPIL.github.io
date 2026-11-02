import numpy as np

def construct():
    # Strassen factors for 2x2x2
    # U_sub: 4x7
    # V_sub: 4x7
    # W_sub: 4x7
    
    U_sub = np.zeros((4, 7), dtype=int)
    V_sub = np.zeros((4, 7), dtype=int)
    W_sub = np.zeros((4, 7), dtype=int)
    
    # r=1
    U_sub[0, 0] = 1; U_sub[3, 0] = 1
    V_sub[0, 0] = 1; V_sub[3, 0] = 1
    W_sub[0, 0] = 1; W_sub[3, 0] = 1
    
    # r=2
    U_sub[2, 1] = 1; U_sub[3, 1] = 1
    V_sub[0, 1] = 1
    W_sub[1, 1] = 1; W_sub[3, 1] = -1
    
    # r=3
    U_sub[0, 2] = 1
    V_sub[1, 2] = 1; V_sub[3, 2] = -1
    W_sub[2, 2] = 1; W_sub[3, 2] = 1
    
    # r=4
    U_sub[3, 3] = 1
    V_sub[2, 3] = 1; V_sub[0, 3] = -1
    W_sub[0, 3] = 1; W_sub[1, 3] = 1
    
    # r=5
    U_sub[0, 4] = 1; U_sub[1, 4] = 1
    V_sub[3, 4] = 1
    W_sub[0, 4] = -1; W_sub[2, 4] = 1
    
    # r=6
    U_sub[2, 5] = 1; U_sub[0, 5] = -1
    V_sub[0, 5] = 1; V_sub[1, 5] = 1
    W_sub[3, 5] = 1
    
    # r=7
    U_sub[1, 6] = 1; U_sub[3, 6] = -1
    V_sub[2, 6] = 1; V_sub[3, 6] = 1
    W_sub[0, 6] = 1
    
    # Global dimensions
    n, m, p = 2, 4, 4
    rank = 28
    
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    
    # Define mappings for the 4 sub-problems
    # Each sub-problem uses 7 rank dimensions
    # Sub 1: A1 B11 -> C_left
    # Sub 2: A2 B21 -> C_left
    # Sub 3: A1 B12 -> C_right
    # Sub 4: A2 B22 -> C_right
    
    # Helper to add a sub-problem
    def add_sub(u_rows, v_rows, w_rows, offset):
        for i in range(4):
            for r in range(7):
                U[u_rows[i], offset + r] += U_sub[i, r]
                V[v_rows[i], offset + r] += V_sub[i, r]
                W[w_rows[i], offset + r] += W_sub[i, r]
                
    # Sub 1: A1 (rows 0-3), B11 (rows 0,1; cols 0,1 -> idx 0,1,4,5), C_left (idx 0-3)
    # A1 indices: 0,1,2,3
    # B11 indices: 0,1,4,5
    # C_left indices: 0,1,2,3
    add_sub([0, 1, 2, 3], [0, 1, 4, 5], [0, 1, 2, 3], 0)
    
    # Sub 2: A2 (rows 4-7), B21 (rows 2,3; cols 0,1 -> idx 8,9,12,13), C_left (idx 0-3)
    # A2 indices: 4,5,6,7
    # B21 indices: 8,9,12,13
    # C_left indices: 0,1,2,3
    add_sub([4, 5, 6, 7], [8, 9, 12, 13], [0, 1, 2, 3], 7)
    
    # Sub 3: A1 (rows 0-3), B12 (rows 0,1; cols 2,3 -> idx 2,3,6,7), C_right (idx 4-7)
    # A1 indices: 0,1,2,3
    # B12 indices: 2,3,6,7
    # C_right indices: 4,5,6,7
    add_sub([0, 1, 2, 3], [2, 3, 6, 7], [4, 5, 6, 7], 14)
    
    # Sub 4: A2 (rows 4-7), B22 (rows 2,3; cols 2,3 -> idx 10,11,14,15), C_right (idx 4-7)
    # A2 indices: 4,5,6,7
    # B22 indices: 10,11,14,15
    # C_right indices: 4,5,6,7
    add_sub([4, 5, 6, 7], [10, 11, 14, 15], [4, 5, 6, 7], 21)
    
    return {'U': U, 'V': V, 'W': W}
