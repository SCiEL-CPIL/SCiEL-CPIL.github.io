import numpy as np

def construct():
    n, m, p = 3, 4, 4
    rank = 44
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen coefficients for 2x2x2
    # t: 0..6
    S_U = [
        [1, 0, 1, 1], # 1: (a+d)
        [0, 0, 1, 1], # 2: (c+d)
        [1, 0, 0, 0], # 3: a
        [0, 0, 0, 1], # 4: d
        [1, 1, 0, 0], # 5: (a+b)
        [1, 0, -1, 0],# 6: (c-a)
        [0, 1, 0, -1] # 7: (b-d)
    ]
    S_V = [
        [1, 1, 0, 0], # 1: (e+h)
        [1, 0, 0, 0], # 2: e
        [0, 1, 0, -1],# 3: (f-h)
        [0, -1, 1, 0],# 4: (g-e)
        [0, 0, 0, 1], # 5: h
        [1, 1, 0, 0], # 6: (e+f)
        [0, 0, 1, 1]  # 7: (g+h)
    ]
    S_W = [
        [1, 0, 0, 1], # 1
        [0, 0, 1, 0], # 2
        [0, 1, 0, 1], # 3
        [1, 0, 1, 0], # 4
        [-1, 1, 0, 0],# 5
        [0, 0, 0, 1], # 6
        [1, 1, 0, 0]  # 7
    ]
    
    col = 0
    
    # Helper to add a term for 2x2x4 block
    # a_base: offset in A for the 2x2 block (col in global A index)
    # b_base: offset in B for the 2x4 block (row in global B index)
    # c_base: offset in C for the 2x2 block (row in global C index)
    def add_2x2x4(a_cols, b_rows, c_rows):
        nonlocal col
        for t in range(7):
            u_coeff = S_U[t]
            v_coeff = S_V[t]
            w_coeff = S_W[t]
            
            # Add term for B_L (cols 0,1 of B block)
            U_term = np.zeros(n * m)
            V_term = np.zeros(m * p)
            W_term = np.zeros(p * n)
            
            # Fill U
            for i in range(2):
                for j in range(2):
                    global_a = (c_rows[i] * m) + (a_cols[0] + j) # Wait, c_rows are row indices of A in global?
                    # A index: row * m + col
                    # c_rows are the global row indices of the 2x2 block in A
                    # a_cols are the global col indices of the 2x2 block in A
                    # Mapping in S_U: [a, b, c, d] -> [[a,b],[c,d]]
                    # i=0, j=0 -> a. i=0, j=1 -> b. i=1, j=0 -> c. i=1, j=1 -> d.
                    U[global_a, col] += u_coeff[i * 2 + j]
            
            # Fill V
            for j in range(2): # Rows of B block
                for k in range(2): # Cols of B block (L part is 0,1)
                    global_b = (b_rows[j] * p) + (k) # Col 0,1
                    V[global_b, col] += v_coeff[j * 2 + k]
            
            # Fill W
            for i in range(2): # Rows of C block
                for k in range(2): # Cols of C block (L part is 0,1)
                    global_c = (k * n) + (c_rows[i])
                    W[global_c, col] += w_coeff[i * 2 + k]
            
            col += 1
            
            # Add term for B_R (cols 2,3 of B block)
            U_term = np.zeros(n * m)
            V_term = np.zeros(m * p)
            W_term = np.zeros(p * n)
            
            # Fill U (Same)
            for i in range(2):
                for j in range(2):
                    global_a = (c_rows[i] * m) + (a_cols[0] + j)
                    U[global_a, col] += u_coeff[i * 2 + j]
            
            # Fill V
            for j in range(2):
                for k in range(2): # Cols 2,3
                    global_b = (b_rows[j] * p) + (2 + k)
                    V[global_b, col] += v_coeff[j * 2 + k]
            
            # Fill W
            for i in range(2):
                for k in range(2): # Cols 2,3
                    global_c = ((2 + k) * n) + (c_rows[i])
                    W[global_c, col] += w_coeff[i * 2 + k]
                    
            col += 1

    # Part A: Rows 1,2 of A, Cols 0,1 of A. Rows 0,1 of B.
    # A rows: 1, 2. A cols: 0, 1.
    # B rows: 0, 1.
    # C rows: 1, 2.
    add_2x2x4(a_cols=[0, 1], b_rows=[0, 1], c_rows=[1, 2])
    
    # Part B: Rows 1,2 of A, Cols 2,3 of A. Rows 2,3 of B.
    add_2x2x4(a_cols=[2, 3], b_rows=[2, 3], c_rows=[1, 2])
    
    # Part C: Row 0 of A. Naive 1x4x4.
    for j in range(4):
        for k in range(4):
            U[j, col] = 1.0
            V[j * 4 + k, col] = 1.0
            W[k * 3 + 0, col] = 1.0
            col += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

