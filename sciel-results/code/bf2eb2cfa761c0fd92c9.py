import numpy as np

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u, v, w):
        U_cols.append(u)
        V_cols.append(v)
        W_cols.append(w)
        
    def add_strassen_block(a_rows, a_cols, b_rows, b_cols, c_rows, c_cols):
        # a_rows, c_rows: 2 indices
        # a_cols, b_rows: 2 indices (must be same set)
        # b_cols, c_cols: 2 indices
        
        def a_vec(coeffs):
            vec = [0.0] * (n * m)
            for lr in range(2):
                for lc in range(2):
                    if (lr, lc) in coeffs:
                        gi = a_rows[lr]
                        gj = a_cols[lc]
                        vec[gi * m + gj] = coeffs[(lr, lc)]
            return vec
            
        def b_vec(coeffs):
            vec = [0.0] * (m * p)
            for lr in range(2):
                for lc in range(2):
                    if (lr, lc) in coeffs:
                        gj = b_rows[lr]
                        gk = b_cols[lc]
                        vec[gj * p + gk] = coeffs[(lr, lc)]
            return vec
            
        def c_vec(coeffs):
            vec = [0.0] * (p * n)
            for lr in range(2):
                for lc in range(2):
                    if (lr, lc) in coeffs:
                        gi = c_rows[lr]
                        gk = c_cols[lc]
                        vec[gk * n + gi] = coeffs[(lr, lc)]
            return vec

        # P1 = (A00 + A11) * (B00 + B11)
        u = a_vec({(0,0): 1, (1,1): 1})
        v = b_vec({(0,0): 1, (1,1): 1})
        w = c_vec({(0,0): 1, (1,1): 1})
        add_term(u, v, w)
        
        # P2 = (A10 + A11) * B00
        u = a_vec({(1,0): 1, (1,1): 1})
        v = b_vec({(0,0): 1})
        w = c_vec({(0,0): -1})
        add_term(u, v, w)
        
        # P3 = A00 * (B01 - B11)
        u = a_vec({(0,0): 1})
        v = b_vec({(0,1): 1, (1,1): -1})
        w = c_vec({(0,1): 1, (1,1): -1})
        add_term(u, v, w)
        
        # P4 = A11 * (B10 - B00)
        u = a_vec({(1,1): 1})
        v = b_vec({(1,0): 1, (0,0): -1})
        w = c_vec({(0,0): 1})
        add_term(u, v, w)
        
        # P5 = (A00 + A01) * B11
        u = a_vec({(0,0): 1, (0,1): 1})
        v = b_vec({(1,1): 1})
        w = c_vec({(1,0): 1, (1,1): 1})
        add_term(u, v, w)
        
        # P6 = (A10 - A00) * (B00 + B01)
        u = a_vec({(1,0): 1, (0,0): -1})
        v = b_vec({(0,0): 1, (0,1): 1})
        w = c_vec({(1,0): 1, (1,1): -1})
        add_term(u, v, w)
        
        # P7 = (A01 - A11) * (B10 + B11)
        u = a_vec({(0,1): 1, (1,1): -1})
        v = b_vec({(1,0): 1, (1,1): 1})
        w = c_vec({(0,0): -1, (0,1): 1})
        add_term(u, v, w)
        
    def add_schoolbook_block(a_rows, a_cols, b_rows, b_cols, c_rows, c_cols):
        for i in a_rows:
            for j in a_cols:
                for k in c_cols:
                    # Check if j is in b_rows (it should be)
                    if j in b_rows:
                        u = [0.0] * (n * m)
                        u[i * m + j] = 1.0
                        v = [0.0] * (m * p)
                        v[j * p + k] = 1.0
                        w = [0.0] * (p * n)
                        w[k * n + i] = 1.0
                        add_term(u, v, w)
                        
    # Part 1: J1 = {0, 1}
    J1 = [0, 1]
    add_strassen_block([0, 1], J1, J1, [0, 1], [0, 1], [0, 1])
    add_strassen_block([0, 1], J1, J1, [2, 3], [0, 1], [2, 3])
    add_schoolbook_block([0, 1], J1, J1, [4], [0, 1], [4])
    add_schoolbook_block([2], J1, J1, [0, 1], [2], [0, 1])
    add_schoolbook_block([2], J1, J1, [2, 3], [2], [2, 3])
    add_schoolbook_block([2], J1, J1, [4], [2], [4])
    
    # Part 2: J2 = {2, 3}
    J2 = [2, 3]
    add_strassen_block([0, 1], J2, J2, [0, 1], [0, 1], [0, 1])
    add_strassen_block([0, 1], J2, J2, [2, 3], [0, 1], [2, 3])
    add_schoolbook_block([0, 1], J2, J2, [4], [0, 1], [4])
    add_schoolbook_block([2], J2, J2, [0, 1], [2], [0, 1])
    add_schoolbook_block([2], J2, J2, [2, 3], [2], [2, 3])
    add_schoolbook_block([2], J2, J2, [4], [2], [4])
    
    U = np.array(U_cols).T
    V = np.array(V_cols).T
    W = np.array(W_cols).T
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
