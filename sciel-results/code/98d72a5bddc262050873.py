import numpy as np

def construct():
    """
    Construct a rank-25 exact tensor decomposition for 2x2 * 2x7 matrix multiplication.
    Uses Strassen's rank-7 algorithm for three 2-column blocks and schoolbook for the last column.
    Total rank: 3 * 7 + 4 = 25.
    """
    n, m, p = 2, 2, 7
    rank = 25
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Helper to add a term to U, V, W
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val, rank_idx):
        U[u_idx, rank_idx] += u_val
        V[v_idx, rank_idx] += v_val
        W[w_idx, rank_idx] += w_val

    # Strassen block for columns c, c+1
    def strassen_block(c, r_start):
        # M1 = (A00 + A11) * (B0,c + B1,c+1)
        add_term(0, 1, 7*0 + c, 1, 2*c + 0, 1, r_start + 0)
        add_term(3, 1, 7*0 + c, 1, 2*c + 0, 1, r_start + 0)
        add_term(0, 1, 7*0 + c, 1, 2*(c+1) + 1, 1, r_start + 0)
        add_term(3, 1, 7*0 + c, 1, 2*(c+1) + 1, 1, r_start + 0)
        
        # M2 = (A10 + A11) * B0,c
        add_term(2, 1, 7*0 + c, 1, 2*c + 1, 1, r_start + 1)
        add_term(3, 1, 7*0 + c, 1, 2*c + 1, 1, r_start + 1)
        add_term(2, 1, 7*0 + c, 1, 2*(c+1) + 1, -1, r_start + 1)
        add_term(3, 1, 7*0 + c, 1, 2*(c+1) + 1, -1, r_start + 1)
        
        # M3 = A00 * (B0,c+1 - B1,c+1)
        add_term(0, 1, 7*0 + (c+1), 1, 2*(c+1) + 0, 1, r_start + 2)
        add_term(0, 1, 7*0 + (c+1), 1, 2*(c+1) + 1, 1, r_start + 2)
        add_term(0, 1, 7*1 + (c+1), -1, 2*(c+1) + 0, 1, r_start + 2)
        add_term(0, 1, 7*1 + (c+1), -1, 2*(c+1) + 1, 1, r_start + 2)
        
        # M4 = A11 * (B1,c - B0,c)
        add_term(3, 1, 7*1 + c, 1, 2*c + 0, 1, r_start + 3)
        add_term(3, 1, 7*1 + c, 1, 2*c + 1, 1, r_start + 3)
        add_term(3, 1, 7*0 + c, -1, 2*c + 0, 1, r_start + 3)
        add_term(3, 1, 7*0 + c, -1, 2*c + 1, 1, r_start + 3)
        
        # M5 = (A00 + A01) * B1,c+1
        add_term(0, 1, 7*1 + (c+1), 1, 2*c + 0, -1, r_start + 4)
        add_term(1, 1, 7*1 + (c+1), 1, 2*c + 0, -1, r_start + 4)
        add_term(0, 1, 7*1 + (c+1), 1, 2*(c+1) + 0, 1, r_start + 4)
        add_term(1, 1, 7*1 + (c+1), 1, 2*(c+1) + 0, 1, r_start + 4)
        
        # M6 = (A10 - A00) * (B0,c + B0,c+1)
        add_term(2, 1, 7*0 + c, 1, 2*(c+1) + 1, 1, r_start + 5)
        add_term(0, -1, 7*0 + c, 1, 2*(c+1) + 1, 1, r_start + 5)
        add_term(2, 1, 7*0 + (c+1), 1, 2*(c+1) + 1, 1, r_start + 5)
        add_term(0, -1, 7*0 + (c+1), 1, 2*(c+1) + 1, 1, r_start + 5)
        
        # M7 = (A01 - A11) * (B1,c + B1,c+1)
        add_term(1, 1, 7*1 + c, 1, 2*c + 0, 1, r_start + 6)
        add_term(3, -1, 7*1 + c, 1, 2*c + 0, 1, r_start + 6)
        add_term(1, 1, 7*1 + (c+1), 1, 2*c + 0, 1, r_start + 6)
        add_term(3, -1, 7*1 + (c+1), 1, 2*c + 0, 1, r_start + 6)

    # Apply Strassen to columns (0,1), (2,3), (4,5)
    strassen_block(0, 0)
    strassen_block(2, 7)
    strassen_block(4, 14)
    
    # Schoolbook for column 6
    # A00 * B0,6 -> C0,6
    add_term(0, 1, 7*0 + 6, 1, 2*6 + 0, 1, 21)
    # A01 * B1,6 -> C0,6
    add_term(1, 1, 7*1 + 6, 1, 2*6 + 0, 1, 22)
    # A10 * B0,6 -> C1,6
    add_term(2, 1, 7*0 + 6, 1, 2*6 + 1, 1, 23)
    # A11 * B1,6 -> C1,6
    add_term(3, 1, 7*1 + 6, 1, 2*6 + 1, 1, 24)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
