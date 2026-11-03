"""Rewritten program for matrix multiplication 2,3,8 rank minimization."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 22
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    def add_term(col, u_list, v_list, w_list):
        for r, val in u_list:
            U[r][col] += val
        for r, val in v_list:
            V[r][col] += val
        for r, val in w_list:
            W[r][col] += val
            
    col = 0
    for block in range(2):
        k = block * 4
        
        # P1: (A00 + A11) * (B0k + B1,k+1)
        add_term(col, 
                 [(0, 1), (3, 1)], 
                 [(k, 1), (8 + k + 1, 1)], 
                 [(2*k, 1), (2*k + 3, 1)])
        col += 1
        
        # P2: (A10 + A11) * B0k
        add_term(col, 
                 [(2, 1), (3, 1)], 
                 [(k, 1)], 
                 [(2*k + 1, 1)])
        col += 1
        
        # P3: A00 * (B1,k+1 - B1k)
        add_term(col, 
                 [(0, 1)], 
                 [(8 + k + 1, 1), (8 + k, -1)], 
                 [(2*k + 2, 1), (2*k + 3, 1)])
        col += 1
        
        # P4: A11 * (B0,k+1 - B0k)
        add_term(col, 
                 [(3, 1)], 
                 [(k + 1, 1), (k, -1)], 
                 [(2*k, 1), (2*k + 1, 1)])
        col += 1
        
        # P5: (A00 + A01) * B1k
        add_term(col, 
                 [(0, 1), (1, 1)], 
                 [(8 + k, 1)], 
                 [(2*k, -1), (2*k + 2, 1)])
        col += 1
        
        # P6: (A11 - A00) * (B0k + B0,k+1)
        add_term(col, 
                 [(3, 1), (0, -1)], 
                 [(k, 1), (k + 1, 1)], 
                 [(2*k + 3, 1)])
        col += 1
        
        # P7: (A01 - A11) * (B1k + B1,k+1)
        add_term(col, 
                 [(1, 1), (3, -1)], 
                 [(8 + k, 1), (8 + k + 1, 1)], 
                 [(2*k, 1), (2*k + 3, 1)])
        col += 1
        
        # M8: A02 * B2k
        add_term(col, 
                 [(4, 1)], 
                 [(16 + k, 1)], 
                 [(2*k, 1)])
        col += 1
        
        # M9: A02 * B2,k+1
        add_term(col, 
                 [(4, 1)], 
                 [(16 + k + 1, 1)], 
                 [(2*k + 2, 1)])
        col += 1
        
        # M10: A12 * B2k
        add_term(col, 
                 [(5, 1)], 
                 [(16 + k, 1)], 
                 [(2*k + 1, 1)])
        col += 1
        
        # M11: A12 * B2,k+1
        add_term(col, 
                 [(5, 1)], 
                 [(16 + k + 1, 1)], 
                 [(2*k + 3, 1)])
        col += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
