import time

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 4, 4
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen(u_indices, v_indices, w_indices):
        nonlocal col
        # u_indices: [a00, a01, a10, a11]
        # v_indices: [b00, b01, b10, b11]
        # w_indices: [c00, c01, c10, c11]
        
        # M1 = (a00 + a11)(b00 + b11)
        # c00 += M1, c11 += M1
        U[u_indices[0]][col] += 1
        U[u_indices[3]][col] += 1
        V[v_indices[0]][col] += 1
        V[v_indices[3]][col] += 1
        W[w_indices[0]][col] += 1
        W[w_indices[3]][col] += 1
        col += 1
        
        # M2 = (a10 + a11)b00
        # c10 += M2, c11 -= M2
        U[u_indices[2]][col] += 1
        U[u_indices[3]][col] += 1
        V[v_indices[0]][col] += 1
        W[w_indices[2]][col] += 1
        W[w_indices[3]][col] -= 1
        col += 1
        
        # M3 = a00(b10 - b11)
        # c01 += M3, c11 += M3
        U[u_indices[0]][col] += 1
        V[v_indices[2]][col] += 1
        V[v_indices[3]][col] -= 1
        W[w_indices[1]][col] += 1
        W[w_indices[3]][col] += 1
        col += 1
        
        # M4 = a11(b01 - b00)
        # c00 -= M4, c01 += M4
        U[u_indices[3]][col] += 1
        V[v_indices[1]][col] += 1
        V[v_indices[0]][col] -= 1
        W[w_indices[0]][col] -= 1
        W[w_indices[1]][col] += 1
        col += 1
        
        # M5 = (a00 + a01)b11
        # c00 += M5, c11 += M5
        U[u_indices[0]][col] += 1
        U[u_indices[1]][col] += 1
        V[v_indices[3]][col] += 1
        W[w_indices[0]][col] += 1
        W[w_indices[3]][col] += 1
        col += 1
        
        # M6 = (a10 - a00)(b00 + b01)
        # c11 += M6
        U[u_indices[2]][col] += 1
        U[u_indices[0]][col] -= 1
        V[v_indices[0]][col] += 1
        V[v_indices[1]][col] += 1
        W[w_indices[3]][col] += 1
        col += 1
        
        # M7 = (a01 - a11)(b10 + b11)
        # c00 -= M7, c10 += M7
        U[u_indices[1]][col] += 1
        U[u_indices[3]][col] -= 1
        V[v_indices[2]][col] += 1
        V[v_indices[3]][col] += 1
        W[w_indices[0]][col] -= 1
        W[w_indices[2]][col] += 1
        col += 1

    def add_schoolbook_1x2x2(u_indices, v_indices, w_indices):
        nonlocal col
        # u_indices: [a0, a1]
        # v_indices: [b00, b01, b10, b11]
        # w_indices: [c0, c1]
        
        # c0 = a0*b00 + a1*b10
        # c1 = a0*b01 + a1*b11
        
        # Term 1: a0*b00 -> c0
        U[u_indices[0]][col] += 1
        V[v_indices[0]][col] += 1
        W[w_indices[0]][col] += 1
        col += 1
        
        # Term 2: a1*b10 -> c0
        U[u_indices[1]][col] += 1
        V[v_indices[2]][col] += 1
        W[w_indices[0]][col] += 1
        col += 1
        
        # Term 3: a0*b01 -> c1
        U[u_indices[0]][col] += 1
        V[v_indices[1]][col] += 1
        W[w_indices[1]][col] += 1
        col += 1
        
        # Term 4: a1*b11 -> c1
        U[u_indices[1]][col] += 1
        V[v_indices[3]][col] += 1
        W[w_indices[1]][col] += 1
        col += 1

    # Part 0: j in {0, 1}
    # A cols 0,1. B rows 0,1.
    
    # Strassen 1: A rows 0,1; B cols 0,1 -> C rows 0,1; cols 0,1
    u_idx = [0*4+0, 0*4+1, 1*4+0, 1*4+1] # [0, 1, 4, 5]
    v_idx = [0*4+0, 0*4+1, 1*4+0, 1*4+1] # [0, 1, 4, 5]
    w_idx = [0*3+0, 1*3+0, 0*3+1, 1*3+1] # [0, 3, 1, 4]
    add_strassen(u_idx, v_idx, w_idx)
    
    # Strassen 2: A rows 0,1; B cols 2,3 -> C rows 0,1; cols 2,3
    u_idx = [0*4+0, 0*4+1, 1*4+0, 1*4+1] # [0, 1, 4, 5]
    v_idx = [0*4+2, 0*4+3, 1*4+2, 1*4+3] # [2, 3, 6, 7]
    w_idx = [2*3+0, 3*3+0, 2*3+1, 3*3+1] # [6, 9, 7, 10]
    add_strassen(u_idx, v_idx, w_idx)
    
    # Schoolbook 3: A row 2; B cols 0,1 -> C row 2; cols 0,1
    u_idx = [2*4+0, 2*4+1] # [8, 9]
    v_idx = [0*4+0, 0*4+1, 1*4+0, 1*4+1] # [0, 1, 4, 5]
    w_idx = [0*3+2, 1*3+2] # [2, 5]
    add_schoolbook_1x2x2(u_idx, v_idx, w_idx)
    
    # Schoolbook 4: A row 2; B cols 2,3 -> C row 2; cols 2,3
    u_idx = [2*4+0, 2*4+1] # [8, 9]
    v_idx = [0*4+2, 0*4+3, 1*4+2, 1*4+3] # [2, 3, 6, 7]
    w_idx = [2*3+2, 3*3+2] # [8, 11]
    add_schoolbook_1x2x2(u_idx, v_idx, w_idx)
    
    # Part 1: j in {2, 3}
    # A cols 2,3. B rows 2,3.
    
    # Strassen 5: A rows 0,1; B cols 0,1 -> C rows 0,1; cols 0,1
    u_idx = [0*4+2, 0*4+3, 1*4+2, 1*4+3] # [2, 3, 6, 7]
    v_idx = [2*4+0, 2*4+1, 3*4+0, 3*4+1] # [8, 9, 12, 13]
    w_idx = [0*3+0, 1*3+0, 0*3+1, 1*3+1] # [0, 3, 1, 4]
    add_strassen(u_idx, v_idx, w_idx)
    
    # Strassen 6: A rows 0,1; B cols 2,3 -> C rows 0,1; cols 2,3
    u_idx = [0*4+2, 0*4+3, 1*4+2, 1*4+3] # [2, 3, 6, 7]
    v_idx = [2*4+2, 2*4+3, 3*4+2, 3*4+3] # [10, 11, 14, 15]
    w_idx = [2*3+0, 3*3+0, 2*3+1, 3*3+1] # [6, 9, 7, 10]
    add_strassen(u_idx, v_idx, w_idx)
    
    # Schoolbook 7: A row 2; B cols 0,1 -> C row 2; cols 0,1
    u_idx = [2*4+2, 2*4+3] # [10, 11]
    v_idx = [2*4+0, 2*4+1, 3*4+0, 3*4+1] # [8, 9, 12, 13]
    w_idx = [0*3+2, 1*3+2] # [2, 5]
    add_schoolbook_1x2x2(u_idx, v_idx, w_idx)
    
    # Schoolbook 8: A row 2; B cols 2,3 -> C row 2; cols 2,3
    u_idx = [2*4+2, 2*4+3] # [10, 11]
    v_idx = [2*4+2, 2*4+3, 3*4+2, 3*4+3] # [10, 11, 14, 15]
    w_idx = [2*3+2, 3*3+2] # [8, 11]
    add_schoolbook_1x2x2(u_idx, v_idx, w_idx)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
