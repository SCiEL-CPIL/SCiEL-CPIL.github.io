import numpy as np

DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS
    rank = 48
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a rank-1 tensor
    def add_tensor(u_idx, v_idx, w_idx, val_u, val_v, val_w):
        nonlocal col
        U[u_idx][col] = val_u
        V[v_idx][col] = val_v
        W[w_idx][col] = val_w
        col += 1

    # Strassen 2x2 by 2x2
    # A is 2x2, B is 2x2
    # u_idx for A: i*m + j -> i*3 + j? No, A is 2x2, so i in 0,1, j in 0,1.
    # But in our global A, j is 0,1,2.
    # We are using specific sub-blocks.
    
    def add_strassen_2x2(a_offset_i, a_offset_j, b_offset_j, b_offset_k, c_offset_i, c_offset_k):
        # a_offset_i: row offset in A (0 or 1)
        # a_offset_j: col offset in A (0 or 1)
        # b_offset_j: row offset in B (0 or 1)
        # b_offset_k: col offset in B (0 or 1)
        # c_offset_i: row offset in C (0 or 1)
        # c_offset_k: col offset in C (0 or 1)
        
        # P1 = A00 (B01 - B11)
        # A00 is at (a_offset_i, a_offset_j)
        # B01 is at (b_offset_j, b_offset_k+1)
        # B11 is at (b_offset_j+1, b_offset_k+1)
        
        # Tensors are (u, v, w)
        # u index: i*m + j
        # v index: j*p + k
        # w index: k*n + i
        
        # P1
        add_tensor(a_offset_i*m + a_offset_j, b_offset_j*p + (b_offset_k+1), (b_offset_k+1)*n + c_offset_i, 1, 1, 1)
        add_tensor(a_offset_i*m + a_offset_j, (b_offset_j+1)*p + (b_offset_k+1), (b_offset_k+1)*n + c_offset_i, 1, -1, 1)
        
        # P2 = (A00 + A01) B11
        add_tensor(a_offset_i*m + a_offset_j, (b_offset_j+1)*p + (b_offset_k+1), (b_offset_k+1)*n + c_offset_i, 1, 1, 1)
        add_tensor(a_offset_i*m + a_offset_j+1, (b_offset_j+1)*p + (b_offset_k+1), (b_offset_k+1)*n + c_offset_i, 1, 1, 1)
        
        # P3 = (A10 + A11) B00
        add_tensor((a_offset_i+1)*m + a_offset_j, b_offset_j*p + b_offset_k, b_offset_k*n + c_offset_i, 1, 1, 1)
        add_tensor((a_offset_i+1)*m + a_offset_j+1, b_offset_j*p + b_offset_k, b_offset_k*n + c_offset_i, 1, 1, 1)
        
        # P4 = A11 (B10 - B00)
        add_tensor((a_offset_i+1)*m + a_offset_j+1, (b_offset_j+1)*p + b_offset_k, b_offset_k*n + c_offset_i, 1, 1, 1)
        add_tensor((a_offset_i+1)*m + a_offset_j+1, b_offset_j*p + b_offset_k, b_offset_k*n + c_offset_i, 1, -1, 1)
        
        # P5 = (A00 + A11) (B00 + B11)
        add_tensor(a_offset_i*m + a_offset_j, b_offset_j*p + b_offset_k, b_offset_k*n + c_offset_i, 1, 1, 1)
        add_tensor(a_offset_i*m + a_offset_j, (b_offset_j+1)*p + (b_offset_k+1), (b_offset_k+1)*n + c_offset_i, 1, 1, 1)
        add_tensor((a_offset_i+1)*m + a_offset_j+1, b_offset_j*p + b_offset_k, b_offset_k*n + c_offset_i, 1, 1, 1)
        add_tensor((a_offset_i+1)*m + a_offset_j+1, (b_offset_j+1)*p + (b_offset_k+1), (b_offset_k+1)*n + c_offset_i, 1, 1, 1)
        
        # P6 = (A01 - A11) (B10 + B11)
        add_tensor(a_offset_i*m + a_offset_j+1, (b_offset_j+1)*p + b_offset_k, b_offset_k*n + c_offset_i, 1, 1, 1)
        add_tensor(a_offset_i*m + a_offset_j+1, (b_offset_j+1)*p + (b_offset_k+1), (b_offset_k+1)*n + c_offset_i, 1, 1, 1)
        add_tensor((a_offset_i+1)*m + a_offset_j+1, (b_offset_j+1)*p + b_offset_k, b_offset_k*n + c_offset_i, -1, 1, 1)
        add_tensor((a_offset_i+1)*m + a_offset_j+1, (b_offset_j+1)*p + (b_offset_k+1), (b_offset_k+1)*n + c_offset_i, -1, 1, 1)
        
        # P7 = (A00 - A10) (B00 + B01)
        add_tensor(a_offset_i*m + a_offset_j, b_offset_j*p + b_offset_k, b_offset_k*n + c_offset_i, 1, 1, 1)
        add_tensor(a_offset_i*m + a_offset_j, b_offset_j*p + (b_offset_k+1), (b_offset_k+1)*n + c_offset_i, 1, 1, 1)
        add_tensor((a_offset_i+1)*m + a_offset_j, b_offset_j*p + b_offset_k, b_offset_k*n + c_offset_i, -1, 1, 1)
        add_tensor((a_offset_i+1)*m + a_offset_j, b_offset_j*p + (b_offset_k+1), (b_offset_k+1)*n + c_offset_i, -1, 1, 1)

    # Schoolbook 2x1 by 1xK
    # A is 2x1, B is 1xK
    def add_schoolbook_2x1_1xk(a_offset_i, a_offset_j, b_offset_j, b_offset_k_start, k_count, c_offset_i, c_offset_k_start):
        for i in range(2):
            for k in range(k_count):
                u_idx = (a_offset_i + i)*m + a_offset_j
                v_idx = b_offset_j*p + (b_offset_k_start + k)
                w_idx = (c_offset_k_start + k)*n + c_offset_i
                add_tensor(u_idx, v_idx, w_idx, 1, 1, 1)

    # Loop over the 2 groups of 5 columns
    for g in range(2):
        k_start = g * 5
        # Group has 5 columns: 3 for M_233, 2 for M_232
        
        # M_233 part (cols k_start to k_start+2)
        # A cols 0,1 (j=0,1), B rows 0,1 (j=0,1)
        # A col 2 (j=2), B row 2 (j=2)
        
        # Strassen 2x2 by 2x3? No, Strassen is 2x2 by 2x2.
        # We need 2x2 by 2x3.
        # Split 2x3 into 2x2 and 2x1.
        # M_222 (rank 7) + M_221 (rank 4).
        
        # M_222: A(2x2), B(2x2) cols 0,1 of the 3-col block.
        add_strassen_2x2(0, 0, 0, k_start, 0, k_start)
        
        # M_221: A(2x2), B(2x1) col 2 of the 3-col block.
        # Schoolbook 2x2 by 2x1.
        # Rank 4.
        for i in range(2):
            for j in range(2):
                u_idx = i*m + j
                v_idx = j*p + (k_start + 2)
                w_idx = (k_start + 2)*n + i
                add_tensor(u_idx, v_idx, w_idx, 1, 1, 1)
                
        # M_213: A(2x1) col 2, B(1x3) row 2.
        # Rank 6.
        add_schoolbook_2x1_1xk(0, 2, 2, k_start, 3, 0, k_start)
        
        # M_232 part (cols k_start+3 to k_start+4)
        # A cols 0,1 (j=0,1), B rows 0,1 (j=0,1)
        # A col 2 (j=2), B row 2 (j=2)
        
        # M_222: A(2x2), B(2x2) cols 0,1 of the 2-col block.
        add_strassen_2x2(0, 0, 0, k_start+3, 0, k_start+3)
        
        # M_212: A(2x1) col 2, B(1x2) row 2.
        # Rank 4.
        add_schoolbook_2x1_1xk(0, 2, 2, k_start+3, 2, 0, k_start+3)

    return {"U": U, "V": V, "W": W}
