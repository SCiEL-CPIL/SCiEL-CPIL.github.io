import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def construct():
    start_time = time.time()
    DEADLINE = 140.0
    
    n, m, p = DIMENSIONS  # 4, 5, 5
    
    rank = 85
    # U: (n*m, rank) = (20, 85)
    # V: (m*p, rank) = (25, 85)
    # W: (p*n, rank) = (20, 85)
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Helper to set a block of U, V, W for a term
    # u_idx: list of (row, val) for U
    # v_idx: list of (row, val) for V
    # w_idx: list of (row, val) for W
    
    def add_term(col, u_data, v_data, w_data):
        for idx, val in u_data:
            U[idx, col] += val
        for idx, val in v_data:
            V[idx, col] += val
        for idx, val in w_data:
            W[idx, col] += val

    col = 0
    
    # Part 1: 4x4x4 using Strassen (Rank 49)
    # A[0:4, 0:4] * B[0:4, 0:4] -> C[0:4, 0:4]
    # Indices: A(i,j) i<4,j<4. B(j,k) j<4,k<4. C(i,k) i<4,k<4.
    # U row: i*5+j. V row: j*5+k. W row: k*4+i.
    
    # Strassen 2x2x2 factors
    # s_u, s_v, s_w are 4x7
    s_u = np.zeros((4, 7))
    s_v = np.zeros((4, 7))
    s_w = np.zeros((4, 7))
    
    # Term 0: (a00+a11)(b00+b11)
    s_u[0,0]=1; s_u[3,0]=1
    s_v[0,0]=1; s_v[3,0]=1
    s_w[0,0]=1; s_w[3,0]=1
    
    # Term 1: (a10+a11)b00
    s_u[2,1]=1; s_u[3,1]=1
    s_v[0,1]=1
    s_w[2,1]=1; s_w[3,1]=-1
    
    # Term 2: a00(b01-b11)
    s_u[0,2]=1
    s_v[1,2]=1; s_v[3,2]=-1
    s_w[1,2]=1; s_w[3,2]=1
    
    # Term 3: a11(b10-b00)
    s_u[3,3]=1
    s_v[2,3]=1; s_v[0,3]=-1
    s_w[0,3]=1; s_w[2,3]=1
    
    # Term 4: (a00+a01)b11
    s_u[0,4]=1; s_u[1,4]=1
    s_v[3,4]=1
    s_w[0,4]=-1; s_w[1,4]=1
    
    # Term 5: (a10-a00)(b00+b01)
    s_u[0,5]=-1; s_u[2,5]=1
    s_v[0,5]=1; s_v[1,5]=1
    s_w[3,5]=1
    
    # Term 6: (a01-a11)(b10+b11)
    s_u[1,6]=1; s_u[3,6]=-1
    s_v[2,6]=1; s_v[3,6]=1
    s_w[0,6]=1
    
    # Block Strassen for 4x4x4
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    # Each block is 2x2.
    
    block_terms = [
        # M1 = (A00+A11)(B00+B11)
        # A blocks: (0,0), (1,1)
        # B blocks: (0,0), (1,1)
        # C blocks: (0,0), (1,1)
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # M2 = (A10+A11)B00
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        # M3 = A00(B01-B11)
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        # M4 = A11(B10-B00)
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        # M5 = (A00+A01)B11
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        # M6 = (A10-A00)(B00+B01)
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        # M7 = (A01-A11)(B10+B11)
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]
    
    for a_coeffs, b_coeffs, c_coeffs in block_terms:
        for inner_k in range(7):
            u_data = []
            v_data = []
            w_data = []
            
            # U part
            for (a_br, a_bc), a_coeff in a_coeffs.items():
                for lr in range(2):
                    for lc in range(2):
                        local_idx = lr * 2 + lc
                        u_val = s_u[local_idx, inner_k]
                        if u_val != 0:
                            global_i = 2 * a_br + lr
                            global_j = 2 * a_bc + lc
                            global_u_row = global_i * 5 + global_j
                            u_data.append((global_u_row, a_coeff * u_val))
            
            # V part
            for (b_br, b_bc), b_coeff in b_coeffs.items():
                for lr in range(2):
                    for lc in range(2):
                        local_idx = lr * 2 + lc
                        v_val = s_v[local_idx, inner_k]
                        if v_val != 0:
                            global_j = 2 * b_br + lr
                            global_k = 2 * b_bc + lc
                            global_v_row = global_j * 5 + global_k
                            v_data.append((global_v_row, b_coeff * v_val))
            
            # W part
            for (c_br, c_bc), c_coeff in c_coeffs.items():
                for lr in range(2):
                    for lc in range(2):
                        local_idx = lr * 2 + lc
                        w_val = s_w[local_idx, inner_k]
                        if w_val != 0:
                            global_i = 2 * c_br + lr
                            global_k = 2 * c_bc + lc
                            global_w_row = global_k * 4 + global_i
                            w_data.append((global_w_row, c_coeff * w_val))
            
            add_term(col, u_data, v_data, w_data)
            col += 1
            
    assert col == 49, f"Part 1 col error: {col}"
    
    # Part 2: 4x4x1: A[:,0:4] * B[0:4,4] -> C[:,4]
    # A(i,j) i<4,j<4. B(j,4) j<4. C(i,4) i<4.
    for i in range(4):
        for j in range(4):
            u_data = [(i*5 + j, 1)]
            v_data = [(j*5 + 4, 1)]
            w_data = [(4*4 + i, 1)]
            add_term(col, u_data, v_data, w_data)
            col += 1
            
    assert col == 65, f"Part 2 col error: {col}"
    
    # Part 3: 4x1x4: A[:,4] * B[4,0:4] -> C[:,0:4]
    # A(i,4) i<4. B(4,k) k<4. C(i,k) i<4,k<4.
    for i in range(4):
        for k in range(4):
            u_data = [(i*5 + 4, 1)]
            v_data = [(4*5 + k, 1)]
            w_data = [(k*4 + i, 1)]
            add_term(col, u_data, v_data, w_data)
            col += 1
            
    assert col == 81, f"Part 3 col error: {col}"
    
    # Part 4: 4x1x1: A[:,4] * B[4,4] -> C[:,4]
    # A(i,4) i<4. B(4,4). C(i,4) i<4.
    for i in range(4):
        u_data = [(i*5 + 4, 1)]
        v_data = [(4*5 + 4, 1)]
        w_data = [(4*4 + i, 1)]
        add_term(col, u_data, v_data, w_data)
        col += 1
        
    assert col == 85, f"Part 4 col error: {col}"
    
    if time.time() > DEADLINE:
        return {"U": U, "V": V, "W": W}
        
    return {"U": U, "V": V, "W": W}
