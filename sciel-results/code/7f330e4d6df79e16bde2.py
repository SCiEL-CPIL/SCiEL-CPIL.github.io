import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    # Total rank
    # 4x4x4 part: 49
    # 4x4x1 part: 16
    # 4x1x4 part: 16
    # 4x1x1 part: 4
    rank = 49 + 16 + 16 + 4
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to get 2x2x2 Strassen factors
    # Returns U, V, W of shape 4x7
    def get_strassen_2x2_factors():
        # U indices: 0:00, 1:01, 2:10, 3:11
        # V indices: 0:00, 1:01, 2:10, 3:11
        # W indices: 0:00, 1:01, 2:10, 3:11 (using k*n+i, n=2)
        
        u = [[0]*7 for _ in range(4)]
        v = [[0]*7 for _ in range(4)]
        w = [[0]*7 for _ in range(4)]
        
        # Term 0: (a00+a11)(b00+b11)
        # U: 1,0,0,1
        u[0][0]=1; u[3][0]=1
        # V: 1,0,0,1
        v[0][0]=1; v[3][0]=1
        # W: c00+1, c11+1 -> idx 0, 3
        w[0][0]=1; w[3][0]=1
        
        # Term 1: (a10+a11)b00
        # U: 0,0,1,1
        u[2][1]=1; u[3][1]=1
        # V: 1,0,0,0
        v[0][1]=1
        # W: c10+1, c11-1 -> idx 2, 3
        w[2][1]=1; w[3][1]=-1
        
        # Term 2: a00(b01-b11)
        # U: 1,0,0,0
        u[0][2]=1
        # V: 0,1,0,-1
        v[1][2]=1; v[3][2]=-1
        # W: c01+1, c11+1 -> idx 1, 3
        w[1][2]=1; w[3][2]=1
        
        # Term 3: a11(b10-b00)
        # U: 0,0,0,1
        u[3][3]=1
        # V: 0,0,1,-1
        v[2][3]=1; v[0][3]=-1
        # W: c00+1, c10+1 -> idx 0, 2
        w[0][3]=1; w[2][3]=1
        
        # Term 4: (a00+a01)b11
        # U: 1,1,0,0
        u[0][4]=1; u[1][4]=1
        # V: 0,0,0,1
        v[3][4]=1
        # W: c00-1, c01+1 -> idx 0, 1
        w[0][4]=-1; w[1][4]=1
        
        # Term 5: (a10-a00)(b00+b01)
        # U: -1,0,1,0
        u[0][5]=-1; u[2][5]=1
        # V: 1,1,0,0
        v[0][5]=1; v[1][5]=1
        # W: c11+1 -> idx 3
        w[3][5]=1
        
        # Term 6: (a01-a11)(b10+b11)
        # U: 0,1,0,-1
        u[1][6]=1; u[3][6]=-1
        # V: 0,0,1,1
        v[2][6]=1; v[3][6]=1
        # W: c00+1 -> idx 0
        w[0][6]=1
        
        return u, v, w

    s_u, s_v, s_w = get_strassen_2x2_factors()
    
    col = 0
    # 4x4x4 part
    # A blocks: br, bc in {0,1}
    # B blocks: br, bc in {0,1}
    # C blocks: br, bc in {0,1}
    
    def add_4x4_term(a_block_coeffs, b_block_coeffs, c_block_coeffs, start_col):
        # a_block_coeffs: dict (br,bc) -> coeff
        # b_block_coeffs: dict (br,bc) -> coeff
        # c_block_coeffs: dict (br,bc) -> coeff
        # This term is sum of (sum a_b A_b) * (sum b_b B_b)
        # Let X = sum a_b A_b, Y = sum b_b B_b.
        # We decompose X*Y using 2x2 Strassen.
        
        for inner_k in range(7):
            # Get u_vec, v_vec, w_vec for this inner term
            u_vec = [s_u[r][inner_k] for r in range(4)] # 4 entries
            v_vec = [s_v[r][inner_k] for r in range(4)]
            w_vec = [s_w[r][inner_k] for r in range(4)]
            
            col_idx = start_col + inner_k
            
            # For each of the 4 local entries in u_vec (00, 01, 10, 11)
            # We need to distribute the coefficient to the global A entries.
            
            # u_vec[0] is coeff for x00.
            # x00 = sum a_b * A_b[0,0]
            # So global U row (i, j) corresponding to A_b[0,0] gets u_vec[0] * a_b.
            
            # Mapping local (r,c) to global (i,j) for a block (br,bc):
            # i = 2*br + r
            # j = 2*bc + c
            
            # We iterate over all A blocks in a_block_coeffs
            for (a_br, a_bc), a_coeff in a_block_coeffs.items():
                for lr in range(2):
                    for lc in range(2):
                        local_idx = lr * 2 + lc # 0:00, 1:01, 2:10, 3:11
                        u_val = u_vec[local_idx]
                        if u_val != 0:
                            global_i = 2 * a_br + lr
                            global_j = 2 * a_bc + lc
                            global_u_row = global_i * 5 + global_j
                            U[global_u_row][col_idx] += a_coeff * u_val
            
            # Similarly for V
            for (b_br, b_bc), b_coeff in b_block_coeffs.items():
                for lr in range(2):
                    for lc in range(2):
                        local_idx = lr * 2 + lc
                        v_val = v_vec[local_idx]
                        if v_val != 0:
                            global_j = 2 * b_br + lr
                            global_k = 2 * b_bc + lc
                            global_v_row = global_j * 5 + global_k
                            V[global_v_row][col_idx] += b_coeff * v_val
            
            # Similarly for W
            # w_vec[r] is coeff for local c_rr of the 2x2 product.
            # This 2x2 product is added to C blocks specified in c_block_coeffs.
            # If C block (c_br, c_bc) has coeff c_coeff, then local c_rr adds to global C at
            # i = 2*c_br + lr, k = 2*c_bc + lc.
            
            for (c_br, c_bc), c_coeff in c_block_coeffs.items():
                for lr in range(2):
                    for lc in range(2):
                        local_idx = lr * 2 + lc
                        w_val = w_vec[local_idx]
                        if w_val != 0:
                            global_k = 2 * c_bc + lc
                            global_i = 2 * c_br + lr
                            global_w_row = global_k * 4 + global_i
                            W[global_w_row][col_idx] += c_coeff * w_val

    # Define the 7 Strassen 4x4 terms
    terms = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}), # M1
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}), # M2
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}), # M3
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}), # M4
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}), # M5
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}), # M6
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}), # M7
    ]
    
    for t_idx, (a_coeff, b_coeff, c_coeff) in enumerate(terms):
        start_col = t_idx * 7
        add_4x4_term(a_coeff, b_coeff, c_coeff, start_col)
        
    col = 49
    
    # 4x4x1 part: A_L (4x4) * B_LR (4x1) -> C (4x1)
    # Rank 16
    for i in range(4):
        for j in range(4):
            U[i*5 + j][col] = 1
            V[j*5 + 4][col] = 1 # j in 0..3, k=4
            W[4*4 + i][col] = 1 # k=4, i in 0..3
            col += 1
            
    # 4x1x4 part: A_R (4x1) * B_RL (1x4) -> C (4x4)
    # Rank 16
    for i in range(4):
        for k in range(4):
            U[i*5 + 4][col] = 1 # j=4
            V[4*5 + k][col] = 1 # j=4, k in 0..3
            W[k*4 + i][col] = 1
            col += 1
            
    # 4x1x1 part: A_R (4x1) * B_RR (1x1) -> C (4x1)
    # Rank 4
    for i in range(4):
        U[i*5 + 4][col] = 1
        V[4*5 + 4][col] = 1
        W[4*4 + i][col] = 1
        col += 1
        
    # Deadline guard: if we're running out of time, return what we have
    if time.time() > DEADLINE:
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
