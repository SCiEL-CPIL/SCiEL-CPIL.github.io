import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    DEADLINE = 140.0
    
    n, m, p = DIMENSIONS  # 4, 5, 5
    
    # Rank 85 decomposition:
    # Part 1: 4x4x4 using Strassen: rank 49
    # Part 2: 4x4x1 (A[:,0:4] * B[0:4,4]): rank 16
    # Part 3: 4x1x4 (A[:,4] * B[4,0:4]): rank 16
    # Part 4: 4x1x1 (A[:,4] * B[4,4]): rank 4
    # Total: 85
    
    rank = 85
    U = [[0] * rank for _ in range(n * m)]  # 20 x 85
    V = [[0] * rank for _ in range(m * p)]  # 25 x 85
    W = [[0] * rank for _ in range(p * n)]  # 20 x 85
    
    # Strassen 2x2x2 factors
    def get_strassen_2x2():
        """Return U, V, W factors for 2x2x2 Strassen, each 4x7."""
        u = [[0]*7 for _ in range(4)]
        v = [[0]*7 for _ in range(4)]
        w = [[0]*7 for _ in range(4)]
        
        # Term 0: (a00+a11)(b00+b11)
        u[0][0]=1; u[3][0]=1
        v[0][0]=1; v[3][0]=1
        w[0][0]=1; w[3][0]=1
        
        # Term 1: (a10+a11)b00
        u[2][1]=1; u[3][1]=1
        v[0][1]=1
        w[2][1]=1; w[3][1]=-1
        
        # Term 2: a00(b01-b11)
        u[0][2]=1
        v[1][2]=1; v[3][2]=-1
        w[1][2]=1; w[3][2]=1
        
        # Term 3: a11(b10-b00)
        u[3][3]=1
        v[2][3]=1; v[0][3]=-1
        w[0][3]=1; w[2][3]=1
        
        # Term 4: (a00+a01)b11
        u[0][4]=1; u[1][4]=1
        v[3][4]=1
        w[0][4]=-1; w[1][4]=1
        
        # Term 5: (a10-a00)(b00+b01)
        u[0][5]=-1; u[2][5]=1
        v[0][5]=1; v[1][5]=1
        w[3][5]=1
        
        # Term 6: (a01-a11)(b10+b11)
        u[1][6]=1; u[3][6]=-1
        v[2][6]=1; v[3][6]=1
        w[0][6]=1
        
        return u, v, w
    
    s_u, s_v, s_w = get_strassen_2x2()
    
    # Part 1: 4x4x4 part using Strassen at block level
    # A = [[A00, A01], [A10, A11]] where each Ai is 2x2
    # B = [[B00, B01], [B10, B11]] where each Bi is 2x2
    # C = [[C00, C01], [C10, C11]] where each Ci is 2x2
    #
    # Block-level Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    block_terms = [
        # M1 = (A00+A11)(B00+B11), contributes to C00 and C11
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # M2 = (A10+A11)B00, contributes to C10 and C11
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        # M3 = A00(B01-B11), contributes to C01 and C11
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        # M4 = A11(B10-B00), contributes to C00 and C10
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        # M5 = (A00+A01)B11, contributes to C00 and C01
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        # M6 = (A10-A00)(B00+B01), contributes to C11
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        # M7 = (A01-A11)(B10+B11), contributes to C00
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]
    
    col = 0
    for a_coeffs, b_coeffs, c_coeffs in block_terms:
        for inner_k in range(7):
            col_idx = col
            u_vec = [s_u[r][inner_k] for r in range(4)]
            v_vec = [s_v[r][inner_k] for r in range(4)]
            w_vec = [s_w[r][inner_k] for r in range(4)]
            
            # U part
            for (a_br, a_bc), a_coeff in a_coeffs.items():
                for lr in range(2):
                    for lc in range(2):
                        local_idx = lr * 2 + lc
                        u_val = u_vec[local_idx]
                        if u_val != 0:
                            global_i = 2 * a_br + lr
                            global_j = 2 * a_bc + lc
                            global_u_row = global_i * 5 + global_j
                            U[global_u_row][col_idx] += a_coeff * u_val
            
            # V part
            for (b_br, b_bc), b_coeff in b_coeffs.items():
                for lr in range(2):
                    for lc in range(2):
                        local_idx = lr * 2 + lc
                        v_val = v_vec[local_idx]
                        if v_val != 0:
                            global_j = 2 * b_br + lr
                            global_k = 2 * b_bc + lc
                            global_v_row = global_j * 5 + global_k
                            V[global_v_row][col_idx] += b_coeff * v_val
            
            # W part
            for (c_br, c_bc), c_coeff in c_coeffs.items():
                for lr in range(2):
                    for lc in range(2):
                        local_idx = lr * 2 + lc
                        w_val = w_vec[local_idx]
                        if w_val != 0:
                            global_i = 2 * c_br + lr
                            global_k = 2 * c_bc + lc
                            global_w_row = global_k * 4 + global_i
                            W[global_w_row][col_idx] += c_coeff * w_val
            
            col += 1
    
    assert col == 49, f"Expected col=49, got {col}"
    
    # Part 2: 4x4x1 part: A[:,0:4] * B[0:4,4] -> C[:,4]
    for i in range(4):
        for j in range(4):
            U[i*5 + j][col] = 1
            V[j*5 + 4][col] = 1
            W[4*4 + i][col] = 1
            col += 1
    
    assert col == 65, f"Expected col=65, got {col}"
    
    # Part 3: 4x1x4 part: A[:,4] * B[4,0:4] -> C[:,0:4]
    for i in range(4):
        for k in range(4):
            U[i*5 + 4][col] = 1
            V[4*5 + k][col] = 1
            W[k*4 + i][col] = 1
            col += 1
    
    assert col == 81, f"Expected col=81, got {col}"
    
    # Part 4: 4x1x1 part: A[:,4] * B[4,4] -> C[:,4]
    for i in range(4):
        U[i*5 + 4][col] = 1
        V[4*5 + 4][col] = 1
        W[4*4 + i][col] = 1
        col += 1
    
    assert col == 85, f"Expected col=85, got {col}"
    
    if time.time() > DEADLINE:
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
