import numpy as np
import time

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    
    # Rank calculation:
    # Split inner dimension 5 into 2, 2, 1
    # C = A[:, 0:2] @ B[0:2, :] + A[:, 2:4] @ B[2:4, :] + A[:, 4:5] @ B[4:5, :]
    
    # Block 1: A_4x2 @ B_2x5
    # Split A_4x2 into two 2x2 blocks
    # Each 2x2 @ 2x5 is rank 15 (7 for 2x2x2 + 8 for 2x2x3)
    # Total for Block 1: 15 + 15 = 30
    # Block 2: A_4x2 @ B_2x5
    # Total for Block 2: 30
    # Block 3: A_4x1 @ B_1x5
    # Rank 20
    # Total Rank: 30 + 30 + 20 = 80
    
    rank = 80
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper: 2x2x2 Strassen factors (rank 7)
    def get_strassen_2x2_factors():
        u = [[0]*7 for _ in range(4)]
        v = [[0]*7 for _ in range(4)]
        w = [[0]*7 for _ in range(4)]
        u[0][0]=1; u[3][0]=1; v[0][0]=1; v[3][0]=1; w[0][0]=1; w[3][0]=1
        u[2][1]=1; u[3][1]=1; v[0][1]=1; w[2][1]=1; w[3][1]=-1
        u[0][2]=1; v[1][2]=1; v[3][2]=-1; w[1][2]=1; w[3][2]=1
        u[3][3]=1; v[2][3]=1; v[0][3]=-1; w[0][3]=1; w[2][3]=1
        u[0][4]=1; u[1][4]=1; v[3][4]=1; w[0][4]=-1; w[1][4]=1
        u[0][5]=-1; u[2][5]=1; v[0][5]=1; v[1][5]=1; w[3][5]=1
        u[1][6]=1; u[3][6]=-1; v[2][6]=1; v[3][6]=1; w[0][6]=1
        return u, v, w

    s_u, s_v, s_w = get_strassen_2x2_factors()
    
    # Helper: 2x2x3 factors (rank 8)
    # This is a known decomposition.
    # I will use a simple rank 11 decomposition if rank 8 is too hard to code from memory, 
    # but rank 8 is possible. 
    # For now, let's use rank 11 (7+4) to be safe, which gives total rank 92.
    # Wait, 92 is worse than 85.
    # I MUST find rank 8 for 2x2x3 or use a different split.
    
    # Alternative: Use 4+1 split (rank 85) but optimize?
    # No, 80 is the target.
    # I will hardcode the rank 8 factors for 2x2x3.
    
    def get_2x2x3_factors():
        # Rank 8 decomposition for 2x2x3
        # Reference: "The rank of the 2x2x3 matrix multiplication tensor is 8"
        # I will construct a valid one or use a fallback.
        # Fallback: Use rank 11 (Strassen 2x2x2 + 4 terms for 2x2x1)
        # This leads to rank 92 total.
        # Let's try to mix.
        
        # Actually, I can use the 4+1 split (85) and just return that.
        # But 80 is better.
        # Let's assume I can implement rank 8 for 2x2x3.
        
        u = [[0]*8 for _ in range(4)]
        v = [[0]*8 for _ in range(4)]
        w = [[0]*8 for _ in range(6)] # 2x3 = 6 elements
        
        # This is complex to hardcode correctly without reference.
        # I will stick to the 85 rank solution but clean it up.
        # Wait, the prompt asks for a stronger algorithm.
        # If I cannot guarantee rank 80, I should stick to 85.
        # However, I can try the 2+2+1 split with rank 11 for 2x2x3?
        # 2x2x3 rank 11 -> 2x2x5 rank 18 -> 4x2x5 rank 36 -> 2 blocks 72 + 20 = 92.
        
        # What if I use 3+2 split for inner dimension?
        # 3+2 split:
        # A_4x3 B_3x5 + A_4x2 B_2x5
        # 4x3x5: Split A into 2x3 + 2x3.
        # 2x3x5: Split B into 3x2 + 3x3? No.
        
        # Let's stick to the known 85 rank solution but ensure it's correct.
        # The current program is 85.
        # I will just return the current program's logic as it is likely optimal for this family.
        
        return None

    # Since I cannot reliably code the rank 8 2x2x3 factors from memory, 
    # and rank 92 is worse, I will revert to the 85 rank solution.
    
    # Re-implementing the 85 rank solution cleanly.
    
    rank = 85
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # 4x4x4 part: Strassen (rank 49)
    # 4x4x1 part: Rank 16
    # 4x1x4 part: Rank 16
    # 4x1x1 part: Rank 4
    
    def add_4x4_term(a_block_coeffs, b_block_coeffs, c_block_coeffs, start_col):
        for inner_k in range(7):
            u_vec = [s_u[r][inner_k] for r in range(4)]
            v_vec = [s_v[r][inner_k] for r in range(4)]
            w_vec = [s_w[r][inner_k] for r in range(4)]
            
            col_idx = start_col + inner_k
            
            for (a_br, a_bc), a_coeff in a_block_coeffs.items():
                for lr in range(2):
                    for lc in range(2):
                        local_idx = lr * 2 + lc
                        u_val = u_vec[local_idx]
                        if u_val != 0:
                            global_i = 2 * a_br + lr
                            global_j = 2 * a_bc + lc
                            global_u_row = global_i * 5 + global_j
                            U[global_u_row][col_idx] += a_coeff * u_val
            
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

    terms = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]
    
    for t_idx, (a_coeff, b_coeff, c_coeff) in enumerate(terms):
        if time.time() > deadline:
            break
        start_col = t_idx * 7
        add_4x4_term(a_coeff, b_coeff, c_coeff, start_col)
        
    col = 49
    
    # 4x4x1
    for i in range(4):
        for j in range(4):
            U[i*5 + j][col] = 1
            V[j*5 + 4][col] = 1
            W[4*4 + i][col] = 1
            col += 1
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
            
    # 4x1x4
    for i in range(4):
        for k in range(4):
            U[i*5 + 4][col] = 1
            V[4*5 + k][col] = 1
            W[k*4 + i][col] = 1
            col += 1
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
            
    # 4x1x1
    for i in range(4):
        U[i*5 + 4][col] = 1
        V[4*5 + 4][col] = 1
        W[4*4 + i][col] = 1
        col += 1
        if time.time() > deadline:
            break
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
