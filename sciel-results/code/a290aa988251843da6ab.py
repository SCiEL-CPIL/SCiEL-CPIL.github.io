import numpy as np
import time

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # 40s safety margin

    # Helper to verify the decomposition
    def verify(U, V, W, rank):
        # Convert to numpy arrays
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # Check shapes
        if U_np.shape != (n * m, rank) or V_np.shape != (m * p, rank) or W_np.shape != (p * n, rank):
            return False

        # Compute T = sum U_c outer V_c outer W_c
        # T[a, b, c] = sum_col U[a, col] * V[b, col] * W[c, col]
        # We can do this efficiently using einsum or matrix multiplication
        # T[a, b, c] = (U @ V.T)[a, col] * W[c, col]? No.
        # T = sum_c U_c V_c^T W_c^T? No, U, V, W are vectors.
        # T is a 3D tensor.
        # T[a, b, c] = sum_k U[a, k] V[b, k] W[c, k]
        
        # Construct the target tensor T_target
        T_target = np.zeros((n * m, m * p, p * n), dtype=int)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T_target[a, b, c] = 1

        # Compute T_calc
        # Efficient way:
        # T_calc[a, b, c] = sum_k U[a, k] V[b, k] W[c, k]
        # This is like a tensor contraction.
        # We can compute M = U @ V.T (n*m x m*p). M[a, b] = sum_k U[a, k] V[b, k].
        # Then T_calc[a, b, c] = sum_k M[a, b] * ... no, M already summed over k.
        # We need to keep the k dimension.
        # T_calc = np.einsum('ak,bk,ck->abc', U_np, V_np, W_np)
        
        try:
            T_calc = np.einsum('ak,bk,ck->abc', U_np, V_np, W_np)
        except Exception:
            return False

        # Check if T_calc matches T_target
        # Allow for floating point errors, but entries should be integers
        if not np.allclose(T_calc, T_target, atol=1e-6):
            return False
            
        # Check if entries are integers or half-integers
        # The problem statement says "integer/half-integer entries"
        # This refers to the entries of U, V, W.
        # We are using integers, so this is fine.
        
        return True

    # Schoolbook fallback
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    # Try the optimized rank 85 decomposition
    if time.time() < deadline:
        U_opt, V_opt, W_opt, rank_opt = _build_optimized_decomposition(n, m, p)
        if verify(U_opt, V_opt, W_opt, rank_opt):
            return {"U": U_opt, "V": V_opt, "W": W_opt}
        else:
            # Verification failed, fallback to schoolbook
            pass

    return {"U": U_sb, "V": V_sb, "W": W_sb}

def _build_optimized_decomposition(n, m, p):
    # n=4, m=5, p=5
    # Split m=5 into 4 and 1.
    # A (4x5) -> A_L (4x4), A_R (4x1)
    # B (5x5) -> B_T (4x5), B_B (1x5)
    # B_T (4x5) -> B_TL (4x4), B_TR (4x1)
    # B_B (1x5) -> B_BL (1x4), B_BR (1x1)
    
    # C (4x5) -> C_L (4x4), C_R (4x1)
    
    # Parts:
    # 1. A_L (4x4) * B_TL (4x4) -> C_L (4x4). Rank 49.
    # 2. A_L (4x4) * B_TR (4x1) -> C_R (4x1). Rank 16.
    # 3. A_R (4x1) * B_BL (1x4) -> C_L (4x4). Rank 16.
    # 4. A_R (4x1) * B_BR (1x1) -> C_R (4x1). Rank 4.
    
    rank = 49 + 16 + 16 + 4
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper for 2x2 Strassen
    def get_strassen_2x2_factors():
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

    s_u, s_v, s_w = get_strassen_2x2_factors()
    
    # Function to add 4x4x4 Strassen decomposition
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
                            # A_L is 4x4. Global A index: i * 5 + j.
                            # A_L is cols 0..3 of A.
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
                            # B_TL is 4x4. Global B index: j * 5 + k.
                            # B_TL is rows 0..3, cols 0..3 of B.
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
                            # C_L is 4x4. Global C index: k * 4 + i.
                            # C_L is cols 0..3 of C.
                            global_w_row = global_k * 4 + global_i
                            W[global_w_row][col_idx] += c_coeff * w_val

    # 7 Strassen terms for 4x4x4
    terms = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}), # M1
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}), # M2
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}), # M3
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}), # M4
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}), # M5
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}), # M6
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}), # M7
    ]
    
    col = 0
    for t_idx, (a_coeff, b_coeff, c_coeff) in enumerate(terms):
        start_col = t_idx * 7
        add_4x4_term(a_coeff, b_coeff, c_coeff, start_col)
        
    col = 49
    
    # 4x4x1 part: A_L (4x4) * B_TR (4x1) -> C_R (4x1)
    # A_L rows i, cols j (0..3). B_TR rows j (0..3), col 4. C_R rows i, col 4.
    for i in range(4):
        for j in range(4):
            U[i*5 + j][col] = 1
            V[j*5 + 4][col] = 1
            W[4*4 + i][col] = 1
            col += 1
            
    # 4x1x4 part: A_R (4x1) * B_BL (1x4) -> C_L (4x4)
    # A_R rows i, col 4. B_BL row 4, cols k (0..3). C_L rows i, cols k.
    for i in range(4):
        for k in range(4):
            U[i*5 + 4][col] = 1
            V[4*5 + k][col] = 1
            W[k*4 + i][col] = 1
            col += 1
            
    # 4x1x1 part: A_R (4x1) * B_BR (1x1) -> C_R (4x1)
    for i in range(4):
        U[i*5 + 4][col] = 1
        V[4*5 + 4][col] = 1
        W[4*4 + i][col] = 1
        col += 1
        
    return U, V, W, rank
# EVOLVE-BLOCK-END
