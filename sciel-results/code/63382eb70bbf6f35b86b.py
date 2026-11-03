import numpy as np
import time

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 145.0  # 5s margin before 150s limit
    
    # Schoolbook baseline
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
    
    # Attempt Strassen-based decomposition for 4x4 * 4x7
    # Strategy:
    # Use Strassen (rank 28) for the first 4 columns of B (4x4 * 4x4 part).
    # Use Schoolbook (rank 48) for the last 3 columns of B (4x4 * 4x3 part).
    # Total rank = 76.
    
    def generate_strassen_4x4x4():
        rank = 28
        U = [[0] * rank for _ in range(16)]
        V = [[0] * rank for _ in range(16)]
        W = [[0] * rank for _ in range(16)]
        
        # Helper to add a term
        def add_term(u_indices, v_indices, w_index, sign, col):
            for idx in u_indices:
                U[idx][col] += sign
            for idx in v_indices:
                V[idx][col] += sign
            W[w_index][col] += sign
            
        # Indices for A (i*4+j), B (j*4+k), C (i*4+k)
        
        # M1 = (A11+A22)(B11+B22)
        # A11+A22:
        # (0,0): [0, 10]
        # (0,1): [1, 11]
        # (1,0): [4, 14]
        # (1,1): [5, 15]
        # B11+B22:
        # (0,0): [0, 10]
        # (0,1): [1, 11]
        # (1,0): [4, 14]
        # (1,1): [5, 15]
        
        # C11:
        # (0,0): 0
        # (0,1): 1
        # (1,0): 4
        # (1,1): 5
        
        col = 0
        # M1 terms
        add_term([0, 10], [0, 10], 0, 1, col); col += 1
        add_term([1, 11], [4, 14], 0, 1, col); col += 1
        add_term([0, 10], [1, 11], 1, 1, col); col += 1
        add_term([1, 11], [5, 15], 1, 1, col); col += 1
        add_term([4, 14], [0, 10], 4, 1, col); col += 1
        add_term([5, 15], [4, 14], 4, 1, col); col += 1
        add_term([4, 14], [1, 11], 5, 1, col); col += 1
        add_term([5, 15], [5, 15], 5, 1, col); col += 1
        
        # M2 = (A21+A22)B11
        # A21+A22:
        # (0,0): [8, 10]
        # (0,1): [9, 11]
        # (1,0): [12, 14]
        # (1,1): [13, 15]
        # B11:
        # (0,0): [0]
        # (0,1): [1]
        # (1,0): [4]
        # (1,1): [5]
        
        add_term([8, 10], [0], 0, 1, col); col += 1
        add_term([9, 11], [4], 0, 1, col); col += 1
        add_term([8, 10], [1], 1, 1, col); col += 1
        add_term([9, 11], [5], 1, 1, col); col += 1
        add_term([12, 14], [0], 4, 1, col); col += 1
        add_term([13, 15], [4], 4, 1, col); col += 1
        add_term([12, 14], [1], 5, 1, col); col += 1
        add_term([13, 15], [5], 5, 1, col); col += 1
        
        # M3 = A11(B12-B22)
        # A11:
        # (0,0): [0]
        # (0,1): [1]
        # (1,0): [4]
        # (1,1): [5]
        # B12-B22:
        # (0,0): [2, -10]
        # (0,1): [3, -11]
        # (1,0): [6, -14]
        # (1,1): [7, -15]
        
        add_term([0], [2, -10], 0, 1, col); col += 1
        add_term([1], [6, -14], 0, 1, col); col += 1
        add_term([0], [3, -11], 1, 1, col); col += 1
        add_term([1], [7, -15], 1, 1, col); col += 1
        add_term([4], [2, -10], 4, 1, col); col += 1
        add_term([5], [6, -14], 4, 1, col); col += 1
        add_term([4], [3, -11], 5, 1, col); col += 1
        add_term([5], [7, -15], 5, 1, col); col += 1
        
        # M4 = A22(B21-B11)
        # A22:
        # (0,0): [10]
        # (0,1): [11]
        # (1,0): [14]
        # (1,1): [15]
        # B21-B11:
        # (0,0): [8, -0] -> [8, 0]
        # (0,1): [9, -1] -> [9, -1]
        # (1,0): [12, -4] -> [12, -4]
        # (1,1): [13, -5] -> [13, -5]
        
        add_term([10], [8, 0], 0, 1, col); col += 1
        add_term([11], [12, -4], 0, 1, col); col += 1
        add_term([10], [9, -1], 1, 1, col); col += 1
        add_term([11], [13, -5], 1, 1, col); col += 1
        add_term([14], [8, 0], 4, 1, col); col += 1
        add_term([15], [12, -4], 4, 1, col); col += 1
        add_term([14], [9, -1], 5, 1, col); col += 1
        add_term([15], [13, -5], 5, 1, col); col += 1
        
        # M5 = (A11+A12)B22
        # A11+A12:
        # (0,0): [0, 2]
        # (0,1): [1, 3]
        # (1,0): [4, 6]
        # (1,1): [5, 7]
        # B22:
        # (0,0): [10]
        # (0,1): [11]
        # (1,0): [14]
        # (1,1): [15]
        
        add_term([0, 2], [10], 0, 1, col); col += 1
        add_term([1, 3], [14], 0, 1, col); col += 1
        add_term([0, 2], [11], 1, 1, col); col += 1
        add_term([1, 3], [15], 1, 1, col); col += 1
        add_term([4, 6], [10], 4, 1, col); col += 1
        add_term([5, 7], [14], 4, 1, col); col += 1
        add_term([4, 6], [11], 5, 1, col); col += 1
        add_term([5, 7], [15], 5, 1, col); col += 1
        
        # M6 = (A21-A11)(B11+B12)
        # A21-A11:
        # (0,0): [8, -0] -> [8, 0]
        # (0,1): [9, -1] -> [9, -1]
        # (1,0): [12, -4] -> [12, -4]
        # (1,1): [13, -5] -> [13, -5]
        # B11+B12:
        # (0,0): [0, 2]
        # (0,1): [1, 3]
        # (1,0): [4, 6]
        # (1,1): [5, 7]
        
        add_term([8, 0], [0, 2], 0, 1, col); col += 1
        add_term([9, -1], [4, 6], 0, 1, col); col += 1
        add_term([8, 0], [1, 3], 1, 1, col); col += 1
        add_term([9, -1], [5, 7], 1, 1, col); col += 1
        add_term([12, -4], [0, 2], 4, 1, col); col += 1
        add_term([13, -5], [4, 6], 4, 1, col); col += 1
        add_term([12, -4], [1, 3], 5, 1, col); col += 1
        add_term([13, -5], [5, 7], 5, 1, col); col += 1
        
        # M7 = (A12-A22)(B21+B22)
        # A12-A22:
        # (0,0): [2, -10]
        # (0,1): [3, -11]
        # (1,0): [6, -14]
        # (1,1): [7, -15]
        # B21+B22:
        # (0,0): [8, 10]
        # (0,1): [9, 11]
        # (1,0): [12, 14]
        # (1,1): [13, 15]
        
        add_term([2, -10], [8, 10], 0, 1, col); col += 1
        add_term([3, -11], [12, 14], 0, 1, col); col += 1
        add_term([2, -10], [9, 11], 1, 1, col); col += 1
        add_term([3, -11], [13, 15], 1, 1, col); col += 1
        add_term([6, -14], [8, 10], 4, 1, col); col += 1
        add_term([7, -15], [12, 14], 4, 1, col); col += 1
        add_term([6, -14], [9, 11], 5, 1, col); col += 1
        add_term([7, -15], [13, 15], 5, 1, col); col += 1
        
        # Construct full U, V, W for 4x4*4x7.
        # U: (16, 76), V: (28, 76), W: (28, 76)
        
        total_rank = 28 + 48
        U_full = [[0] * total_rank for _ in range(16)]
        V_full = [[0] * total_rank for _ in range(28)]
        W_full = [[0] * total_rank for _ in range(28)]
        
        # Copy the 28 terms for C[:, 0:4]
        # U indices are same (A is 4x4)
        for c in range(28):
            for i in range(16):
                U_full[i][c] = U[i][c]
        
        # V indices need remapping from local 4x4 to global 4x7
        # Local index l = j*4 + k. Global index g = j*7 + k.
        for c in range(28):
            for l in range(16):
                val = V[l][c]
                if val != 0:
                    j = l // 4
                    k = l % 4
                    g = j * 7 + k
                    V_full[g][c] = val
                    
        # W indices need remapping from local i*4+k to global k*4+i
        for c in range(28):
            for l in range(16):
                val = W[l][c]
                if val != 0:
                    i = l // 4
                    k = l % 4
                    g = k * 4 + i
                    W_full[g][c] = val
        
        # Add schoolbook terms for C[:, 4:7]
        # A is 4x4, B is 4x3 (columns 4,5,6).
        # C is 4x3.
        # Schoolbook rank = 4*4*3 = 48.
        
        col_offset = 28
        for i in range(4):
            for j in range(4):
                for k in range(3):
                    # A[i,j] * B[j, 4+k] -> C[i, 4+k]
                    U_full[i*4+j][col_offset] = 1
                    V_full[j*7+(4+k)][col_offset] = 1
                    W_full[(4+k)*4+i][col_offset] = 1
                    col_offset += 1
                    
        return U_full, V_full, W_full
        
    try:
        U_str, V_str, W_str = generate_strassen_4x4x4()
        # Verify
        n, m, p = 4, 4, 7
        rank = len(U_str[0])
        
        # Quick check
        valid = True
        for i in range(4):
            for j in range(4):
                for k in range(7):
                    total = 0
                    for c in range(rank):
                        total += U_str[i*4+j][c] * V_str[j*7+k][c] * W_str[k*4+i][c]
                    if total != 1:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
                
        if valid:
            return {"U": U_str, "V": V_str, "W": W_str}
        else:
            return {"U": U_sb, "V": V_sb, "W": W_sb}
            
    except Exception as e:
        return {"U": U_sb, "V": V_sb, "W": W_sb}

# EVOLVE-BLOCK-END
