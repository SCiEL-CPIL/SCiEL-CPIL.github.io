import numpy as np
import time

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 150.0
    time_limit_fraction = 0.8
    
    # Seed solution: Rank 48 (n*m*p)
    rank_seed = n * m * p
    U_seed = [[0] * rank_seed for _ in range(n * m)]
    V_seed = [[0] * rank_seed for _ in range(m * p)]
    W_seed = [[0] * rank_seed for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_seed[i * m + j][column] = 1
                V_seed[j * p + k][column] = 1
                W_seed[k * n + i][column] = 1
                column += 1
    
    def verify(U, V, W):
        U_np = np.array(U)
        V_np = np.array(V)
        W_np = np.array(W)
        r = len(U[0])
        
        # Check all 2304 entries of the tensor T[a,b,d]
        # T[a,b,d] = sum_c U[a,c] V[b,c] W[d,c]
        # Expected: 1 if (a,b,d) corresponds to valid (i,j,k)
        # a = i*m + j
        # b = j*p + k
        # d = k*n + i
        
        # Vectorized check for speed
        # We can compute the full tensor T_recon[a,b,d] using einsum
        # But 8*24*12 = 2304 entries is small.
        # Let's just loop.
        
        for a in range(n*m):
            for b in range(m*p):
                for d in range(p*n):
                    val = 0.0
                    for c in range(r):
                        val += U_np[a, c] * V_np[b, c] * W_np[d, c]
                    
                    # Determine expected value
                    i_a = a // m
                    j_a = a % m
                    j_b = b // p
                    k_b = b % p
                    k_d = d // n
                    i_d = d % n
                    
                    if i_a == i_d and j_a == j_b and k_b == k_d:
                        expected = 1.0
                    else:
                        expected = 0.0
                        
                    if abs(val - expected) > 1e-9:
                        return False
        return True

    # Try to find a better rank using Strassen-inspired block decomposition
    # We know rank 42 is possible via 6 blocks of Strassen 2x2x2.
    # Can we do better?
    # Theoretical lower bound is ceil(n*p*m / (something))? No, just rank.
    # For 2x4x6, rank 42 is the best known simple construction.
    # Is rank 40 possible?
    # Let's try to construct rank 42 first to ensure we beat the seed (48).
    
    def construct_strassen_42():
        rank_total = 42
        U = [[0] * rank_total for _ in range(n * m)]
        V = [[0] * rank_total for _ in range(m * p)]
        W = [[0] * rank_total for _ in range(p * n)]
        
        # Strassen 2x2 @ 2x2 rank-7 decomposition
        # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
        # P1 = (a11+a22)*(b11+b22)
        # P2 = (a21+a22)*b11
        # P3 = a11*(b12-b22)
        # P4 = a22*(b21-b11)
        # P5 = (a11+a12)*b22
        # P6 = (a21-a11)*(b11+b12)
        # P7 = (a12-a22)*(b21+b22)
        #
        # c11 = P1 + P4 - P5 + P7
        # c12 = P3 + P5
        # c21 = P2 + P4
        # c22 = P1 - P2 + P3 + P6
        
        # U local index mapping for A (2x2):
        # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
        # V local index mapping for B (2x2):
        # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
        # W local index mapping for C (2x2) with convention W[k*n+i]:
        # C[0,0] -> idx 0*2+0 = 0
        # C[0,1] -> idx 1*2+0 = 2
        # C[1,0] -> idx 0*2+1 = 1
        # C[1,1] -> idx 1*2+1 = 3
        # So W indices: 0:(0,0), 1:(1,0), 2:(0,1), 3:(1,1)
        
        # Strassen terms (U_coeffs, V_coeffs, W_coeffs)
        # U_coeffs: dict {local_idx: sign}
        # V_coeffs: dict {local_idx: sign}
        # W_coeffs: dict {local_idx: sign}
        
        strassen_terms = [
            # P1
            ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
            # P2
            ({2: 1, 3: 1}, {0: 1}, {1: 1, 3: -1}),
            # P3
            ({0: 1}, {1: 1, 3: -1}, {2: 1, 3: 1}),
            # P4
            ({3: 1}, {2: 1, 0: -1}, {0: 1, 1: 1}),
            # P5
            ({0: 1, 1: 1}, {3: 1}, {0: -1, 2: 1}),
            # P6
            ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
            # P7
            ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1})
        ]
        
        # We have 3 blocks of columns in B (size 2x2 each)
        # And 2 blocks of rows in A? No, A is 2x4.
        # Split A into two 2x2 blocks (cols 0-1, cols 2-3).
        # Split B into three 2x2 blocks (rows 0-1, rows 2-3 ... wait B is 4x6).
        # B is 4x6. Split rows 0-1 and 2-3.
        # So we have:
        # A_left (2x2) @ B_top (2x6)
        # A_right (2x2) @ B_bottom (2x6)
        
        # For A_left @ B_top:
        # A_left is A[:, 0:2]
        # B_top is B[0:2, :]
        # B_top is 2x6. Split into three 2x2 blocks: B[0:2, 0:2], B[0:2, 2:4], B[0:2, 4:6]
        # So 3 Strassen applications.
        
        # For A_right @ B_bottom:
        # A_right is A[:, 2:4]
        # B_bottom is B[2:4, :]
        # B_bottom is 2x6. Split into three 2x2 blocks: B[2:4, 0:2], B[2:4, 2:4], B[2:4, 4:6]
        # So 3 Strassen applications.
        
        # Total 6 applications * 7 rank = 42.
        
        col_offset = 0
        
        # Helper to apply one Strassen block
        def apply_strassen(U_base_row_offset, V_base_row_offset, W_base_col_offset, col_start):
            # U_base_row_offset: 0 for A_left, 2 for A_right (since A is 2x4, rows 0-1, cols 0-3)
            # Wait, A is 2x4.
            # A_left: cols 0,1. A_right: cols 2,3.
            # U index for A[i, j] is i*m + j = i*4 + j.
            # For A_left (j in 0,1):
            # local 0 (0,0) -> U[0*4+0] = U[0]
            # local 1 (0,1) -> U[0*4+1] = U[1]
            # local 2 (1,0) -> U[1*4+0] = U[4]
            # local 3 (1,1) -> U[1*4+1] = U[5]
            
            # For A_right (j in 2,3):
            # local 0 (0,2) -> U[0*4+2] = U[2]
            # local 1 (0,3) -> U[0*4+3] = U[3]
            # local 2 (1,2) -> U[1*4+2] = U[6]
            # local 3 (1,3) -> U[1*4+3] = U[7]
            
            # V_base_row_offset: 0 for B_top, 2 for B_bottom.
            # V index for B[i, j] is i*p + j = i*6 + j.
            # B_top (i in 0,1):
            # local 0 (0, col) -> V[0*6+col]
            # local 1 (0, col+1) -> V[0*6+col+1]
            # local 2 (1, col) -> V[1*6+col]
            # local 3 (1, col+1) -> V[1*6+col+1]
            
            # B_bottom (i in 2,3):
            # local 0 (2, col) -> V[2*6+col]
            # local 1 (2, col+1) -> V[2*6+col+1]
            # local 2 (3, col) -> V[3*6+col]
            # local 3 (3, col+1) -> V[3*6+col+1]
            
            # W_base_col_offset: 0, 2, or 4.
            # W index for C[i, k] is k*n + i = k*2 + i.
            # local 0 (0, col) -> W[col*2+0]
            # local 1 (1, col) -> W[col*2+1]
            # local 2 (0, col+1) -> W[(col+1)*2+0]
            # local 3 (1, col+1) -> W[(col+1)*2+1]
            
            for t_idx in range(7):
                u_coeffs, v_coeffs, w_coeffs = strassen_terms[t_idx]
                col = col_start + t_idx
                
                # U
                for li, sign in u_coeffs.items():
                    if U_base_row_offset == 0:
                        # A_left
                        global_idx = li # 0,1,4,5 mapping? No.
                        # li 0 -> (0,0) -> idx 0
                        # li 1 -> (0,1) -> idx 1
                        # li 2 -> (1,0) -> idx 4
                        # li 3 -> (1,1) -> idx 5
                        if li == 0: idx = 0
                        elif li == 1: idx = 1
                        elif li == 2: idx = 4
                        elif li == 3: idx = 5
                    else:
                        # A_right
                        # li 0 -> (0,2) -> idx 2
                        # li 1 -> (0,3) -> idx 3
                        # li 2 -> (1,2) -> idx 6
                        # li 3 -> (1,3) -> idx 7
                        if li == 0: idx = 2
                        elif li == 1: idx = 3
                        elif li == 2: idx = 6
                        elif li == 3: idx = 7
                    U[idx][col] += sign
                
                # V
                for li, sign in v_coeffs.items():
                    if V_base_row_offset == 0:
                        # B_top
                        # li 0 -> (0, col) -> idx col
                        # li 1 -> (0, col+1) -> idx col+1
                        # li 2 -> (1, col) -> idx 6+col
                        # li 3 -> (1, col+1) -> idx 6+col+1
                        if li == 0: idx = W_base_col_offset
                        elif li == 1: idx = W_base_col_offset + 1
                        elif li == 2: idx = 6 + W_base_col_offset
                        elif li == 3: idx = 6 + W_base_col_offset + 1
                    else:
                        # B_bottom
                        # li 0 -> (2, col) -> idx 12+col
                        # li 1 -> (2, col+1) -> idx 12+col+1
                        # li 2 -> (3, col) -> idx 18+col
                        # li 3 -> (3, col+1) -> idx 18+col+1
                        if li == 0: idx = 12 + W_base_col_offset
                        elif li == 1: idx = 12 + W_base_col_offset + 1
                        elif li == 2: idx = 18 + W_base_col_offset
                        elif li == 3: idx = 18 + W_base_col_offset + 1
                    V[idx][col] += sign
                
                # W
                for li, sign in w_coeffs.items():
                    # W indices are same for both A_left and A_right parts
                    # li 0 -> (0, col) -> idx col*2+0
                    # li 1 -> (1, col) -> idx col*2+1
                    # li 2 -> (0, col+1) -> idx (col+1)*2+0
                    # li 3 -> (1, col+1) -> idx (col+1)*2+1
                    if li == 0: idx = W_base_col_offset * 2 + 0
                    elif li == 1: idx = W_base_col_offset * 2 + 1
                    elif li == 2: idx = (W_base_col_offset + 1) * 2 + 0
                    elif li == 3: idx = (W_base_col_offset + 1) * 2 + 1
                    W[idx][col] += sign
        
        # 3 blocks of columns for B
        for col_block in range(3):
            w_offset = col_block * 2
            
            # A_left @ B_top
            apply_strassen(0, 0, w_offset, col_offset)
            col_offset += 7
            
            # A_right @ B_bottom
            apply_strassen(2, 2, w_offset, col_offset)
            col_offset += 7
        
        return U, V, W

    U_42, V_42, W_42 = construct_strassen_42()
    
    if verify(U_42, V_42, W_42):
        return {"U": U_42, "V": V_42, "W": W_42}
    else:
        # Fallback to seed if Strassen construction failed (should not happen)
        return {"U": U_seed, "V": V_seed, "W": W_seed}
