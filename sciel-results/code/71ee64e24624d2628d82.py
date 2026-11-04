import numpy as np
import time

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    # Rank 24: schoolbook decomposition of 2x4x6 into two 2x2x3 subproblems
    # Each 2x2x3 uses 6 multiplications (schoolbook: 2*2*3 = 12? No, 2*3=6 outputs, each needs 2 terms = 12)
    # Wait: C[i,j] = A[i,0]*B[0,j] + A[i,1]*B[1,j]
    # For 2x2x3: 2 rows * 3 cols = 6 outputs, each sum of 2 products = 12 total products in schoolbook
    # So two 2x2x3 blocks = 24 products total. Rank = 24.
    
    rank = 24
    U = [[0] * rank for _ in range(n * m)]   # 8 x 24
    V = [[0] * rank for _ in range(m * p)]   # 24 x 24
    W = [[0] * rank for _ in range(p * n)]   # 12 x 24
    
    # Split the 4x6 B matrix into two 2x3 blocks:
    # Block 0: rows 0-1 of B (which are columns 0-1 of the middle dimension), cols 0-2
    # Block 1: rows 2-3 of B (columns 2-3 of middle dim), cols 3-5
    # 
    # A is 2x4: split into A_left (2x2, cols 0-1) and A_right (2x2, cols 2-3)
    # B is 4x6: split into B_top (2x3, rows 0-1) and B_bottom (2x3, rows 2-3)
    # 
    # C[0,0:3] = A_left * B_top
    # C[0,3:6] = A_right * B_top  -- NO, that's wrong.
    
    # Correct decomposition:
    # C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    # = sum_{j=0}^{1} A[i,j] * B[j,k] + sum_{j=2}^{3} A[i,j] * B[j,k]
    
    # For k=0,1,2 (first 3 columns of C):
    #   C[i,0:3] = A[i,0:2] * B[0:2, 0:3] + A[i,2:4] * B[2:4, 0:3]
    # For k=3,4,5 (last 3 columns of C):
    #   C[i,3:6] = A[i,0:2] * B[0:2, 3:6] + A[i,2:4] * B[2:4, 3:6]
    
    # So we have 4 subproblems, each 2x2x3:
    # Subproblem (a, b) where a in {0,1} (which half of A's columns), b in {0,1} (which half of B's rows)
    # contributes to C[:, 3*b : 3*b+3]
    
    # For each subproblem (a, b):
    #   A_sub = A[:, 2*a : 2*a+2]  (2x2)
    #   B_sub = B[2*b : 2*b+2, 3*b : 3*b+3]  (2x3)
    #   Contribution to C[:, 3*b : 3*b+3] = A_sub * B_sub
    #
    # Each 2x2x3 schoolbook: 2*2*3 = 12 products? No.
    # C[i,j] = A_sub[i,0]*B_sub[0,j] + A_sub[i,1]*B_sub[1,j]
    # That's 2 terms per output, 6 outputs = 12 products.
    
    # 4 subproblems * 12 = 48. That's the same as full schoolbook!
    
    # I need to use rank-6 for each 2x2x3. Let me find the correct rank-6 decomposition.
    
    # Known: M_{2,2,3} has rank 6. One decomposition:
    # Products (using indices i in {0,1}, k in {0,1}, j in {0,1,2}):
    # P0 = (A[0,0]+A[0,1]) * (B[0,0]+B[1,0])
    # P1 = (A[0,0]+A[0,1]) * (B[0,1]+B[1,1])  
    # P2 = (A[0,0]+A[0,1]) * (B[0,2]+B[1,2])
    # P3 = (A[1,0]+A[1,1]) * (B[0,0]+B[1,0])
    # P4 = (A[1,0]+A[1,1]) * (B[0,1]+B[1,1])
    # P5 = (A[1,0]+A[1,1]) * (B[0,2]+B[1,2])
    
    # Verify C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # P0 = A[0,0]*B[0,0] + A[0,0]*B[1,0] + A[0,1]*B[0,0] + A[0,1]*B[1,0]
    # P3 = A[1,0]*B[0,0] + A[1,0]*B[1,0] + A[1,1]*B[0,0] + A[1,1]*B[1,0]
    # P0 - P3 = A[0,0]*B[0,0] + A[0,0]*B[1,0] + A[0,1]*B[0,0] + A[0,1]*B[1,0] 
    #           - A[1,0]*B[0,0] - A[1,0]*B[1,0] - A[1,1]*B[0,0] - A[1,1]*B[1,0]
    # This doesn't simplify to C[0,0].
    
    # I think the rank-6 for 2x2x3 requires a different structure. Let me just use rank 12 per subproblem
    # (schoolbook) giving total rank 48, which is no better. 
    # 
    # Alternative: use Strassen on the 2x2x2 parts. 
    # 2x4x6: think of it as 2x2x6 concatenated with 2x2x6.
    # Each 2x2x6 can be done with 2x2x2 Strassen (rank 7) for 3 blocks = 21 per half, total 42.
    # That's what the original does.
    
    # Let me try a different split: 4x2x6 = (2x2x6) + (2x2x6), same thing.
    
    # Or: 2x4x6 = 2x2x3 + 2x2x3 + 2x2x3 + 2x2x3 (four 2x2x3 blocks)
    # If each 2x2x3 has rank 6, total = 24.
    
    # I need the correct rank-6 for 2x2x3. Let me derive it.
    # 
    # We want: C[i,j] = A[i,0]*B[0,j] + A[i,1]*B[1,j] for i in {0,1}, j in {0,1,2}
    #
    # 6 rank-1 tensors:
    # t=0: U[0,0]=1, U[1,0]=0; V[0,0]=1, V[1,0]=0; W[0,0]=1, W[1,0]=0, W[2,0]=0, W[3,0]=0, W[4,0]=0, W[5,0]=0
    #       -> A[0,0]*B[0,0] -> C[0,0]
    # t=1: U[0,0]=1, U[1,0]=0; V[0,0]=1, V[1,0]=0; W[0,0]=0, W[1,0]=1, ...
    #       -> A[0,0]*B[0,1] -> C[0,1]
    # ...
    
    # This gives 12 terms for schoolbook. To get rank 6, we need to combine.
    
    # I'll just implement the full schoolbook (rank 48) as a fallback, but that's worse.
    # Let me stick with the original rank-42 Strassen approach but clean it up.
    
    # Actually, let me just return the original rank-42 solution since I can't easily verify rank-12.
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    strassen_A = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    
    strassen_B = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    
    w_coeffs = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]
    
    for b in range(6):
        s = b // 3
        k_block = b % 3
        
        for t in range(7):
            c = 7 * b + t
            
            u_coeff = strassen_A[t]
            for i in range(2):
                for j_loc in range(2):
                    idx_A = i * 2 + j_loc
                    coeff = u_coeff[idx_A]
                    if coeff != 0:
                        global_i = i
                        global_j = 2 * s + j_loc
                        U[global_i * m + global_j][c] = coeff
                        
            v_coeff = strassen_B[t]
            for j_loc in range(2):
                for k_loc in range(2):
                    idx_B = j_loc * 2 + k_loc
                    coeff = v_coeff[idx_B]
                    if coeff != 0:
                        global_j = 2 * s + j_loc
                        global_k = 2 * k_block + k_loc
                        V[global_j * p + global_k][c] = coeff
                        
            w_coeff = w_coeffs[t]
            for i in range(2):
                for k_loc in range(2):
                    idx_W = i * 2 + k_loc
                    coeff = w_coeff[idx_W]
                    if coeff != 0:
                        global_k = 2 * k_block + k_loc
                        global_i = i
                        W[global_k * n + global_i][c] = coeff
                        
    return {"U": U, "V": V, "W": W}
