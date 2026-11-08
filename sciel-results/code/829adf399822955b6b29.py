import time
import sys

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110  # 10s margin for 120s limit
    
    n, m, p = DIMENSIONS
    # n=2, m=4, p=8
    # U: (n*m, R) = (8, R)
    # V: (m*p, R) = (32, R)
    # W: (p*n, R) = (16, R)
    
    # Strassen coefficients for 2x2 matrix multiplication
    # Local indices for 2x2 matrix:
    # 0: 0,0; 1: 0,1; 2: 1,0; 3: 1,1
    # Strassen terms:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # U_local vectors (4 elements) for each term
    U_s = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0],# M6
        [0, 1, 0, -1] # M7
    ]
    
    # V_local vectors (4 elements) for each term
    V_s = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1],# M3
        [0, 0, 1, -1],# M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    
    # W_local vectors (4 elements) for each term
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    W_s = [
        [1, 0, 0, 1],  # M1: C00=1, C01=0, C10=0, C11=1
        [0, 0, 1, -1], # M2: C00=0, C01=0, C10=1, C11=-1
        [0, 1, 0, 1],  # M3: C00=0, C01=1, C10=0, C11=1
        [1, 0, 1, 0],  # M4: C00=1, C01=0, C10=1, C11=0
        [-1, 1, 0, 0], # M5: C00=-1, C01=1, C10=0, C11=0
        [0, 0, 0, 1],  # M6: C00=0, C01=0, C10=0, C11=1
        [1, 0, 0, 0]   # M7: C00=1, C01=0, C10=0, C11=0
    ]
    
    # Define the 8 sub-problems
    # Each sub-problem is defined by:
    # A_row_start, A_col_start (for 2x2 block in A)
    # B_row_start, B_col_start (for 2x2 block in B)
    # C_row_start, C_col_start (for 2x2 block in C)
    
    # A is 2x4. B is 4x8. C is 2x8.
    # A_L: rows 0-1, cols 0-1. A_R: rows 0-1, cols 2-3.
    # B_T: rows 0-1. B_B: rows 2-3.
    # B cols: 0-1, 2-3, 4-5, 6-7.
    
    sub_problems = []
    # A_L (0,0) with B_T (0,0..7)
    for b_col in [0, 2, 4, 6]:
        sub_problems.append((0, 0, 0, b_col, 0, b_col))
    # A_R (0,2) with B_B (2,0..7)
    for b_col in [0, 2, 4, 6]:
        sub_problems.append((0, 2, 2, b_col, 0, b_col))
        
    rank = 8 * 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for sp in sub_problems:
        a_r, a_c, b_r, b_c, c_r, c_c = sp
        
        for t in range(7):
            # Map local U indices (0..3) to global U indices
            # Local (lr, lc) -> global (a_r + lr, a_c + lc)
            # Global U index: (a_r + lr) * m + (a_c + lc)
            for lr in range(2):
                for lc in range(2):
                    local_idx = lr * 2 + lc
                    val = U_s[t][local_idx]
                    if val != 0:
                        global_u_idx = (a_r + lr) * m + (a_c + lc)
                        U[global_u_idx][col] += val
                        
            # Map local V indices (0..3) to global V indices
            # Local (lr, lc) -> global (b_r + lr, b_c + lc)
            # Global V index: (b_r + lr) * p + (b_c + lc)
            for lr in range(2):
                for lc in range(2):
                    local_idx = lr * 2 + lc
                    val = V_s[t][local_idx]
                    if val != 0:
                        global_v_idx = (b_r + lr) * p + (b_c + lc)
                        V[global_v_idx][col] += val
                        
            # Map local W indices (0..3) to global W indices
            # Local (lr, lc) -> global (c_r + lr, c_c + lc)
            # Global W index: (c_c + lc) * n + (c_r + lr)
            # Note: W[k*n+i, r] where k is col, i is row.
            # So index is k * n + i.
            for lr in range(2):
                for lc in range(2):
                    local_idx = lr * 2 + lc
                    val = W_s[t][local_idx]
                    if val != 0:
                        global_w_idx = (c_c + lc) * n + (c_r + lr)
                        W[global_w_idx][col] += val
                        
            col += 1
            
            if time.time() > deadline:
                # If we run out of time, return what we have (might be incomplete)
                # But we are just filling zeros, so it's safe to return partial?
                # No, better to finish quickly as it's just 56 cols.
                pass

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
