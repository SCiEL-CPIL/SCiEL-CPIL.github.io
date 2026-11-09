import numpy as np
import time

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Strassen coefficients for 2x2x2
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    
    # U (A) coefficients
    U_s = np.zeros((2, 2, 7))
    U_s[0,0,0] = 1; U_s[1,1,0] = 1
    U_s[1,0,1] = 1; U_s[1,1,1] = 1
    U_s[0,0,2] = 1
    U_s[1,1,3] = 1
    U_s[0,0,4] = 1; U_s[0,1,4] = 1
    U_s[1,0,5] = 1; U_s[0,0,5] = -1
    U_s[0,1,6] = 1; U_s[1,1,6] = -1
    
    # V (B) coefficients
    V_s = np.zeros((2, 2, 7))
    V_s[0,0,0] = 1; V_s[1,1,0] = 1
    V_s[0,0,1] = 1
    V_s[0,1,2] = 1; V_s[1,1,2] = -1
    V_s[1,0,3] = 1; V_s[0,0,3] = -1
    V_s[1,1,4] = 1
    V_s[0,0,5] = 1; V_s[0,1,5] = 1
    V_s[1,0,6] = 1; V_s[1,1,6] = 1
    
    # W (C) coefficients
    # C11 = M1 - M4 + M5 - M7
    # C12 = M3 + M7
    # C21 = M2 + M6
    # C22 = M1 - M3 + M4 + M6
    W_s = np.zeros((2, 2, 7))
    W_s[0,0,0] = 1; W_s[0,0,3] = -1; W_s[0,0,4] = 1; W_s[0,0,6] = -1
    W_s[0,1,2] = 1; W_s[0,1,6] = 1
    W_s[1,0,1] = 1; W_s[1,0,5] = 1
    W_s[1,1,0] = 1; W_s[1,1,2] = -1; W_s[1,1,3] = 1; W_s[1,1,5] = 1
    
    # Define blocks
    I_blocks = [(0, 1), (2, 3)]
    J_blocks = [(0, 1), (2, 3)]
    K_blocks = [(0, 1), (2, 3), (4, 5)]
    
    # Total rank
    num_strassen_blocks = len(I_blocks) * len(J_blocks) * len(K_blocks)
    rank = num_strassen_blocks * 7 + 1  # +1 for j=4 part
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col_idx = 0
    
    for i_s, i_e in I_blocks:
        for j_s, j_e in J_blocks:
            for k_s, k_e in K_blocks:
                # Map local indices to global
                # i_loc 0 -> i_s, 1 -> i_e (assuming i_e = i_s + 1)
                # But I_blocks are (0,1) and (2,3). So i_e is inclusive?
                # Let's assume the tuple is (start, end) where end is exclusive for range, 
                # but here we just have 2 indices.
                # Let's use the values directly.
                i_vals = [i_s, i_s + 1]
                j_vals = [j_s, j_s + 1]
                k_vals = [k_s, k_s + 1]
                
                for r in range(7):
                    # U
                    for i_loc in range(2):
                        for j_loc in range(2):
                            val = U_s[i_loc, j_loc, r]
                            if val != 0:
                                U[i_vals[i_loc] * m + j_vals[j_loc], col_idx] += val
                    # V
                    for j_loc in range(2):
                        for k_loc in range(2):
                            val = V_s[j_loc, k_loc, r]
                            if val != 0:
                                V[j_vals[j_loc] * p + k_vals[k_loc], col_idx] += val
                    # W
                    for i_loc in range(2):
                        for k_loc in range(2):
                            val = W_s[k_loc, i_loc, r] # Note: W_s is indexed [k_loc, i_loc]? 
                            # My W_s definition above:
                            # W_s[0,0] is C11 (i=0, k=0)
                            # So W_s[i_loc, k_loc] ?
                            # Let's re-verify W_s indices.
                            # C11 (i=0, k=0) -> W[0,0]
                            # C12 (i=0, k=1) -> W[0,1]
                            # C21 (i=1, k=0) -> W[1,0]
                            # C22 (i=1, k=1) -> W[1,1]
                            # So W_s[i_loc, k_loc] is correct for local i,k.
                            # But in the code above I defined:
                            # W_s[0,0,0] ... which is i=0, k=0.
                            # So yes, W_s[i_loc, k_loc, r].
                            if val != 0:
                                # Global index for W is k*n + i
                                W[k_vals[k_loc] * n + i_vals[i_loc], col_idx] += val
                    col_idx += 1
    
    # Add the j=4 part
    r_last = col_idx
    for i in range(n):
        U[i * m + 4, r_last] = 1
    for k in range(p):
        V[4 * p + k, r_last] = 1
    for i in range(n):
        for k in range(p):
            W[k * n + i, r_last] = 1
            
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
