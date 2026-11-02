import time
import numpy as np

DIMENSIONS = [4, 5, 6]
n, m, p = DIMENSIONS

# Strassen steps for 2x2x2
# Each step is defined by:
# (u_indices, u_vals, v_indices, v_vals, w_indices, w_vals)
# u_indices: (i, j) in 0..1
# v_indices: (j, k) in 0..1
# w_indices: (i, k) in 0..1
# Note: In the global tensor sum, the index j in U and V must match for the contraction to be non-zero?
# No, the tensor is T[i, j, k]. U[i, j, r], V[j, k, r], W[k, i, r].
# The Strassen decomposition is for the map A, B -> C.
# C_ik = sum_j A_ij B_jk.
# The tensor element is 1.
# The decomposition is sum_r u_r(i, j) v_r(j, k) w_r(k, i) = 1.
#
# Strassen Step 1: M1 = (A00+A11)(B00+B11)
# Contributes to:
# C00: +1
# C11: +1
# (Check formula: C00 = M1 + M4 - M5 + M7, C11 = M1 - M2 + M3 + M6)
# So W00 += 1, W11 += 1.
#
# Step 2: M2 = (A10+A11)B00
# Contributes to:
# C10: +1
# C11: -1
#
# Step 3: M3 = A00(B01-B11)
# Contributes to:
# C01: +1
# C11: +1
#
# Step 4: M4 = A11(B10-B00)
# Contributes to:
# C00: +1
# C10: +1
#
# Step 5: M5 = (A00+A01)B11
# Contributes to:
# C00: -1
# C01: +1
#
# Step 6: M6 = (A10-A00)(B00+B01)
# Contributes to:
# C11: +1
#
# Step 7: M7 = (A01-A11)(B10+B11)
# Contributes to:
# C00: +1

def get_strassen_steps():
    # Returns list of 7 tuples: (U_dict, V_dict, W_dict)
    # Keys are (i, j), (j, k), (i, k)
    steps = []
    
    # Step 1
    steps.append((
        {(0,0): 1, (1,1): 1}, 
        {(0,0): 1, (1,1): 1}, 
        {(0,0): 1, (1,1): 1}
    ))
    # Step 2
    steps.append((
        {(1,0): 1, (1,1): 1}, 
        {(0,0): 1}, 
        {(1,0): 1, (1,1): -1}
    ))
    # Step 3
    steps.append((
        {(0,0): 1}, 
        {(0,1): 1, (1,1): -1}, 
        {(0,1): 1, (1,1): 1}
    ))
    # Step 4
    steps.append((
        {(1,1): 1}, 
        {(1,0): 1, (0,0): -1}, 
        {(0,0): 1, (1,0): 1}
    ))
    # Step 5
    steps.append((
        {(0,0): 1, (0,1): 1}, 
        {(1,1): 1}, 
        {(0,0): -1, (0,1): 1}
    ))
    # Step 6
    steps.append((
        {(1,0): 1, (0,0): -1}, 
        {(0,0): 1, (0,1): 1}, 
        {(1,1): 1}
    ))
    # Step 7
    steps.append((
        {(0,1): 1, (1,1): -1}, 
        {(1,0): 1, (1,1): 1}, 
        {(0,0): 1}
    ))
    
    return steps

STRASSEN_STEPS = get_strassen_steps()

def construct():
    # Total rank calculation
    # Term 1 (j=0,1): 2 A-blocks * 3 B-blocks * 7 steps = 42
    # Term 2 (j=2,3): 2 A-blocks * 3 B-blocks * 7 steps = 42
    # Term 3 (j=4): 1 rank
    rank_total = 42 + 42 + 1
    
    U = [[0.0] * rank_total for _ in range(n * m)]
    V = [[0.0] * rank_total for _ in range(m * p)]
    W = [[0.0] * rank_total for _ in range(p * n)]
    
    current_col = 0
    
    # Helper to add a Strassen step
    def add_strassen_block(global_i_offset, global_j_offset, global_k_offset, col_idx):
        for step in STRASSEN_STEPS:
            u_dict, v_dict, w_dict = step
            for (i, j), val in u_dict.items():
                gi = global_i_offset + i
                gj = global_j_offset + j
                U[gi * m + gj][col_idx] += val
            
            for (j, k), val in v_dict.items():
                gj = global_j_offset + j
                gk = global_k_offset + k
                V[gj * p + gk][col_idx] += val
            
            for (i, k), val in w_dict.items():
                gi = global_i_offset + i
                gk = global_k_offset + k
                W[gk * n + gi][col_idx] += val

    # Process Block 1: j in {0, 1}
    for i_block in range(2): # 0, 1
        for k_block in range(3): # 0, 1, 2
            for s in range(7):
                col = current_col + s
                add_strassen_block(i_block * 2, 0, k_block * 2, col)
            current_col += 7
            
    # Process Block 2: j in {2, 3}
    for i_block in range(2): # 0, 1
        for k_block in range(3): # 0, 1, 2
            for s in range(7):
                col = current_col + s
                add_strassen_block(i_block * 2, 2, k_block * 2, col)
            current_col += 7
            
    # Process Block 3: j = 4
    # Rank 1: U[i, 4] = 1, V[4, k] = 1, W[k, i] = 1
    col = current_col
    for i in range(4):
        U[i * m + 4][col] = 1.0
    for k in range(6):
        V[4 * p + k][col] = 1.0
    for k in range(6):
        for i in range(4):
            W[k * n + i][col] = 1.0
            
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    start = time.time()
    res = construct()
    end = time.time()
    print(f"Time taken: {end - start}")
    # Basic verification print
    U = np.array(res['U'])
    V = np.array(res['V'])
    W = np.array(res['W'])
    
    # Check a few elements
    # T[i, j, k] should be 1
    # sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    
    errors = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(np.dot(U[i*m+j], V[j*p+k]), W[k*n+i])
                if abs(val - 1.0) > 1e-9:
                    errors += 1
    if errors == 0:
        print("Verification PASSED")
    else:
        print(f"Verification FAILED with {errors} errors")
