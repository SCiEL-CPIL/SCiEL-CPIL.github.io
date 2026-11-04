"""Program for matrix multiplication 4x4x8 tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START

def construct():
    """Construct a rank-98 tensor decomposition for 4x4x8 matrix multiplication
    using recursive Strassen algorithm (Strassen-7 for 4x4x4, duplicated for 4x4x8)."""
    t0 = time.time()
    deadline = t0 + 140.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    # Strassen 2x2x2 factors (7 terms)
    # Each term has U, V, W as 2x2 matrices with entries in {-1, 0, 1}
    # Indexing: U[row, col] where row = i*2+j (input A), col = term_index
    #           V[row, col] where row = j*2+k (input B), col = term_index
    #           W[row, col] where row = k*2+i (output C), col = term_index
    
    # Strassen formulas:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A10)B11
    # M6 = (A11 - A10)(B00 + B01)
    # M7 = (A00 - A11)(B11 + B10)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For each Strassen term, define U (2x2), V (2x2), W (2x2)
    # U[i,j] = coefficient of A[i//2][i%2] in the first factor
    # V[j,k] = coefficient of B[j//2][j%2] in the second factor
    # W[k,i] = coefficient of C[k//2][k%2] in the output
    
    strassen_U = []  # List of 7 (2,2) matrices
    strassen_V = []
    strassen_W = []
    
    # Term 1: M1 = (A00+A11)(B00+B11)
    # First factor: A00 + A11 -> U[:, 0] = [1,1] for rows 0,1 (A00, A11)
    # Actually let's be more careful. 
    # A is 2x2: A[0][0]=A00, A[0][1]=A01, A[1][0]=A10, A[1][1]=A11
    # B is 2x2: B[0][0]=B00, B[0][1]=B01, B[1][0]=B10, B[1][1]=B11
    # C is 2x2: C[0][0]=C00, C[0][1]=C01, C[1][0]=C10, C[1][1]=C11
    
    # For tensor decomposition, we need:
    # U[i, term] where i indexes A (i = row*2+col, so i=0:A00, i=1:A01, i=2:A10, i=3:A11)
    # V[j, term] where j indexes B (j = row*2+col, so j=0:B00, j=1:B01, j=2:B10, j=3:B11)
    # W[k, term] where k indexes C (k = col*2+row? or row*2+col?)
    
    # The task says: W[k*n+i, r] for 4x4x8 case
    # For 2x2x2: W[k*2+i, r] where k is column of C, i is row of C
    # So W index = k*2 + i, meaning C[i][k] is at index k*2+i
    
    # Let me redefine:
    # A[i][j] for i=0,1 and j=0,1. Index in U: i*2+j
    # B[j][k] for j=0,1 and k=0,1. Index in V: j*2+k  
    # C[i][k] for i=0,1 and k=0,1. Index in W: k*2+i (as per trace convention W[k*n+i])
    
    # Strassen term 1: M1 = (A00+A11)(B00+B11)
    # U1: A00 has coeff 1, A11 has coeff 1, A01 and A10 have coeff 0
    # U1 = [1, 0, 0, 1] (for A indices 0,1,2,3 = A00,A01,A10,A11)
    # V1: B00 has coeff 1, B11 has coeff 1, B01 and B10 have coeff 0
    # V1 = [1, 0, 0, 1] (for B indices 0,1,2,3 = B00,B01,B10,B11)
    # W1: M1 contributes to C00 (+1) and C11 (-1)
    # C00 is at W index k=0,i=0 -> 0*2+0 = 0
    # C11 is at W index k=1,i=1 -> 1*2+1 = 3
    # W1 = [1, 0, 0, -1]
    
    U1 = [1, 0, 0, 1]
    V1 = [1, 0, 0, 1]
    W1 = [1, 0, 0, -1]
    
    # Term 2: M2 = (A10+A11)B00
    # U2: A10 coeff 1, A11 coeff 1, others 0 -> [0, 0, 1, 1]
    # V2: B00 coeff 1, others 0 -> [1, 0, 0, 0]
    # W2: M2 contributes to C10 (+1) and C11 (-1)
    # C10 at k=0,i=1 -> 0*2+1 = 1
    # C11 at k=1,i=1 -> 1*2+1 = 3
    # W2 = [0, 1, 0, -1]
    
    U2 = [0, 0, 1, 1]
    V2 = [1, 0, 0, 0]
    W2 = [0, 1, 0, -1]
    
    # Term 3: M3 = A00(B01-B11)
    # U3: A00 coeff 1 -> [1, 0, 0, 0]
    # V3: B01 coeff 1, B11 coeff -1 -> [0, 1, 0, -1]
    # W3: M3 contributes to C01 (+1) and C11 (+1)
    # C01 at k=1,i=0 -> 1*2+0 = 2
    # C11 at k=1,i=1 -> 1*2+1 = 3
    # W3 = [0, 0, 1, 1]
    
    U3 = [1, 0, 0, 0]
    V3 = [0, 1, 0, -1]
    W3 = [0, 0, 1, 1]
    
    # Term 4: M4 = A11(B10-B00)
    # U4: A11 coeff 1 -> [0, 0, 0, 1]
    # V4: B10 coeff 1, B00 coeff -1 -> [-1, 0, 1, 0]
    # W4: M4 contributes to C00 (+1) and C10 (+1)
    # C00 at k=0,i=0 -> 0
    # C10 at k=0,i=1 -> 1
    # W4 = [1, 1, 0, 0]
    
    U4 = [0, 0, 0, 1]
    V4 = [-1, 0, 1, 0]
    W4 = [1, 1, 0, 0]
    
    # Term 5: M5 = (A00+A10)B11
    # U5: A00 coeff 1, A10 coeff 1 -> [1, 0, 1, 0]
    # V5: B11 coeff 1 -> [0, 0, 0, 1]
    # W5: M5 contributes to C00 (-1) and C01 (+1)
    # C00 at k=0,i=0 -> 0
    # C01 at k=1,i=0 -> 2
    # W5 = [-1, 0, 1, 0]
    
    U5 = [1, 0, 1, 0]
    V5 = [0, 0, 0, 1]
    W5 = [-1, 0, 1, 0]
    
    # Term 6: M6 = (A11-A10)(B00+B01)
    # U6: A11 coeff 1, A10 coeff -1 -> [0, 0, -1, 1]
    # V6: B00 coeff 1, B01 coeff 1 -> [1, 1, 0, 0]
    # W6: M6 contributes to C11 (+1)
    # C11 at k=1,i=1 -> 3
    # W6 = [0, 0, 0, 1]
    
    U6 = [0, 0, -1, 1]
    V6 = [1, 1, 0, 0]
    W6 = [0, 0, 0, 1]
    
    # Term 7: M7 = (A00-A11)(B11+B10)
    # U7: A00 coeff 1, A11 coeff -1 -> [1, 0, 0, -1]
    # V7: B11 coeff 1, B10 coeff 1 -> [0, 0, 1, 1]
    # W7: M7 contributes to C00 (+1)
    # C00 at k=0,i=0 -> 0
    # W7 = [1, 0, 0, 0]
    
    U7 = [1, 0, 0, -1]
    V7 = [0, 0, 1, 1]
    W7 = [1, 0, 0, 0]
    
    strassen_U = [U1, U2, U3, U4, U5, U6, U7]
    strassen_V = [V1, V2, V3, V4, V5, V6, V7]
    strassen_W = [W1, W2, W3, W4, W5, W6, W7]
    
    # Now build 4x4x4 decomposition using Strassen block recursion
    # A is 4x4, split into 2x2 blocks of 2x2 matrices
    # A[0:2, 0:2] = A00, A[0:2, 2:4] = A01, A[2:4, 0:2] = A10, A[2:4, 2:4] = A11
    # Similarly for B and C.
    
    # For the 4x4x4 case, we use Strassen at the block level:
    # 7 block-level terms, each requiring a 2x2x2 multiplication (rank 7)
    # Total rank: 7 * 7 = 49
    
    # Block-level Strassen coefficients (same pattern as 2x2x2 but at block level)
    # Block indices: 0=A00, 1=A01, 2=A10, 3=A11 (using i*2+j where i,j in {0,1})
    
    block_U = []  # 7 terms, each is 4-element vector (block coefficients)
    block_V = []
    block_W = []
    
    # Same formulas as before but at block level
    block_U.append([1, 0, 0, 1])   # M1: A00+A11
    block_U.append([0, 0, 1, 1])   # M2: A10+A11
    block_U.append([1, 0, 0, 0])   # M3: A00
    block_U.append([0, 0, 0, 1])   # M4: A11
    block_U.append([1, 0, 1, 0])   # M5: A00+A10
    block_U.append([0, 0, -1, 1])  # M6: A11-A10
    block_U.append([1, 0, 0, -1])  # M7: A00-A11
    
    block_V.append([1, 0, 0, 1])   # M1: B00+B11
    block_V.append([1, 0, 0, 0])   # M2: B00
    block_V.append([0, 1, 0, -1])  # M3: B01-B11
    block_V.append([-1, 0, 1, 0])  # M4: B10-B00
    block_V.append([0, 0, 0, 1])   # M5: B11
    block_V.append([1, 1, 0, 0])   # M6: B00+B01
    block_V.append([0, 0, 1, 1])   # M7: B11+B10
    
    block_W.append([1, 0, 0, -1])  # M1: C00+C11
    block_W.append([0, 1, 0, -1])  # M2: C10-C11
    block_W.append([0, 0, 1, 1])   # M3: C01+C11
    block_W.append([1, 1, 0, 0])   # M4: C00+C10
    block_W.append([-1, 0, 1, 0])  # M5: C01-C00
    block_W.append([0, 0, 0, 1])   # M6: C11
    block_W.append([1, 0, 0, 0])   # M7: C00
    
    # Now construct the 49-term 4x4x4 decomposition
    U_4x4 = []  # List of 49 vectors of length 16 (4*4)
    V_4x4 = []  # List of 49 vectors of length 16 (4*4)
    W_4x4 = []  # List of 49 vectors of length 16 (4*4)
    
    for s in range(7):  # Block-level Strassen term
        for t in range(7):  # Sub-term (2x2x2 Strassen)
            # Build U vector of length 16
            u_vec = [0] * 16
            for bi in range(4):  # Block index
                if block_U[s][bi] != 0:
                    for local_i in range(4):  # Local index within 2x2 block
                        if strassen_U[t][local_i] != 0:
                            # Block bi corresponds to rows bi//2, cols bi%2
                            # Local index local_i corresponds to row local_i//2, col local_i%2 within block
                            row = bi // 2
                            col = bi % 2
                            local_row = local_i // 2
                            local_col = local_i % 2
                            global_idx = (row * 2 + local_row) * 4 + (col * 2 + local_col)
                            u_vec[global_idx] += block_U[s][bi] * strassen_U[t][local_i]
            U_4x4.append(u_vec)
            
            # Build V vector of length 16
            v_vec = [0] * 16
            for bj in range(4):
                if block_V[s][bj] != 0:
                    for local_j in range(4):
                        if strassen_V[t][local_j] != 0:
                            row = bj // 2
                            col = bj % 2
                            local_row = local_j // 2
                            local_col = local_j % 2
                            global_idx = (row * 2 + local_row) * 4 + (col * 2 + local_col)
                            v_vec[global_idx] += block_V[s][bj] * strassen_V[t][local_j]
            V_4x4.append(v_vec)
            
            # Build W vector of length 16
            w_vec = [0] * 16
            for bk in range(4):
                if block_W[s][bk] != 0:
                    for local_k in range(4):
                        if strassen_W[t][local_k] != 0:
                            # Block bk: row bk//2, col bk%2
                            # Local index local_k: k*2+i where k is col, i is row within block
                            local_k_col = local_k // 2
                            local_k_row = local_k % 2
                            row = bk // 2
                            col = bk % 2
                            global_row = row * 2 + local_k_row
                            global_col = col * 2 + local_k_col
                            # W index = global_col * 4 + global_row
                            global_idx = global_col * 4 + global_row
                            w_vec[global_idx] += block_W[s][bk] * strassen_W[t][local_k]
            W_4x4.append(w_vec)
    
    # Now extend to 4x4x8 by duplicating the 4x4x4 decomposition
    # B is 4x8, split into B_left (4x4) and B_right (4x4)
    # C is 4x8, split into C_left (4x4) and C_right (4x4)
    # 
    # For the left half (cols 0-3 of B and C):
    #   Use U_4x4, V_4x4 (with B indices in range 0-15), W_4x4 (with C indices in range 0-15)
    # For the right half (cols 4-7 of B and C):
    #   Use U_4x4, V_4x4 shifted (B indices in range 16-31), W_4x4 shifted (C indices in range 16-31)
    
    rank = 98
    U = [[0] * rank for _ in range(16)]  # 4*4 = 16 rows
    V = [[0] * rank for _ in range(32)]  # 4*8 = 32 rows
    W = [[0] * rank for _ in range(32)]  # 8*4 = 32 rows
    
    # Left half: columns 0-47
    for term in range(49):
        for i in range(16):
            U[i][term] = U_4x4[term][i]
        for j in range(16):
            V[j][term] = V_4x4[term][j]
        for k in range(16):
            W[k][term] = W_4x4[term][k]
    
    # Right half: columns 49-97
    for term in range(49):
        for i in range(16):
            U[i][term + 49] = U_4x4[term][i]
        for j in range(16, 32):
            V[j][term + 49] = V_4x4[term][j - 16]
        for k in range(16, 32):
            W[k][term + 49] = W_4x4[term][k - 16]
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
