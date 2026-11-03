"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import time

def _strassen_2x2_decomp():
    """
    Return the rank-7 Strassen decomposition for 2x2 matrix multiplication.
    
    For multiplying (2x2) A by (2x2) B to get (2x2) C:
    C[0,0] = M1 - M4 + M5 + M7
    C[0,1] = M3 + M5
    C[1,0] = M2 + M4
    C[1,1] = M1 - M3 + M6 + M7
    
    Where:
    M1 = (A00+A11)*(B00+B11)
    M2 = (A10+A11)*B00
    M3 = A00*(B01-B11)
    M4 = A11*(B10-B00)
    M5 = (A00+A01)*B11
    M6 = (A10-A00)*(B00+B01)
    M7 = (A01-A11)*(B10+B11)
    
    U index: i*2+j (row i, col j of A), so U[i*2+j, col]
    V index: j*2+k (row j, col k of B), so V[j*2+k, col]  
    W index: k*2+i (row k, col i of C), so W[k*2+i, col]
    """
    # 7 products for 2x2x2
    # Each product is a bilinear form: sum of U[i*2+j]*V[j*2+k]*W[k*2+i]
    
    # M1 = (A00+A11)*(B00+B11)
    # U: A00, A11; V: B00, B11; W: C00, C01, C10, C11
    m1_u = {0: 1, 3: 1}  # A[0,0]=idx0, A[1,1]=idx3
    m1_v = {0: 1, 3: 1}  # B[0,0]=idx0, B[1,1]=idx3
    m1_w = {0: 1, 1: 1, 2: 1, 3: 1}  # C[0,0]=0, C[0,1]=1, C[1,0]=2, C[1,1]=3
    
    # M2 = (A10+A11)*B00
    m2_u = {2: 1, 3: 1}  # A[1,0]=idx2, A[1,1]=idx3
    m2_v = {0: 1}  # B[0,0]=idx0
    m2_w = {2: 1, 3: 1}  # C[1,0]=2, C[1,1]=3
    
    # M3 = A00*(B01-B11)
    m3_u = {0: 1}  # A[0,0]=idx0
    m3_v = {1: 1, 3: -1}  # B[0,1]=idx1, B[1,1]=idx3
    m3_w = {1: 1, 3: -1}  # C[0,1]=1, C[1,1]=-1
    
    # M4 = A11*(B10-B00)
    m4_u = {3: 1}  # A[1,1]=idx3
    m4_v = {2: 1, 0: -1}  # B[1,0]=idx2, B[0,0]=idx0
    m4_w = {2: 1, 0: -1}  # C[1,0]=1, C[0,0]=-1
    
    # M5 = (A00+A01)*B11
    m5_u = {0: 1, 1: 1}  # A[0,0]=idx0, A[0,1]=idx1
    m5_v = {3: 1}  # B[1,1]=idx3
    m5_w = {0: 1, 1: 1}  # C[0,0]=1, C[0,1]=1
    
    # M6 = (A10-A00)*(B00+B01)
    m6_u = {2: 1, 0: -1}  # A[1,0]=idx2, A[0,0]=idx0
    m6_v = {0: 1, 1: 1}  # B[0,0]=idx0, B[0,1]=idx1
    m6_w = {3: 1}  # C[1,1]=1
    
    # M7 = (A01-A11)*(B10+B11)
    m7_u = {1: 1, 3: -1}  # A[0,1]=idx1, A[1,1]=idx3
    m7_v = {2: 1, 3: 1}  # B[1,0]=idx2, B[1,1]=idx3
    m7_w = {0: 1, 1: -1, 3: 1}  # C[0,0]=1, C[0,1]=-1, C[1,1]=1
    
    products = [
        (m1_u, m1_v, m1_w),
        (m2_u, m2_v, m2_w),
        (m3_u, m3_v, m3_w),
        (m4_u, m4_v, m4_w),
        (m5_u, m5_v, m5_w),
        (m6_u, m6_v, m6_w),
        (m7_u, m7_v, m7_w),
    ]
    
    # Build U (4x7), V (4x7), W (4x7) for 2x2x2
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    for col, (u_dict, v_dict, w_dict) in enumerate(products):
        for idx, val in u_dict.items():
            U[idx][col] += val
        for idx, val in v_dict.items():
            V[idx][col] += val
        for idx, val in w_dict.items():
            W[idx][col] += val
    
    return U, V, W

def _verify_2x2_decomp(U, V, W):
    """Verify the 2x2x2 decomposition is correct."""
    import numpy as np
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Check: sum_c U[i*2+j,c] * V[j*2+k,c] * W[k*2+i,c] should equal delta(i,j,k) 
    # where the "tensor" T[i,j,k] = 1 if this is a valid matmul term
    # Actually, the matmul tensor for 2x2x2: T[(i,j),(j,k),(k,i)] 
    # We need: for all i,j,k: sum_c U[i*2+j,c]*V[j*2+k,c]*W[k*2+i,c] = 1
    
    for i in range(2):
        for j in range(2):
            for k in range(2):
                val = 0
                for c in range(7):
                    val += U[i*2+j][c] * V[j*2+k][c] * W[k*2+i][c]
                if abs(val - 1) > 1e-10:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 100  # safety margin
    
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    # Get the base 2x2x2 Strassen decomposition (rank 7)
    U2, V2, W2 = _strassen_2x2_decomp()
    
    # Verify base decomposition
    if not _verify_2x2_decomp(U2, V2, W2):
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    # Build 4x4x4 decomposition from 2x2 blocks using rank 7*7 = 49
    # For 4x4x4, we use 2x2 block structure:
    # A is 4x4 with blocks A00, A01, A10, A11 each 2x2
    # B is 4x4 with blocks B00, B01, B10, B11 each 2x2
    # C is 4x4 with blocks C00, C01, C10, C11 each 2x2
    #
    # Using Strassen on the outer 2x2x2:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    #
    # Each Mi is a 2x2 matrix product, computed with rank 7
    # Total rank = 7*7 = 49
    
    rank_4x4 = 49
    U4 = [[0] * rank_4x4 for _ in range(16)]  # 4*4 = 16 rows
    V4 = [[0] * rank_4x4 for _ in range(16)]  # 4*4 = 16 rows
    W4 = [[0] * rank_4x4 for _ in range(16)]  # 4*4 = 16 rows
    
    # Outer Strassen structure: 7 products, each with 7 inner products
    outer_products = [
        # (A_coeffs, B_coeffs, C_coeffs)
        # Each coeff is a dict from block_index to scalar
        # Block indices: 00, 01, 10, 11 -> 0, 1, 2, 3
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 1: 1, 2: 1, 3: 1}),  # M1
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: 1}),  # M2
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: -1}),  # M3
        ({3: 1}, {2: 1, 0: -1}, {2: 1, 0: -1}),  # M4
        ({0: 1, 1: 1}, {3: 1}, {0: 1, 1: 1}),  # M5
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),  # M6
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1, 1: -1, 3: 1}),  # M7
    ]
    
    for outer_col, (a_coeffs, b_coeffs, c_coeffs) in enumerate(outer_products):
        for inner_col in range(7):
            col = outer_col * 7 + inner_col
            # For each inner product in the 2x2x2 Strassen:
            # U2[a_idx][inner_col] * V2[b_idx][inner_col] * W2[c_idx][inner_col]
            for a_block, a_val in a_coeffs.items():
                for b_block, b_val in b_coeffs.items():
                    for c_block, c_val in c_coeffs.items():
                        if a_val == 0 or b_val == 0 or c_val == 0:
                            continue
                        # A block a_block has rows a_block//2, cols a_block%2
                        # In 4x4, A[row_start:row_start+2, col_start:col_start+2]
                        a_row = a_block // 2
                        a_col = a_block % 2
                        b_row = b_block // 2
                        b_col = b_block % 2
                        c_row = c_block // 2
                        c_col = c_block % 2
                        
                        # For each element in the 2x2 block:
                        for di in range(2):
                            for dj in range(2):
                                # A element at (a_row*2+di, a_col*2+dj)
                                a_elem = (a_row * 2 + di) * 4 + (a_col * 2 + dj)
                                # B element at (b_row*2+di, b_col*2+dj) -- wait, need to think about indices
                                # B row = b_row*2+di, B col = b_col*2+dj
                                b_elem = (b_row * 2 + di) * 4 + (b_col * 2 + dj)
                                # C element at (c_row*2+dj, c_col*2+di) -- trace convention W[k*n+i]
                                # C row = c_row*2+dj, C col = c_col*2+di
                                # W index = c_col*2+di (col) * n + c_row*2+dj (row)? 
                                # W[k*n+i] where k is column of C, i is row of C
                                # So W index = (c_col*2+di)*4 + (c_row*2+dj)
                                c_elem = (c_col * 2 + di) * 4 + (c_row * 2 + dj)
                                
                                # The inner 2x2x2 decomposition:
                                # U2[di*2+dj][inner_col] for the A part
                                # V2[di*2+dj][inner_col] for the B part  
                                # W2[dj*2+di][inner_col] for the C part
                                u_val = a_val * U2[di * 2 + dj][inner_col]
                                v_val = b_val * V2[di * 2 + dj][inner_col]
                                w_val = c_val * W2[dj * 2 + di][inner_col]
                                
                                U4[a_elem][col] += u_val
                                V4[b_elem][col] += v_val
                                W4[c_elem][col] += w_val
    
    # Verify 4x4x4 decomposition
    def verify_4x4(U4, V4, W4):
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    val = 0
                    for c in range(49):
                        val += U4[i * 4 + j][c] * V4[j * 4 + k][c] * W4[k * 4 + i][c]
                    if abs(val - 1) > 1e-10:
                        return False
        return True
    
    if not verify_4x4(U4, V4, W4):
        # Fallback: use schoolbook for 4x4x4 part
        # For 4x4x8, just do schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    # Now build 4x4x8 from two 4x4x4 decompositions
    # C = A * B where B is 4x8, split into B1 (4x4) and B2 (4x4)
    # C1 = A * B1, C2 = A * B2
    # Total rank = 49 + 49 = 98
    
    rank_total = 98
    U_final = [[0] * rank_total for _ in range(16)]  # n*m = 16
    V_final = [[0] * rank_total for _ in range(32)]  # m*p = 32
    W_final = [[0] * rank_total for _ in range(32)]  # p*n = 32
    
    # First half: B1 (columns 0-3 of B)
    for i in range(16):
        for c in range(49):
            U_final[i][c] = U4[i][c]
    for j in range(16):  # B1 is rows 0-3, cols 0-3 of B, so B[j*4+k] for k=0..3
        for k in range(4):
            for c in range(49):
                V_final[j * 4 + k][c] = V4[j * 4 + k][c]
    for k in range(16):  # C1 is rows 0-3, cols 0-3, W index = k*4+i
        for i in range(4):
            for c in range(49):
                W_final[k * 4 + i][c] = W4[k * 4 + i][c]
    
    # Second half: B2 (columns 4-7 of B)
    for i in range(16):
        for c in range(49):
            U_final[i][49 + c] = U4[i][c]
    for j in range(16):  # B2 is rows 0-3, cols 4-7 of B
        for k in range(4):
            for c in range(49):
                V_final[j * 4 + (4 + k)][49 + c] = V4[j * 4 + k][c]
    for k in range(16):  # C2 is rows 4-7, cols 0-3
        for i in range(4):
            for c in range(49):
                W_final[(4 + k) * 4 + i][49 + c] = W4[k * 4 + i][c]
    
    # Verify the full 4x4x8 decomposition
    def verify_full(U, V, W, n, m, p):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for c in range(len(U[0])):
                        val += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(val - 1) > 1e-10:
                        return False
        return True
    
    if not verify_full(U_final, V_final, W_final, n, m, p):
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    elapsed = time.time() - start_time
    if elapsed > deadline - start_time:
        # Shouldn't happen, but safety
        pass
    
    return {"U": U_final, "V": V_final, "W": W_final}
# EVOLVE-BLOCK-END
