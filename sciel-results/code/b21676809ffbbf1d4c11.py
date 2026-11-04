import numpy as np
import time

def construct():
    n, m, p = 2, 4, 8
    start_time = time.time()
    deadline = start_time + 130

    # Target rank 56 (2 * 28)
    # 28 is the rank of 2x4x4 using 4 Strassen blocks (7 each)
    # 2x4x8 is two independent 2x4x4 multiplications
    
    rank = 56
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen factors for 2x2x2
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # U factors (rows of A):
    # 1: a+d, e+h
    # 2: c+d, e
    # 3: a, f-h
    # 4: d, g-e
    # 5: a+b, h
    # 6: c-a, e+f
    # 7: b-d, g+h
    
    # We need to map these to the 2x4x4 structure.
    # A (2x4) = [A1, A2] where A1, A2 are 2x2.
    # B (4x4) = [[B11, B12], [B21, B22]] where Bij are 2x2.
    # C (2x4) = [C1, C2] where C1, C2 are 2x2.
    # C1 = A1 B11 + A2 B21
    # C2 = A1 B12 + A2 B22
    
    # This requires 4 Strassen multiplications:
    # M11 = A1 B11
    # M21 = A2 B21
    # M12 = A1 B12
    # M22 = A2 B22
    
    # Let's build the factors for one 2x2x2 Strassen block.
    # Returns U_block (4x7), V_block (4x7), W_block (4x7)
    # Indices in U_block: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1) for 2x2 A
    # Indices in V_block: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1) for 2x2 B
    # Indices in W_block: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1) for 2x2 C
    
    def get_strassen_factors():
        U_b = np.zeros((4, 7))
        V_b = np.zeros((4, 7))
        W_b = np.zeros((4, 7))
        
        # 1: (a+d)(e+h)
        U_b[0, 0] = 1; U_b[3, 0] = 1
        V_b[0, 0] = 1; V_b[3, 0] = 1
        W_b[0, 0] = 1; W_b[1, 0] = 1; W_b[2, 0] = 1; W_b[3, 0] = 1 # C = M1 + ...
        
        # 2: (c+d)e
        U_b[2, 1] = 1; U_b[3, 1] = 1
        V_b[0, 1] = 1
        W_b[2, 1] = 1; W_b[3, 1] = 1
        
        # 3: a(f-h)
        U_b[0, 2] = 1
        V_b[1, 2] = 1; V_b[3, 2] = -1
        W_b[1, 2] = 1; W_b[3, 2] = 1
        
        # 4: d(g-e)
        U_b[3, 3] = 1
        V_b[2, 3] = 1; V_b[0, 3] = -1
        W_b[2, 3] = 1; W_b[3, 3] = 1
        
        # 5: (a+b)h
        U_b[0, 4] = 1; U_b[1, 4] = 1
        V_b[3, 4] = 1
        W_b[1, 4] = 1; W_b[3, 4] = 1
        
        # 6: (c-a)(e+f)
        U_b[2, 5] = 1; U_b[0, 5] = -1
        V_b[0, 5] = 1; V_b[1, 5] = 1
        W_b[0, 5] = 1; W_b[2, 5] = 1
        
        # 7: (b-d)(g+h)
        U_b[1, 6] = 1; U_b[3, 6] = -1
        V_b[2, 6] = 1; V_b[3, 6] = 1
        W_b[1, 6] = 1; W_b[3, 6] = 1
        
        # Combine W contributions for C
        # C00 = M1 - M4 - M6
        # C01 = M3 + M5
        # C10 = M2 + M6
        # C11 = M1 - M2 - M5 + M7
        
        # The W_b above is not just 1s. It needs to be the coefficients for the output.
        # Let's re-construct W_b properly.
        W_b = np.zeros((4, 7))
        
        # M1 contributes to all 4 outputs
        W_b[0, 0] = 1; W_b[1, 0] = 1; W_b[2, 0] = 1; W_b[3, 0] = 1
        
        # M2 contributes to C10, C11
        W_b[2, 1] = 1; W_b[3, 1] = 1
        
        # M3 contributes to C01, C11
        W_b[1, 2] = 1; W_b[3, 2] = 1
        
        # M4 contributes to C10, C11
        W_b[2, 3] = 1; W_b[3, 3] = 1
        
        # M5 contributes to C01, C11
        W_b[1, 4] = 1; W_b[3, 4] = 1
        
        # M6 contributes to C00, C10
        W_b[0, 5] = 1; W_b[2, 5] = 1
        
        # M7 contributes to C01, C11
        W_b[1, 6] = 1; W_b[3, 6] = 1
        
        # Now apply signs
        # C00 = M1 - M4 - M6
        W_b[0, 3] = -1 # M4
        W_b[0, 5] = -1 # M6
        
        # C01 = M3 + M5
        # (Already +1)
        
        # C10 = M2 + M6
        # (Already +1)
        
        # C11 = M1 - M2 - M5 + M7
        W_b[3, 1] = -1 # M2
        W_b[3, 4] = -1 # M5
        # M1 is +1, M7 is +1
        
        return U_b, V_b, W_b

    U_b, V_b, W_b = get_strassen_factors()
    
    # Map to 2x4x4
    # A indices: 0:(0,0), 1:(0,1), 2:(0,2), 3:(0,3), 4:(1,0), 5:(1,1), 6:(1,2), 7:(1,3)
    # B indices: 0:(0,0), 1:(0,1), 2:(0,2), 3:(0,3), 4:(1,0), 5:(1,1), 6:(1,2), 7:(1,3), 8:(2,0)...
    # Wait, B is 4x4.
    # B11 is B[0:2, 0:2] -> indices 0,1,4,5
    # B12 is B[0:2, 2:4] -> indices 2,3,6,7
    # B21 is B[2:4, 0:2] -> indices 8,9,12,13
    # B22 is B[2:4, 2:4] -> indices 10,11,14,15
    
    # A1 is A[0:2, 0:2] -> indices 0,1,4,5
    # A2 is A[0:2, 2:4] -> indices 2,3,6,7
    
    # C1 is C[0:2, 0:2] -> indices 0,1,4,5 (in 2x4 C, row 0,1 cols 0,1)
    # C2 is C[0:2, 2:4] -> indices 2,3,6,7
    
    # We have 4 blocks to compute:
    # 1. A1 * B11 -> C1_part1
    # 2. A2 * B21 -> C1_part2
    # 3. A1 * B12 -> C2_part1
    # 4. A2 * B22 -> C2_part2
    
    # Total rank 28.
    # We will fill U, V, W for the first 28 columns.
    
    def add_block(u_off, v_off, w_off, target_c_off):
        # u_off: offset in A (2x4) for the 2x2 block
        # v_off: offset in B (4x4) for the 2x2 block
        # w_off: offset in C (2x4) for the 2x2 block
        
        for k in range(7):
            # U
            for i in range(4):
                if U_b[i, k] != 0:
                    # Map local i (0,1,2,3) to global A index
                    # Local 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
                    # Global A is 2x4.
                    # If u_off is for A1 (cols 0,1):
                    #   local 0 -> global 0
                    #   local 1 -> global 1
                    #   local 2 -> global 4
                    #   local 3 -> global 5
                    # If u_off is for A2 (cols 2,3):
                    #   local 0 -> global 2
                    #   local 1 -> global 3
                    #   local 2 -> global 6
                    #   local 3 -> global 7
                    
                    if u_off == 0: # A1
                        g_i = [0, 1, 4, 5][i]
                    else: # A2
                        g_i = [2, 3, 6, 7][i]
                        
                    U[g_i, k] += U_b[i, k]
                    
            # V
            for i in range(4):
                if V_b[i, k] != 0:
                    # Map local i to global B index
                    # Local 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
                    # B11: 0,1,4,5
                    # B12: 2,3,6,7
                    # B21: 8,9,12,13
                    # B22: 10,11,14,15
                    
                    if v_off == 0: # B11
                        g_i = [0, 1, 4, 5][i]
                    elif v_off == 1: # B12
                        g_i = [2, 3, 6, 7][i]
                    elif v_off == 2: # B21
                        g_i = [8, 9, 12, 13][i]
                    else: # B22
                        g_i = [10, 11, 14, 15][i]
                        
                    V[g_i, k] += V_b[i, k]
                    
            # W
            for i in range(4):
                if W_b[i, k] != 0:
                    # Map local i to global C index
                    # C is 2x4.
                    # C1: 0,1,4,5
                    # C2: 2,3,6,7
                    
                    if w_off == 0: # C1
                        g_i = [0, 1, 4, 5][i]
                    else: # C2
                        g_i = [2, 3, 6, 7][i]
                        
                    W[g_i, k] += W_b[i, k]

    # Block 1: A1 B11 -> C1
    add_block(0, 0, 0, 0)
    # Block 2: A2 B21 -> C1
    # Note: W for C1 is sum of contributions from Block 1 and Block 2.
    # My add_block adds to W. So I need to be careful with column indices.
    # I used k in range(7) for all blocks, which is wrong.
    # I need separate columns for each block.
    
    # Reset U, V, W
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    # Block 1: A1 B11 -> C1
    for k in range(7):
        for i in range(4):
            if U_b[i, k] != 0:
                g_i = [0, 1, 4, 5][i]
                U[g_i, col+k] = U_b[i, k]
            if V_b[i, k] != 0:
                g_i = [0, 1, 4, 5][i]
                V[g_i, col+k] = V_b[i, k]
            if W_b[i, k] != 0:
                g_i = [0, 1, 4, 5][i]
                W[g_i, col+k] = W_b[i, k]
    col += 7
    
    # Block 2: A2 B21 -> C1
    for k in range(7):
        for i in range(4):
            if U_b[i, k] != 0:
                g_i = [2, 3, 6, 7][i]
                U[g_i, col+k] = U_b[i, k]
            if V_b[i, k] != 0:
                g_i = [8, 9, 12, 13][i]
                V[g_i, col+k] = V_b[i, k]
            if W_b[i, k] != 0:
                g_i = [0, 1, 4, 5][i] # C1
                W[g_i, col+k] = W_b[i, k]
    col += 7
    
    # Block 3: A1 B12 -> C2
    for k in range(7):
        for i in range(4):
            if U_b[i, k] != 0:
                g_i = [0, 1, 4, 5][i]
                U[g_i, col+k] = U_b[i, k]
            if V_b[i, k] != 0:
                g_i = [2, 3, 6, 7][i]
                V[g_i, col+k] = V_b[i, k]
            if W_b[i, k] != 0:
                g_i = [2, 3, 6, 7][i] # C2
                W[g_i, col+k] = W_b[i, k]
    col += 7
    
    # Block 4: A2 B22 -> C2
    for k in range(7):
        for i in range(4):
            if U_b[i, k] != 0:
                g_i = [2, 3, 6, 7][i]
                U[g_i, col+k] = U_b[i, k]
            if V_b[i, k] != 0:
                g_i = [10, 11, 14, 15][i]
                V[g_i, col+k] = V_b[i, k]
            if W_b[i, k] != 0:
                g_i = [2, 3, 6, 7][i] # C2
                W[g_i, col+k] = W_b[i, k]
    col += 7
    
    # Now we have rank 28 for 2x4x4.
    # We need 2x4x8.
    # B is 4x8. C is 2x8.
    # We can view this as two 2x4x4 multiplications.
    # B = [B_left, B_right]
    # C = [C_left, C_right]
    # C_left = A B_left
    # C_right = A B_right
    
    # The U factors are the same for both!
    # The V factors are different (B_left vs B_right).
    # The W factors are different (C_left vs C_right).
    
    # So we can reuse the U columns from the first 28.
    # But wait, the rank is the number of columns.
    # If we reuse U, we don't add new U columns?
    # No, the tensor decomposition is a sum of rank-1 terms.
    # Term r: u_r v_r^T w_r^T.
    # If we have 28 terms for the first half, and 28 for the second half,
    # and they are independent, the total rank is 56.
    # We cannot "share" the rank-1 terms unless the u, v, w are identical, which they are not (v and w differ).
    # So we just duplicate the structure for the second half.
    
    # Copy U, V, W for the second half (columns 28 to 55)
    # U is same
    U[:, 28:56] = U[:, 0:28]
    
    # V is shifted
    # B_left indices: 0-15
    # B_right indices: 16-31
    # B_left is 4x4. B_right is 4x4.
    # V[:, 0:28] corresponds to B_left.
    # We need V[:, 28:56] to correspond to B_right.
    # The pattern of non-zeros in V for B_left is the same as B_right, just shifted by 16.
    
    for k in range(28):
        for i in range(16):
            if V[i, k] != 0:
                V[i + 16, k + 28] = V[i, k]
                
    # W is shifted
    # C_left indices: 0-7 (2x4)
    # C_right indices: 8-15 (2x4)
    # W[:, 0:28] corresponds to C_left.
    # We need W[:, 28:56] to correspond to C_right.
    # Shift by 8.
    
    for k in range(28):
        for i in range(8):
            if W[i, k] != 0:
                W[i + 8, k + 28] = W[i, k]

    # Verify
    def verify(U, V, W):
        rank = U.shape[1]
        np.random.seed(123)
        A_test = np.random.rand(n, m)
        B_test = np.random.rand(m, p)
        C_expected = A_test @ B_test
        
        C_computed = np.zeros((n, p))
        for r_idx in range(rank):
            u_r = U[:, r_idx].reshape(n, m)
            v_r = V[:, r_idx].reshape(m, p)
            w_r = W[:, r_idx].reshape(p, n) # Convention W[k*n+i, r] -> w_r[j, i]
            for i in range(n):
                for j in range(p):
                    ua = np.dot(u_r[i, :], A_test[i, :])
                    vb = np.dot(v_r[:, j], B_test[:, j])
                    C_computed[i, j] += w_r[j, i] * ua * vb
        
        return np.allclose(C_computed, C_expected, atol=1e-10)

    if verify(U, V, W):
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    else:
        # Fallback to naive
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        r = 0
        for i in range(n):
            for a in range(m):
                for j in range(p):
                    U[i * m + a, r] = 1.0
                    V[a * p + j, r] = 1.0
                    W[j * n + i, r] = 1.0
                    r += 1
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
