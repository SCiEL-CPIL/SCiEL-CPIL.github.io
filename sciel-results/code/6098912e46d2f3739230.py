"""Matrix multiplication 2,4,4 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = 2, 4, 4
    
    # Use Strassen-based decomposition for 2x4x4 matrix multiplication.
    # Block structure:
    # A (2x4) = [A1 A2] where A1, A2 are 2x2 (columns 0-1 and 2-3)
    # B (4x4) = [B11 B12; B21 B22] where each is 2x2
    # C (2x4) = [A1*B11 + A2*B21, A1*B12 + A2*B22]
    #
    # For each 2x2 x 2x2 product, use Strassen (rank 7).
    # We have 4 such products: A1*B11, A1*B12, A2*B21, A2*B22
    # Total rank: 4 * 7 = 28
    
    # Strassen's 7 multiplications for 2x2x2:
    # For P (2x2) * Q (2x2) = R (2x2):
    # M1 = (P00 - P10) * (Q00 + Q01)
    # M2 = (P00 + P10) * Q10
    # M3 = P01 * (Q10 - Q11)
    # M4 = P11 * (Q00 + Q01)
    # M5 = (P00 - P01) * Q01
    # M6 = (P10 + P11) * (Q00 - Q10)
    # M7 = (P00 + P01) * (Q11 - Q00)
    #
    # R00 = M1 + M4 - M5 + M7
    # R01 = M3 + M5
    # R10 = M2 + M4
    # R11 = M1 - M2 + M3 + M6
    
    # Define Strassen coefficients for U, V, W
    # U corresponds to P entries, V to Q entries, W to R entries
    # P indices: 00, 01, 10, 11 -> U rows 0,1,2,3
    # Q indices: 00, 01, 10, 11 -> V rows 0,1,2,3
    # R indices: 00, 01, 10, 11 -> W rows 0,1,2,3
    
    strassen_U = np.array([
        [1, 1, 0, 0, 1, 0, 1],   # P00
        [0, 0, 1, 0, -1, 0, 1],  # P01
        [-1, 1, 0, 0, 0, 1, 0],  # P10
        [0, 0, 0, 1, 0, 1, 0],   # P11
    ])
    
    strassen_V = np.array([
        [1, 0, 0, 1, 0, 1, -1],  # Q00
        [1, 0, -1, 1, 1, 0, 0],  # Q01
        [0, 1, 1, 0, 0, -1, 0],  # Q10
        [0, 0, -1, 0, 0, 0, 1],  # Q11
    ])
    
    strassen_W = np.array([
        [1, 0, 0, 1, -1, 0, 1],  # R00
        [0, 0, 1, 0, 1, 0, 0],   # R01
        [0, 1, 0, 1, 0, 0, 0],   # R10
        [1, -1, 1, 0, 0, 1, 0],  # R11
    ])
    
    rank = 28
    U = np.zeros((n * m, rank))  # 8 x 28
    V = np.zeros((m * p, rank))  # 16 x 28
    W = np.zeros((p * n, rank))  # 8 x 28
    
    # Mapping:
    # A[i,j]: i in {0,1}, j in {0,1,2,3}
    #   A1 = A[:, 0:2], A2 = A[:, 2:4]
    #   A1 row index in U: i*2 + (j%2) for j in 0,1
    #   A2 row index in U: i*2 + (j%2) for j in 2,3
    #   U row for A[i,j]: i*4 + j  (since n*m = 8)
    #
    # B[j,k]: j in {0,1,2,3}, k in {0,1,2,3}
    #   B11 = B[0:2, 0:2], B12 = B[0:2, 2:4]
    #   B21 = B[2:4, 0:2], B22 = B[2:4, 2:4]
    #   V row for B[j,k]: j*4 + k  (since m*p = 16)
    #
    # C[i,k]: i in {0,1}, k in {0,1,2,3}
    #   C = [C_left, C_right] where C_left = A1*B11 + A2*B21, C_right = A1*B12 + A2*B22
    #   W row for C[i,k]: k*2 + i  (since p*n = 8, using W[k*n+i])
    
    # For each of the 4 block products, assign 7 columns
    # Product 0: A1 * B11 -> contributes to C[:, 0:2]
    # Product 1: A1 * B12 -> contributes to C[:, 2:4]
    # Product 2: A2 * B21 -> contributes to C[:, 0:2]
    # Product 3: A2 * B22 -> contributes to C[:, 2:4]
    
    for prod_idx in range(4):
        col_start = prod_idx * 7
        
        # Determine which A block and B block
        if prod_idx == 0:
            # A1 * B11
            a_block = 0  # A1: columns 0,1
            b_block = (0, 0)  # B11: rows 0,1, cols 0,1
            c_cols = (0, 1)  # C columns 0,1
        elif prod_idx == 1:
            # A1 * B12
            a_block = 0  # A1: columns 0,1
            b_block = (0, 1)  # B12: rows 0,1, cols 2,3
            c_cols = (2, 3)  # C columns 2,3
        elif prod_idx == 2:
            # A2 * B21
            a_block = 1  # A2: columns 2,3
            b_block = (1, 0)  # B21: rows 2,3, cols 0,1
            c_cols = (0, 1)  # C columns 0,1
        else:
            # A2 * B22
            a_block = 1  # A2: columns 2,3
            b_block = (1, 1)  # B22: rows 2,3, cols 2,3
            c_cols = (2, 3)  # C columns 2,3
        
        # Map Strassen U coefficients to actual U rows
        for u_row in range(4):
            # u_row: 0=P00, 1=P01, 2=P10, 3=P11
            i = u_row // 2  # 0 or 1
            j_local = u_row % 2  # 0 or 1
            j_global = a_block * 2 + j_local  # 0,1 for a_block=0; 2,3 for a_block=1
            u_actual = i * 4 + j_global  # row in U (n*m = 8)
            
            for r in range(7):
                U[u_actual, col_start + r] = strassen_U[u_row, r]
        
        # Map Strassen V coefficients to actual V rows
        for v_row in range(4):
            # v_row: 0=Q00, 1=Q01, 2=Q10, 3=Q11
            j_local = v_row // 2  # 0 or 1
            k_local = v_row % 2  # 0 or 1
            j_global = b_block[0] * 2 + j_local  # row in B
            k_global = b_block[1] * 2 + k_local  # col in B
            v_actual = j_global * 4 + k_global  # row in V (m*p = 16)
            
            for r in range(7):
                V[v_actual, col_start + r] = strassen_V[v_row, r]
        
        # Map Strassen W coefficients to actual W rows
        for w_row in range(4):
            # w_row: 0=R00, 1=R01, 2=R10, 3=R11
            i = w_row // 2  # 0 or 1
            k_local = w_row % 2  # 0 or 1
            k_global = c_cols[0] + k_local  # col in C
            w_actual = k_global * 2 + i  # row in W (p*n = 8, W[k*n+i])
            
            for r in range(7):
                W[w_actual, col_start + r] = strassen_W[w_row, r]
    
    # Verify the decomposition
    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # should equal 1 iff a=i*4+j, b=j*4+k, c=k*2+i for some valid i,j,k
    
    # Build the tensor and verify
    tensor = np.einsum('ar,br,cr->abc', U, V, W)
    
    # Expected: tensor[i*4+j, j*4+k, k*2+i] = 1 for all valid i,j,k
    # All other entries should be 0
    
    error = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                if abs(tensor[a_idx, b_idx, c_idx] - 1.0) > 1e-10:
                    error += abs(tensor[a_idx, b_idx, c_idx] - 1.0)
    
    # Check all other entries are zero
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                # Check if this is a valid (i,j,k) triple
                i = a // m
                j = a % m
                j2 = b // p
                k = b % p
                k2 = c // n
                i2 = c % n
                if i == i2 and j == j2 and k == k2:
                    continue  # This is a valid entry, already checked
                else:
                    if abs(tensor[a, b, c]) > 1e-10:
                        error += abs(tensor[a, b, c])
    
    if error > 1e-8:
        # Fallback to schoolbook if Strassen decomposition is incorrect
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
