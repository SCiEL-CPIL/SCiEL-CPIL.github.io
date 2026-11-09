import numpy as np
import time

DIMENSIONS = [2, 4, 6]
n, m, p = DIMENSIONS

def _strassen_2x2():
    """
    Returns the 7 coefficient vectors for Strassen's 2x2 matrix multiplication.
    A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    C = [[C00, C01], [C10, C11]]
    
    M1 = (a+d)(e+h)
    M2 = (c+d)e
    M3 = a(f-h)
    M4 = d(g-e)
    M5 = (a+b)h
    M6 = (c-a)(e+f)
    M7 = (b-d)(g+h)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    """
    # A coefficients: order (a, b, c, d)
    A_coeffs = [
        [1, 0, 0, 1],  # M1
        [0, 0, 1, 1],  # M2
        [1, 0, 0, 0],  # M3
        [0, 0, 0, 1],  # M4
        [1, 1, 0, 0],  # M5
        [-1, 0, 1, 0], # M6
        [0, -1, 0, 1], # M7
    ]
    
    # B coefficients: order (e, f, g, h)
    B_coeffs = [
        [1, 0, 0, 1],  # M1
        [1, 0, 0, 0],  # M2
        [0, 1, 0, -1], # M3
        [-1, 0, 1, 0], # M4
        [0, 0, 0, 1],  # M5
        [1, 1, 0, 0],  # M6
        [0, 0, 1, 1],  # M7
    ]
    
    # C coefficients: order (C00, C01, C10, C11)
    C_coeffs = [
        [1, 0, 0, 1],  # M1
        [0, 0, 1, -1], # M2
        [0, 1, 0, 1],  # M3
        [1, 0, 1, 0],  # M4
        [-1, 1, 0, 0], # M5
        [0, 0, 0, 1],  # M6
        [1, 0, 0, 0],  # M7
    ]
    
    return A_coeffs, B_coeffs, C_coeffs

def construct():
    """
    Returns a tensor decomposition for matrix multiplication of size (n, m) x (m, p).
    
    For n=2, m=4, p=6, we use a block Strassen decomposition.
    We view A as a 2x4 matrix (2 rows, 4 cols) and B as a 4x6 matrix (4 rows, 6 cols).
    
    We split A into two 2x2 blocks horizontally: A = [A_L, A_R]
    We split B into three 4x2 blocks horizontally: B = [B_1, B_2, B_3]
    
    C = A * B = A_L * B_1 + A_L * B_2 + A_L * B_3 + A_R * B_1 + A_R * B_2 + A_R * B_3
    
    Each product is a 2x2 * 2x2 multiplication, which can be done with Strassen (rank 7).
    Total rank = 6 * 7 = 42.
    
    The output factor follows the official trace convention W[k*n+i, r].
    U has shape (n*m, rank) = (8, 42)
    V has shape (m*p, rank) = (24, 42)
    W has shape (p*n, rank) = (12, 42)
    """
    start_time = time.time()
    deadline = start_time + 140.0
    
    A_coeffs, B_coeffs, C_coeffs = _strassen_2x2()
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # A is 2x4. Rows 0,1. Cols 0,1 are A_L. Cols 2,3 are A_R.
    # B is 4x6. Rows 0,1,2,3. Cols 0,1 are B_1. Cols 2,3 are B_2. Cols 4,5 are B_3.
    
    # Block definitions:
    # A_L: rows 0-1, cols 0-1. A index: i*m + j.
    # A_R: rows 0-1, cols 2-3. A index: i*m + j.
    # B_1: rows 0-3, cols 0-1. B index: j*p + k.
    # B_2: rows 0-3, cols 2-3. B index: j*p + k.
    # B_3: rows 0-3, cols 4-5. B index: j*p + k.
    
    # We need to map the local 2x2 indices to global indices.
    # For A block (2x2): local (i, j) -> global (i, j_offset + j)
    # For B block (2x2): local (j, k) -> global (j_offset + j, k_offset + k)
    # Note: B is 4x6, but we are multiplying 2x2 (from A) by 4x6 (from B)?
    # Wait. A is 2x4. B is 4x6.
    # C = A * B.
    # C[i, k] = sum_j A[i, j] B[j, k].
    # j goes from 0 to 3.
    # So we sum over j.
    # We can split the sum into j=0,1 and j=2,3.
    # C = A[:, 0:2] * B[0:2, :] + A[:, 2:4] * B[2:4, :]
    # Let A_L = A[:, 0:2] (2x2), A_R = A[:, 2:4] (2x2).
    # Let B_T = B[0:2, :] (2x6), B_B = B[2:4, :] (2x6).
    # C = A_L * B_T + A_R * B_B.
    # Now split B_T into three 2x2 blocks: B_T1, B_T2, B_T3.
    # B_T1 = B[0:2, 0:2], B_T2 = B[0:2, 2:4], B_T3 = B[0:2, 4:6].
    # Similarly B_B1, B_B2, B_B3.
    
    # So we have 6 products of 2x2 matrices:
    # 1. A_L * B_T1
    # 2. A_L * B_T2
    # 3. A_L * B_T3
    # 4. A_R * B_B1
    # 5. A_R * B_B2
    # 6. A_R * B_B3
    
    # Each contributes to C.
    # C is 2x6.
    # Product 1 contributes to C[:, 0:2]
    # Product 2 contributes to C[:, 2:4]
    # Product 3 contributes to C[:, 4:6]
    # Product 4 contributes to C[:, 0:2]
    # Product 5 contributes to C[:, 2:4]
    # Product 6 contributes to C[:, 4:6]
    
    # So C[:, 0:2] = A_L * B_T1 + A_R * B_B1
    # C[:, 2:4] = A_L * B_T2 + A_R * B_B2
    # C[:, 4:6] = A_L * B_T3 + A_R * B_B3
    
    # We will iterate over these 6 instances.
    # For each instance, we use 7 Strassen terms.
    
    instances = [
        # (A_row_offset, A_col_offset, B_row_offset, B_col_offset)
        (0, 0, 0, 0), # A_L * B_T1
        (0, 0, 0, 2), # A_L * B_T2
        (0, 0, 0, 4), # A_L * B_T3
        (0, 2, 2, 0), # A_R * B_B1
        (0, 2, 2, 2), # A_R * B_B2
        (0, 2, 2, 4), # A_R * B_B3
    ]
    
    for inst in instances:
        a_i_off, a_j_off, b_j_off, b_k_off = inst
        
        for t in range(7):
            # U coefficients for A
            # Local A indices: (0,0), (0,1), (1,0), (1,1)
            # Global A indices: (a_i_off + i, a_j_off + j)
            # A global index: (a_i_off + i) * m + (a_j_off + j)
            
            # V coefficients for B
            # Local B indices: (0,0), (0,1), (1,0), (1,1)
            # Global B indices: (b_j_off + j, b_k_off + k)
            # B global index: (b_j_off + j) * p + (b_k_off + k)
            
            # W coefficients for C
            # Local C indices: (0,0), (0,1), (1,0), (1,1)
            # Global C indices: (i, b_k_off + k)
            # Note: C is 2x6. i is 0 or 1. k is 0..5.
            # W index: k * n + i = (b_k_off + k) * n + i
            
            for i in range(2):
                for j in range(2):
                    if A_coeffs[t][i * 2 + j] != 0:
                        U[(a_i_off + i) * m + (a_j_off + j)][col] += A_coeffs[t][i * 2 + j]
            
            for j in range(2):
                for k in range(2):
                    if B_coeffs[t][j * 2 + k] != 0:
                        V[(b_j_off + j) * p + (b_k_off + k)][col] += B_coeffs[t][j * 2 + k]
            
            for i in range(2):
                for k in range(2):
                    if C_coeffs[t][i * 2 + k] != 0:
                        W[(b_k_off + k) * n + i][col] += C_coeffs[t][i * 2 + k]
            
            col += 1
    
    # Verify correctness with a random test
    # A is 2x4, B is 4x6
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_test = A_test @ B_test
    
    # Reconstruct C from U, V, W
    # C[i, k] = sum_r W[k*n + i, r] * (sum_j U[i*m + j, r] * A[i, j]) * (sum_j V[j*p + k, r] * B[j, k])
    # Let's compute the bilinear form.
    # U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    # Flatten A to a, B to b.
    # a = A.flatten() (row major: A[0,0], A[0,1], ..., A[1,3])
    # b = B.flatten() (row major: B[0,0], B[0,1], ..., B[3,5])
    # U is (8, 42). U[i*m+j, r] multiplies a[i*m+j].
    # V is (24, 42). V[j*p+k, r] multiplies b[j*p+k].
    # W is (12, 42). W[k*n+i, r] is the coefficient for C[i,k].
    
    a_flat = A_test.flatten()
    b_flat = B_test.flatten()
    
    # u_vecs = U @ a_flat (shape: rank)
    # v_vecs = V @ b_flat (shape: rank)
    # c_flat = sum_r W[:, r] * u_vecs[r] * v_vecs[r]
    # c_flat is shape (p*n) = (12). Index k*n + i.
    
    u_vecs = np.array(U) @ a_flat
    v_vecs = np.array(V) @ b_flat
    c_flat = np.array(W).T @ (u_vecs * v_vecs)
    
    C_recon = c_flat.reshape(p, n).T # Reshape to (n, p)
    
    if not np.allclose(C_test, C_recon):
        # Fallback to schoolbook if verification fails
        rank_school = n * m * p
        U_school = [[0] * rank_school for _ in range(n * m)]
        V_school = [[0] * rank_school for _ in range(m * p)]
        W_school = [[0] * rank_school for _ in range(p * n)]
        
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i * m + j][column] = 1
                    V_school[j * p + k][column] = 1
                    W_school[k * n + i][column] = 1
                    column += 1
        
        return {"U": U_school, "V": V_school, "W": W_school}
    
    return {"U": U, "V": V, "W": W}
