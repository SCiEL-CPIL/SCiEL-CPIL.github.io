import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Schoolbook decomposition (Rank 56) as fallback
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    # Strassen decomposition for 2x2 * 2x2
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    strassen_U = np.array([
        [1, 0, 0, 0],  # a
        [1, 0, 1, 0],  # a+c
        [0, 0, 1, 1],  # c+d
        [0, 1, 0, -1], # b-d
        [1, 1, 0, 0],  # a+b
        [-1, 0, 1, 0], # c-a
        [0, 1, -1, 0], # b-c
    ], dtype=float)
    
    strassen_V = np.array([
        [1, 0, 0, 1],  # e+h
        [0, 0, 0, 1],  # h
        [1, 0, 0, 0],  # e
        [0, 0, 1, 1],  # g+h
        [1, 0, 0, 0],  # e
        [0, 1, 0, 1],  # f+h
        [1, 1, 0, 0],  # e+f
    ], dtype=float)
    
    strassen_W = np.array([
        [1, 0, 0, 1],  # C11 += 1, C22 += 1
        [0, 0, 1, -1], # C21 += 1, C22 -= 1
        [0, 0, 1, 1],  # C21 += 1, C22 += 1
        [1, 0, 1, 0],  # C11 += 1, C21 += 1
        [-1, 1, 0, 0], # C11 -= 1, C12 += 1
        [0, 0, 0, 1],  # C22 += 1
        [1, 0, 0, 0],  # C11 += 1
    ], dtype=float)
    
    # Build the full decomposition
    # A is 2x4: indices [A00, A01, A02, A03, A10, A11, A12, A13]
    # B is 4x7: indices [B00..B06, B10..B16, B20..B26, B30..B36]
    # C is 2x7: indices [C00..C06, C10..C16]
    
    # Split:
    # A1 = A[:, 0:2] = [[A00, A01], [A10, A11]]
    # A2 = A[:, 2:4] = [[A02, A03], [A12, A13]]
    
    # B_left = B[:, 0:4] (4x4)
    #   BTL = B[0:2, 0:2], BTR = B[0:2, 2:4]
    #   BBL = B[2:4, 0:2], BBR = B[2:4, 2:4]
    
    # B_right = B[:, 4:7] (4x3)
    #   BTL_r = B[0:2, 4:7], BBL_r = B[2:4, 4:7]
    
    # C_left = C[:, 0:4] = A*B_left
    #   C[:, 0:2] = A1*BTL + A2*BBL
    #   C[:, 2:4] = A1*BTR + A2*BBR
    # C_right = C[:, 4:7] = A*B_right
    #   C[:, 4:7] = A1*BTL_r + A2*BBL_r
    
    U_list = []
    V_list = []
    W_list = []
    
    def add_strassen_block(a_row_indices, a_col_indices, b_row_indices, b_col_indices, c_row_indices, c_col_indices):
        """Add a Strassen block for 2x2 * 2x2"""
        for r in range(7):
            u = np.zeros(8)
            v = np.zeros(28)
            w = np.zeros(14)
            
            # Map Strassen U (4-dim) to full A (8-dim)
            for idx, val in enumerate(strassen_U[r]):
                # idx 0 -> A[0,0], idx 1 -> A[0,1], idx 2 -> A[1,0], idx 3 -> A[1,1]
                row = a_row_indices[idx // 2]
                col = a_col_indices[idx % 2]
                u[row * 4 + col] += val
            
            # Map Strassen V (4-dim) to full B (28-dim)
            for idx, val in enumerate(strassen_V[r]):
                row = b_row_indices[idx // 2]
                col = b_col_indices[idx % 2]
                v[row * 7 + col] += val
            
            # Map Strassen W (4-dim) to full C (14-dim)
            for idx, val in enumerate(strassen_W[r]):
                row = c_row_indices[idx // 2]
                col = c_col_indices[idx % 2]
                w[row * 7 + col] += val
            
            U_list.append(u)
            V_list.append(v)
            W_list.append(w)
    
    # C[:, 0:2] = A1*BTL + A2*BBL
    # A1*BTL: A rows 0-1, cols 0-1; B rows 0-1, cols 0-1; C rows 0-1, cols 0-1
    add_strassen_block([0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1])
    # A2*BBL: A rows 0-1, cols 2-3; B rows 2-3, cols 0-1; C rows 0-1, cols 0-1
    add_strassen_block([0, 1], [2, 3], [2, 3], [0, 1], [0, 1], [0, 1])
    
    # C[:, 2:4] = A1*BTR + A2*BBR
    # A1*BTR: A rows 0-1, cols 0-1; B rows 0-1, cols 2-3; C rows 0-1, cols 2-3
    add_strassen_block([0, 1], [0, 1], [0, 1], [2, 3], [0, 1], [2, 3])
    # A2*BBR: A rows 0-1, cols 2-3; B rows 2-3, cols 2-3; C rows 0-1, cols 2-3
    add_strassen_block([0, 1], [2, 3], [2, 3], [2, 3], [0, 1], [2, 3])
    
    # C[:, 4:7] = A1*BTL_r + A2*BBL_r
    # BTL_r is B[0:2, 4:7] (2x3), BBL_r is B[2:4, 4:7] (2x3)
    # For 2x2 * 2x3, use schoolbook (12 multiplications per block)
    # A1*BTL_r: A[i,j]*B[j,k] for i in {0,1}, j in {0,1}, k in {4,5,6}
    for i in range(2):
        for j in range(2):
            for k in range(3):
                u = np.zeros(8)
                v = np.zeros(28)
                w = np.zeros(14)
                u[i * 4 + j] = 1  # A[i, j] where j is 0 or 1
                v[j * 7 + (4 + k)] = 1  # B[j, 4+k]
                w[i * 7 + (4 + k)] = 1  # C[i, 4+k]
                U_list.append(u)
                V_list.append(v)
                W_list.append(w)
    
    # A2*BBL_r: A[i,j]*B[j,k] for i in {0,1}, j in {2,3}, k in {4,5,6}
    for i in range(2):
        for j in range(2, 4):
            for k in range(3):
                u = np.zeros(8)
                v = np.zeros(28)
                w = np.zeros(14)
                u[i * 4 + j] = 1  # A[i, j]
                v[j * 7 + (4 + k)] = 1  # B[j, 4+k]
                w[i * 7 + (4 + k)] = 1  # C[i, 4+k]
                U_list.append(u)
                V_list.append(v)
                W_list.append(w)
    
    U = np.array(U_list)
    V = np.array(V_list)
    W = np.array(W_list)
    
    rank = U.shape[1]
    
    # Verify the decomposition
    if time.time() < deadline - 5:
        # Check: sum_r U[:,r] * V[:,r] * W[:,r] should equal the tensor
        # Tensor T[i,j,k] = delta_{i,j} for appropriate indices
        # Actually, the matrix multiplication tensor: T[i,j,k] = 1 if j == j (always), 
        # no: T[A_idx, B_idx, C_idx] = 1 if A_idx corresponds to (i,j), B_idx to (j,k), C_idx to (i,k)
        
        # Let's verify by checking a few random matrices
        np.random.seed(42)
        A_test = np.random.randn(2, 4)
        B_test = np.random.randn(4, 7)
        C_expected = A_test @ B_test
        
        # Compute C from decomposition
        C_computed = np.zeros((2, 7))
        for r in range(rank):
            # U[:,r] gives linear combo of A entries
            # V[:,r] gives linear combo of B entries
            # W[:,r] gives linear combo of C entries
            # The product is U[:,r] dot A_flat * V[:,r] dot B_flat added to C via W[:,r]
            
            A_flat = A_test.flatten()  # [A00, A01, A02, A03, A10, A11, A12, A13]
            B_flat = B_test.flatten()  # row-major: [B00..B06, B10..B16, ...]
            
            a_val = np.dot(U[:, r], A_flat)
            b_val = np.dot(V[:, r], B_flat)
            c_val = a_val * b_val
            
            # Add to C
            C_flat = W[:, r] * c_val
            C_computed += C_flat.reshape(2, 7)
        
        if not np.allclose(C_computed, C_expected, atol=1e-10):
            # Fall back to schoolbook
            return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    if time.time() > deadline - 2:
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
