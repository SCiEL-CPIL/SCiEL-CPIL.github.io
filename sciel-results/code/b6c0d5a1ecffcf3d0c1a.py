import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    
    n, m, p = 2, 4, 6
    dim_A = n * m  # 8
    dim_B = m * p  # 24
    dim_C = p * n  # 12
    
    # Build the multiplication tensor T of shape (n*m, m*p, p*n)
    # T[a, b, c] where a = i*m+j (i in [0,n), j in [0,m))
    # b = j'*p+k (j' in [0,m), k in [0,p))
    # c = k'*n+i (k' in [0,p), i in [0,n))
    # T[a,b,c] = 1 if j == j' and i == i' and k == k'... 
    # Actually for matrix mult: C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor is T[i,j,k] = delta (it's 1 for each (i,j,k) triple)
    # In flattened form:
    # A index: i*m + j
    # B index: j*p + k  
    # C index: k*n + i
    # T[A_idx, B_idx, C_idx] = 1 if the indices are consistent, i.e.,
    # from A_idx we get i,j; from B_idx we get j',k; from C_idx we get k',i'
    # T = 1 if j==j' and i==i' and k==k'
    
    T = np.zeros((dim_A, dim_B, dim_C))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                T[a_idx, b_idx, c_idx] = 1.0
    
    # Known rank-42 construction using Strassen on 2x2 blocks
    rank = 42
    U = np.zeros((dim_A, rank))
    V = np.zeros((dim_B, rank))
    W = np.zeros((dim_C, rank))
    
    # Strassen 2x2x2 decomposition
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # U factors (for A, 2x2 block):
    U_s = np.array([
        [1, 0, 0, 1],   # M1: A11 + A22
        [0, 0, 1, 1],   # M2: A21 + A22
        [1, 0, 0, 0],   # M3: A11
        [0, 0, 0, 1],   # M4: A22
        [1, 1, 0, 0],   # M5: A11 + A12
        [-1, 0, 1, 0],  # M6: A21 - A11
        [0, 1, 0, -1]   # M7: A12 - A22
    ])
    
    # V factors (for B, 2x2 block):
    V_s = np.array([
        [1, 0, 0, 1],   # M1: B11 + B22
        [1, 0, 0, 0],   # M2: B11
        [0, 1, 0, -1],  # M3: B21 - B22
        [-1, 0, 1, 0],  # M4: B12 - B11
        [0, 0, 0, 1],   # M5: B22
        [1, 1, 0, 0],   # M6: B11 + B12
        [0, 0, 1, 1]    # M7: B21 + B22
    ])
    
    # W factors (for C, 2x2 block):
    W_s = np.array([
        [1, 0, 0, 1],   # M1: C11 += M1, C22 += M1
        [0, 0, 1, -1],  # M2: C21 += M2, C22 -= M2
        [0, 1, 0, 1],   # M3: C12 += M3, C22 += M3
        [1, 0, 1, 0],   # M4: C11 += M4, C12 += M4
        [-1, 1, 0, 0],  # M5: C11 -= M5, C12 += M5
        [0, 0, 0, 1],   # M6: C22 += M6
        [1, 0, 0, 0]    # M7: C11 += M7
    ])
    
    # A is 2x4: split into A_left (cols 0,1) and A_right (cols 2,3), each 2x2
    # B is 4x6: split into B_top (rows 0,1) and B_bot (rows 2,3), each 2x6
    # B_top and B_bot each split into 3 blocks of 2 cols: B_top = [Bt0|Bt1|Bt2], B_bot = [Bb0|Bb1|Bb2]
    
    col_offset = 0
    for cb in range(3):  # 3 column blocks of B
        # Block: A_left (2x2) * B_top_block (2x2)
        for t in range(7):
            # U: A_left indices
            for lr in range(2):
                for lc in range(2):
                    val = U_s[t][lr*2+lc]
                    if val != 0:
                        U[lr*m + lc, col_offset] += val
            
            # V: B_top_block indices (rows 0,1 of B, cols cb*2, cb*2+1)
            for lr in range(2):
                for lc in range(2):
                    val = V_s[t][lr*2+lc]
                    if val != 0:
                        V[lr*p + (cb*2 + lc), col_offset] += val
            
            # W: C indices (cols cb*2, cb*2+1 of C, rows 0,1)
            for lr in range(2):
                for lc in range(2):
                    val = W_s[t][lr*2+lc]
                    if val != 0:
                        W[(cb*2 + lc)*n + lr, col_offset] += val
            
            col_offset += 1
        
        # Block: A_right (2x2) * B_bot_block (2x2)
        for t in range(7):
            # U: A_right indices (cols 2,3 of A)
            for lr in range(2):
                for lc in range(2):
                    val = U_s[t][lr*2+lc]
                    if val != 0:
                        U[lr*m + (2 + lc), col_offset] += val
            
            # V: B_bot_block indices (rows 2,3 of B, cols cb*2, cb*2+1)
            for lr in range(2):
                for lc in range(2):
                    val = V_s[t][lr*2+lc]
                    if val != 0:
                        V[(2 + lr)*p + (cb*2 + lc), col_offset] += val
            
            # W: C indices (cols cb*2, cb*2+1 of C, rows 0,1)
            for lr in range(2):
                for lc in range(2):
                    val = W_s[t][lr*2+lc]
                    if val != 0:
                        W[(cb*2 + lc)*n + lr, col_offset] += val
            
            col_offset += 1
    
    # Verify the decomposition
    # C = U * diag(1) * V^T * W^T ... actually the decomposition is:
    # T = sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    # i.e., T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    
    T_reconstructed = np.einsum('ar,br,cr->abc', U, V, W)
    error = np.max(np.abs(T - T_reconstructed))
    
    if error > 1e-10:
        # Fallback to schoolbook rank 48
        rank = 48
        U = np.zeros((dim_A, rank))
        V = np.zeros((dim_B, rank))
        W = np.zeros((dim_C, rank))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i*m*p + j*p + k
                    a_idx = i*m + j
                    b_idx = j*p + k
                    c_idx = k*n + i
                    U[a_idx, r] = 1
                    V[b_idx, r] = 1
                    W[c_idx, r] = 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
