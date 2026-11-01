"""
Matrix multiplication tensor decomposition for M_{2,2,15}.

Strategy:
- Use block-Strassen construction.
- Split 15 columns into 7 blocks of 2 and 1 block of 1.
- Rank = 7*7 (Strassen) + 4 (Schoolbook for 2x2x1) = 53.
- Use integer arithmetic for exactness.
"""
import time
import numpy as np

DIMENSIONS = [2, 2, 15]

def _strassen_2x2x2_int():
    """
    Return integer Strassen decomposition for 2x2x2.
    U (4,7), V (4,7), W (4,7)
    """
    U = np.zeros((4, 7), dtype=np.int32)
    V = np.zeros((4, 7), dtype=np.int32)
    W = np.zeros((4, 7), dtype=np.int32)
    
    # M1 = (a00 + a11) * (b00 + b11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (a10 + a11) * b00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    
    # M3 = a00 * (b10 - b11)
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1
    
    # M4 = a11 * (b01 - b00)
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[0, 3] = -1; W[2, 3] = 1
    
    # M5 = (a00 + a01) * b11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = 1; W[1, 4] = 1
    
    # M6 = (a10 - a00) * (b00 + b10)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[2, 5] = 1
    W[3, 5] = 1
    
    # M7 = (a01 - a11) * (b01 + b11)
    U[1, 6] = 1; U[3, 6] = -1
    V[1, 6] = 1; V[3, 6] = 1
    W[0, 6] = -1
    
    return U, V, W

def _schoolbook_2x2x1_int():
    """
    Return integer schoolbook decomposition for 2x2x1.
    U (4,4), V (2,4), W (2,4)
    """
    U = np.zeros((4, 4), dtype=np.int32)
    V = np.zeros((2, 4), dtype=np.int32)
    W = np.zeros((2, 4), dtype=np.int32)
    
    # 4 products: a00*b0, a01*b0, a10*b0, a11*b0? No.
    # C0 = a00*b0 + a01*b1
    # C1 = a10*b0 + a11*b1
    # Products:
    # P1 = a00*b0 -> C0
    # P2 = a01*b1 -> C0
    # P3 = a10*b0 -> C1
    # P4 = a11*b1 -> C1
    
    U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
    U[1, 1] = 1; V[1, 1] = 1; W[0, 1] = 1
    U[2, 2] = 1; V[0, 2] = 1; W[1, 2] = 1
    U[3, 3] = 1; V[1, 3] = 1; W[1, 3] = 1
    
    return U, V, W

def _construct_53():
    n, m, p = 2, 2, 15
    U_s, V_s, W_s = _strassen_2x2x2_int()
    U_1, V_1, W_1 = _schoolbook_2x2x1_int()
    
    total_rank = 7 * 7 + 4
    U = np.zeros((n * m, total_rank), dtype=np.int32)
    V = np.zeros((m * p, total_rank), dtype=np.int32)
    W = np.zeros((p * n, total_rank), dtype=np.int32)
    
    col = 0
    k_offset = 0
    
    # 7 blocks of 2
    for _ in range(7):
        for r in range(7):
            c = col + r
            # U
            U[:, c] = U_s[:, r]
            # V
            V[0 * p + k_offset, c] = V_s[0, r]
            V[0 * p + k_offset + 1, c] = V_s[1, r]
            V[1 * p + k_offset, c] = V_s[2, r]
            V[1 * p + k_offset + 1, c] = V_s[3, r]
            # W
            W[k_offset * n + 0, c] = W_s[0, r]
            W[k_offset * n + 1, c] = W_s[1, r]
            W[(k_offset + 1) * n + 0, c] = W_s[2, r]
            W[(k_offset + 1) * n + 1, c] = W_s[3, r]
        col += 7
        k_offset += 2
        
    # 1 block of 1
    k_last = 14
    for r in range(4):
        c = col + r
        U[:, c] = U_1[:, r]
        V[0 * p + k_last, c] = V_1[0, r]
        V[1 * p + k_last, c] = V_1[1, r]
        W[k_last * n + 0, c] = W_1[0, r]
        W[k_last * n + 1, c] = W_1[1, r]
        
    return U, V, W

def _verify(U, V, W, n, m, p):
    # Vectorized verification
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We check if T equals the multiplication tensor.
    # Since dimensions are small, we can just compute the full tensor and compare.
    
    # Reshape to tensors
    # U is (4, R), V is (30, R), W is (30, R)
    # This is hard to vectorize directly for the sum over r.
    # Loop over r is fast enough for R=53.
    
    T = np.zeros((n, m, p), dtype=np.int32)
    for r in range(U.shape[1]):
        u_col = U[:, r]
        v_col = V[:, r]
        w_col = W[:, r]
        
        # u_col is (4,), v_col is (30,), w_col is (30,)
        # Reshape u_col to (2,2)
        u_mat = u_col.reshape(2, 2)
        # Reshape v_col to (2, 15)
        v_mat = v_col.reshape(2, 15)
        # Reshape w_col to (15, 2)
        w_mat = w_col.reshape(15, 2)
        
        # Contribution to T[i,j,k] is u_mat[i,j] * v_mat[j,k] * w_mat[k,i]
        # This is element-wise product summed over i,j,k? No.
        # T[i,j,k] += u_mat[i,j] * v_mat[j,k] * w_mat[k,i]
        
        # We can compute this using broadcasting
        # u_mat[i,j] * v_mat[j,k] * w_mat[k,i]
        # Sum over i,j,k? No, we are adding to T[i,j,k] for each r.
        # So for a fixed r, the contribution is a tensor T_r[i,j,k] = u_mat[i,j] * v_mat[j,k] * w_mat[k,i]
        
        # u_mat[i,j] shape (2,2)
        # v_mat[j,k] shape (2,15)
        # w_mat[k,i] shape (15,2)
        
        # We need T_r[i,j,k] = u_mat[i,j] * v_mat[j,k] * w_mat[k,i]
        
        # Let's construct it:
        # tmp1[i,j,k] = u_mat[i,j] * v_mat[j,k]
        # tmp2[i,j,k] = tmp1[i,j,k] * w_mat[k,i]
        
        # u_mat[i,j] * v_mat[j,k]
        # u_mat is (2,2). v_mat is (2,15).
        # We need to multiply u_mat[i,j] with v_mat[j,k].
        # This is a batched multiplication?
        # u_mat[i,j] depends on i,j. v_mat[j,k] depends on j,k.
        # So for each j, we have a vector u_mat[:,j] (2x1) and a vector v_mat[j,:] (1x15).
        # The product is an outer product? No.
        # u_mat[i,j] * v_mat[j,k] is a (2,15) matrix for each j?
        # No, i and k are free indices.
        # Let's just use loops for i,j,k since 2*2*15=60 is small.
        
        for i in range(2):
            for j in range(2):
                for k in range(15):
                    T[i,j,k] += u_mat[i,j] * v_mat[j,k] * w_mat[k,i]
                    
    # Check if T is the identity-like tensor
    # T[i,j,k] should be 1 if i==i' and j==j' and k==k'?
    # The tensor for matrix multiplication C = AB is:
    # C_ik = sum_j A_ij B_jk
    # So T[i,j,k] = 1 if we consider the basis elements?
    # The tensor T is defined by T[i,j,k] = 1 if the term A_ij B_jk contributes to C_ik.
    # Yes, T[i,j,k] = 1 for all i,j,k?
    # No. The tensor is T[i,j,k] = 1 if the multiplication involves A_ij and B_jk to produce C_ik.
    # In the standard basis, the tensor is 1 for all i,j,k?
    # Let's check: C_ik = sum_j A_ij B_jk.
    # The coefficient of A_ij B_jk in C_ik is 1.
    # So T[i,j,k] = 1 for all i,j,k.
    
    # Wait, is that right?
    # Yes, the matrix multiplication tensor is all 1s.
    
    if not np.all(T == 1):
        return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    U, V, W = _construct_53()
    
    if _verify(U, V, W, n, m, p):
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
        
    # Fallback to schoolbook 60
    rank = 60
    U_s = np.zeros((4, 60), dtype=np.int32)
    V_s = np.zeros((30, 60), dtype=np.int32)
    W_s = np.zeros((30, 60), dtype=np.int32)
    col = 0
    for i in range(2):
        for j in range(2):
            for k in range(15):
                U_s[i*2+j, col] = 1
                V_s[j*15+k, col] = 1
                W_s[k*2+i, col] = 1
                col += 1
                
    return {
        "U": U_s.tolist(),
        "V": V_s.tolist(),
        "W": W_s.tolist()
    }
