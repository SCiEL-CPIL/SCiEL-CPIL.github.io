import numpy as np
import time

DIMENSIONS = [4, 4, 9]

def _get_schoolbook_444_factors():
    """
    Returns U, V, W matrices for 4x4x4 matrix multiplication using schoolbook (Rank 64).
    """
    U = np.zeros((16, 64))
    V = np.zeros((16, 64))
    W = np.zeros((16, 64))
    for i in range(4):
        for j in range(4):
            r = i * 4 + j
            U[i * 4 + j, r] = 1
            V[j * 4 + i, r] = 1 # Wait, V index is j*p + k? No, V is for B (4x4).
            # B index: row j, col k. Flattened: j*4 + k.
            # In 4x4x4, k is the col of C, which is also col of B?
            # C_{i,k} = sum_j A_{i,j} B_{j,k}.
            # V index for B_{j,k} is j*4 + k.
            # W index for C_{k,i} is k*4 + i.
            V[j * 4 + i, r] = 1 # Here k=i? No.
            # The term is A_{i,j} B_{j,i} for C_{i,i}? No.
            # The term is A_{i,j} B_{j,k}.
            # We are decomposing the tensor.
            # For each (i,j,k), we have a term if i,j,k are valid.
            # In schoolbook, each (i,j) pair in A and (j,k) pair in B contributes to C_{i,k}.
            # Rank is n*m*p = 64.
            # Each rank-1 term corresponds to a specific (i,j,k).
            # U[i*4+j, r] = 1
            # V[j*4+k, r] = 1
            # W[k*4+i, r] = 1
            # r = (i*4 + j)*4 + k
            pass
    # Correct Schoolbook
    U = np.zeros((16, 64))
    V = np.zeros((16, 64))
    W = np.zeros((16, 64))
    r = 0
    for i in range(4):
        for j in range(4):
            for k in range(4):
                U[i * 4 + j, r] = 1
                V[j * 4 + k, r] = 1
                W[k * 4 + i, r] = 1
                r += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m  # 16
    dim_v = m * p  # 36
    dim_w = p * n  # 36
    
    # Use schoolbook for 4x4x4 parts (Rank 64)
    U4, V4, W4 = _get_schoolbook_444_factors()
    
    # Rank for 4x4x4 is 64
    R444 = 64
    
    # Total Rank = 64 + 64 + 16 = 144
    R_total = 144
    
    U = np.zeros((dim_u, R_total))
    V = np.zeros((dim_v, R_total))
    W = np.zeros((dim_w, R_total))
    
    # Block 1: B1 (cols 0-3), C1 (cols 0-3)
    U[:, 0:64] = U4
    
    for j in range(4):
        for k_prime in range(4):
            v_idx_444 = j * 4 + k_prime
            v_idx_main = j * 9 + k_prime
            V[v_idx_main, 0:64] = V4[v_idx_444, :]
            
    for k_prime in range(4):
        for i in range(4):
            w_idx_444 = k_prime * 4 + i
            w_idx_main = k_prime * 4 + i
            W[w_idx_main, 0:64] = W4[w_idx_444, :]
            
    # Block 2: B2 (cols 4-7), C2 (cols 4-7)
    U[:, 64:128] = U4
    
    for j in range(4):
        for k_prime in range(4):
            v_idx_444 = j * 4 + k_prime
            v_idx_main = j * 9 + (4 + k_prime)
            V[v_idx_main, 64:128] = V4[v_idx_444, :]
            
    for k_prime in range(4):
        for i in range(4):
            w_idx_444 = k_prime * 4 + i
            w_idx_main = (4 + k_prime) * 4 + i
            W[w_idx_main, 64:128] = W4[w_idx_444, :]
            
    # Block 3: B3 (col 8), C3 (col 8)
    # Schoolbook 4x4x1
    # Rank 16
    for i in range(4):
        for j in range(4):
            r = 128 + i * 4 + j
            u_idx = i * 4 + j
            v_idx = j * 9 + 8
            w_idx = 8 * 4 + i
            
            U[u_idx, r] = 1
            V[v_idx, r] = 1
            W[w_idx, r] = 1
            
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
