import numpy as np
import time

DIMENSIONS = [4, 4, 9]

def _strassen_2x2x2():
    """
    Returns U, V, W for 2x2x2 matrix multiplication using Strassen's algorithm (rank 7).
    U: 4x7, V: 4x7, W: 4x7
    """
    rank = 7
    U = np.zeros((4, rank))
    V = np.zeros((4, rank))
    W = np.zeros((4, rank))
    
    # Indices:
    # U rows: 00, 01, 10, 11
    # V rows: 00, 01, 10, 11
    # W rows: 00, 01, 10, 11 (k*n+i, so 0*2+0, 0*2+1, 1*2+0, 1*2+1)
    
    # P1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1 # C00, C11
    
    # P2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1 # C10
    
    # P3 = A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1 # C01
    
    # P4 = A11(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1 # C00, C10
    
    # P5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1 # C00, C01
    
    # P6 = (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1 # C11
    
    # P7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1; W[1, 6] = 1 # C00, C01
    
    return U, V, W

def _kron_decomp(U1, V1, W1):
    """
    Computes Kronecker product decomposition and applies necessary row permutations
    to match standard row-major indexing for 4x4x4.
    """
    # U1 is 4x7, V1 is 4x7, W1 is 4x7
    # U2 is 4x7, V2 is 4x7, W2 is 4x7
    
    U_kron = np.kron(U1, U1) # 16x49
    V_kron = np.kron(V1, V1) # 16x49
    W_kron = np.kron(W1, W1) # 16x49
    
    # Permute U rows
    # Kron index: 8*i1 + 4*j1 + 2*i2 + j2
    # Target index: 8*i1 + 4*i2 + 2*j1 + j2
    perm_u = np.zeros(16, dtype=int)
    for i1 in range(2):
        for j1 in range(2):
            for i2 in range(2):
                for j2 in range(2):
                    kron_idx = 8*i1 + 4*j1 + 2*i2 + j2
                    target_idx = 8*i1 + 4*i2 + 2*j1 + j2
                    perm_u[target_idx] = kron_idx
    
    # Permute V rows
    # Kron index: 8*j1 + 4*k1 + 2*j2 + k2
    # Target index: 8*j1 + 4*j2 + 2*k1 + k2
    perm_v = np.zeros(16, dtype=int)
    for j1 in range(2):
        for k1 in range(2):
            for j2 in range(2):
                for k2 in range(2):
                    kron_idx = 8*j1 + 4*k1 + 2*j2 + k2
                    target_idx = 8*j1 + 4*j2 + 2*k1 + k2
                    perm_v[target_idx] = kron_idx
    
    # Permute W rows
    # Kron index: 8*k1 + 4*i1 + 2*k2 + i2
    # Target index: 8*k1 + 4*k2 + 2*i1 + i2
    perm_w = np.zeros(16, dtype=int)
    for k1 in range(2):
        for i1 in range(2):
            for k2 in range(2):
                for i2 in range(2):
                    kron_idx = 8*k1 + 4*i1 + 2*k2 + i2
                    target_idx = 8*k1 + 4*k2 + 2*i1 + i2
                    perm_w[target_idx] = kron_idx
    
    U4 = U_kron[perm_u, :]
    V4 = V_kron[perm_v, :]
    W4 = W_kron[perm_w, :]
    
    return U4, V4, W4

def _schoolbook_4x4x1():
    """
    Returns U, V, W for 4x4x1 matrix multiplication (schoolbook, rank 16).
    U: 16x16, V: 4x16, W: 4x16
    """
    U = np.eye(16)
    V = np.zeros((4, 16))
    W = np.zeros((4, 16))
    
    for i in range(4):
        for j in range(4):
            idx = i * 4 + j
            V[j, idx] = 1
            W[i, idx] = 1
            
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    U4, V4, W4 = _kron_decomp(*_strassen_2x2x2())
    U1, V1, W1 = _schoolbook_4x4x1()
    
    rank4 = 49
    rank1 = 16
    total_rank = rank4 + rank4 + rank1
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    # Part 1: B cols 0-3
    U[:, 0:rank4] = U4
    V[0:16, 0:rank4] = V4
    W[0:16, 0:rank4] = W4
    
    # Part 2: B cols 4-7
    offset1 = rank4
    U[:, offset1:offset1+rank4] = U4
    V[16:32, offset1:offset1+rank4] = V4
    W[16:32, offset1:offset1+rank4] = W4
    
    # Part 3: B col 8
    offset2 = offset1 + rank4
    U[:, offset2:offset2+rank1] = U1
    V[32:36, offset2:offset2+rank1] = V1
    W[32:36, offset2:offset2+rank1] = W1
    
    # Convert to lists of lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
