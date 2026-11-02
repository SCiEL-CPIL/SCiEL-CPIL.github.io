import numpy as np

DIMENSIONS = [2, 4, 7]

def _get_strassen_222():
    """Returns U, V, W for 2x2x2 multiplication using Strassen's algorithm (Rank 7)."""
    n, m, p = 2, 2, 2
    r = 7
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    
    # Indices: U[i*m+j], V[j*p+k], W[k*n+i]
    # Strassen terms M1..M7
    
    # M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1 # C00, C11
    
    # M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1 # C10, C11
    
    # M3 = A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1 # C01, C11
    
    # M4 = A11(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1 # C00, C10
    
    # M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1 # C00, C01
    
    # M6 = (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1 # C11
    
    # M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1 # C00
    
    return U, V, W

def _get_school_221():
    """Returns U, V, W for 2x2x1 multiplication using schoolbook (Rank 4)."""
    n, m, p = 2, 2, 1
    r = 4
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    
    for c in range(r):
        i = c // m
        j = c % m
        k = 0
        U[i * m + j, c] = 1.0
        V[j * p + k, c] = 1.0
        W[k * n + i, c] = 1.0
        
    return U, V, W

def _combine_tensors(tensors, n, m, p):
    """
    Combines a list of (U, V, W) tensors for sub-problems into a single tensor for n x m x p.
    tensors: list of (U, V, W) where U is (n*m_sub, r_sub), etc.
    This function assumes the tensors are for blocks that partition the m and p dimensions.
    However, we will do manual placement for clarity and correctness.
    """
    pass

def construct():
    n, m, p = DIMENSIONS
    
    # We split m=4 into m1=2, m2=2.
    # We split p=7 into p1=4, p2=3 for each m-block.
    # Total rank = 2 * (R(2,2,4) + R(2,2,3))
    # R(2,2,4) = 2 * R(2,2,2) = 14
    # R(2,2,3) = R(2,2,2) + R(2,2,1) = 7 + 4 = 11
    # Total Rank = 2 * (14 + 11) = 50
    
    total_rank = 50
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col_offset = 0
    
    # Block 1: m in [0, 2)
    # Sub-block 1.1: p in [0, 4) -> 2x2x4
    # Split p in [0, 4) into [0, 2) and [2, 4)
    
    # 1.1.1: p in [0, 2) -> 2x2x2 (Strassen)
    U_sub, V_sub, W_sub = _get_strassen_222()
    r_sub = U_sub.shape[1]
    
    # Place U
    # U indices: i*4 + j. j in [0, 2).
    for i in range(2):
        for j in range(2):
            U[i*4 + j, col_offset:col_offset+r_sub] = U_sub[i*2 + j, :]
            
    # Place V
    # V indices: j*7 + k. j in [0, 2), k in [0, 2).
    for j in range(2):
        for k in range(2):
            V[j*7 + k, col_offset:col_offset+r_sub] = V_sub[j*2 + k, :]
            
    # Place W
    # W indices: k*2 + i. k in [0, 2), i in [0, 2).
    for k in range(2):
        for i in range(2):
            W[k*2 + i, col_offset:col_offset+r_sub] = W_sub[k*2 + i, :]
            
    col_offset += r_sub
    
    # 1.1.2: p in [2, 4) -> 2x2x2 (Strassen)
    U_sub, V_sub, W_sub = _get_strassen_222()
    r_sub = U_sub.shape[1]
    
    # Place U (same as before)
    for i in range(2):
        for j in range(2):
            U[i*4 + j, col_offset:col_offset+r_sub] = U_sub[i*2 + j, :]
            
    # Place V
    # k in [2, 4)
    for j in range(2):
        for k in range(2):
            V[j*7 + (k+2), col_offset:col_offset+r_sub] = V_sub[j*2 + k, :]
            
    # Place W
    # k in [2, 4)
    for k in range(2):
        for i in range(2):
            W[(k+2)*2 + i, col_offset:col_offset+r_sub] = W_sub[k*2 + i, :]
            
    col_offset += r_sub
    
    # Sub-block 1.2: p in [4, 7) -> 2x2x3
    # Split p in [4, 7) into [4, 6) and [6, 7)
    
    # 1.2.1: p in [4, 6) -> 2x2x2 (Strassen)
    U_sub, V_sub, W_sub = _get_strassen_222()
    r_sub = U_sub.shape[1]
    
    for i in range(2):
        for j in range(2):
            U[i*4 + j, col_offset:col_offset+r_sub] = U_sub[i*2 + j, :]
            
    for j in range(2):
        for k in range(2):
            V[j*7 + (k+4), col_offset:col_offset+r_sub] = V_sub[j*2 + k, :]
            
    for k in range(2):
        for i in range(2):
            W[(k+4)*2 + i, col_offset:col_offset+r_sub] = W_sub[k*2 + i, :]
            
    col_offset += r_sub
    
    # 1.2.2: p in [6, 7) -> 2x2x1 (Schoolbook)
    U_sub, V_sub, W_sub = _get_school_221()
    r_sub = U_sub.shape[1]
    
    for i in range(2):
        for j in range(2):
            U[i*4 + j, col_offset:col_offset+r_sub] = U_sub[i*2 + j, :]
            
    for j in range(2):
        for k in range(1):
            V[j*7 + (k+6), col_offset:col_offset+r_sub] = V_sub[j*1 + k, :]
            
    for k in range(1):
        for i in range(2):
            W[(k+6)*2 + i, col_offset:col_offset+r_sub] = W_sub[k*1 + i, :]
            
    col_offset += r_sub
    
    # Block 2: m in [2, 4)
    # Repeat the same structure for the second half of A and B
    
    # 2.1.1: p in [0, 2)
    U_sub, V_sub, W_sub = _get_strassen_222()
    r_sub = U_sub.shape[1]
    
    for i in range(2):
        for j in range(2):
            U[i*4 + (j+2), col_offset:col_offset+r_sub] = U_sub[i*2 + j, :]
            
    for j in range(2):
        for k in range(2):
            V[(j+2)*7 + k, col_offset:col_offset+r_sub] = V_sub[j*2 + k, :]
            
    for k in range(2):
        for i in range(2):
            W[k*2 + i, col_offset:col_offset+r_sub] = W_sub[k*2 + i, :]
            
    col_offset += r_sub
    
    # 2.1.2: p in [2, 4)
    U_sub, V_sub, W_sub = _get_strassen_222()
    r_sub = U_sub.shape[1]
    
    for i in range(2):
        for j in range(2):
            U[i*4 + (j+2), col_offset:col_offset+r_sub] = U_sub[i*2 + j, :]
            
    for j in range(2):
        for k in range(2):
            V[(j+2)*7 + (k+2), col_offset:col_offset+r_sub] = V_sub[j*2 + k, :]
            
    for k in range(2):
        for i in range(2):
            W[(k+2)*2 + i, col_offset:col_offset+r_sub] = W_sub[k*2 + i, :]
            
    col_offset += r_sub
    
    # 2.2.1: p in [4, 6)
    U_sub, V_sub, W_sub = _get_strassen_222()
    r_sub = U_sub.shape[1]
    
    for i in range(2):
        for j in range(2):
            U[i*4 + (j+2), col_offset:col_offset+r_sub] = U_sub[i*2 + j, :]
            
    for j in range(2):
        for k in range(2):
            V[(j+2)*7 + (k+4), col_offset:col_offset+r_sub] = V_sub[j*2 + k, :]
            
    for k in range(2):
        for i in range(2):
            W[(k+4)*2 + i, col_offset:col_offset+r_sub] = W_sub[k*2 + i, :]
            
    col_offset += r_sub
    
    # 2.2.2: p in [6, 7)
    U_sub, V_sub, W_sub = _get_school_221()
    r_sub = U_sub.shape[1]
    
    for i in range(2):
        for j in range(2):
            U[i*4 + (j+2), col_offset:col_offset+r_sub] = U_sub[i*2 + j, :]
            
    for j in range(2):
        for k in range(1):
            V[(j+2)*7 + (k+6), col_offset:col_offset+r_sub] = V_sub[j*1 + k, :]
            
    for k in range(1):
        for i in range(2):
            W[(k+6)*2 + i, col_offset:col_offset+r_sub] = W_sub[k*1 + i, :]
            
    col_offset += r_sub
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
