import numpy as np

def get_trivial_decomp(n, m, p):
    """
    Returns U, V, W for trivial decomposition of n x m by m x p.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    Rank = n*m*p
    """
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    i_idx, j_idx, k_idx = np.indices((n, m, p))
    i_flat = i_idx.flatten()
    j_flat = j_idx.flatten()
    k_flat = k_idx.flatten()
    
    u_rows = i_flat * m + j_flat
    v_rows = j_flat * p + k_flat
    w_rows = k_flat * n + i_flat
    
    U[u_rows, np.arange(rank)] = 1
    V[v_rows, np.arange(rank)] = 1
    W[w_rows, np.arange(rank)] = 1
    
    return U, V, W

def construct():
    n, m, p = 3, 4, 7
    
    # Sub-problem 1: 3x3x3 (A[0:3, 0:3], B[0:3, 0:3], C[0:3, 0:3])
    U1, V1, W1 = get_trivial_decomp(3, 3, 3)
    
    # Sub-problem 2: 3x3x4 (A[0:3, 0:3], B[0:3, 3:7], C[0:3, 3:7])
    U2, V2, W2 = get_trivial_decomp(3, 3, 4)
    
    # Sub-problem 3: 3x1x7 (A[0:3, 3:4], B[3:4, 0:7], C[0:3, 0:7])
    # Rank 1 decomposition
    U3 = np.zeros((3 * 1, 1))
    V3 = np.zeros((1 * 7, 1))
    W3 = np.zeros((7 * 3, 1))
    
    # U3: n=3, m=1. Row i*1 + j. j is always 0.
    U3[0, 0] = 1
    U3[1, 0] = 1
    U3[2, 0] = 1
    
    # V3: m=1, p=7. Row j*7 + k. j is always 0.
    V3[0, 0] = 1 # j=0, k=0? No, V3 needs to be 1 for all k?
    # Wait, V is (m*p, rank).
    # V[j*p + k, r]
    # For rank 1, we need V[j*p+k, 0] such that sum V * U * W = delta.
    # U[i,0]=1, W[k*n+i,0]=1.
    # We need V[j*p+k,0] to be 1 if j=0, else 0?
    # Let's check the tensor: T[i,j,k] = delta[j,0].
    # Sum_r U[i,r] V[j*p+k,r] W[k*n+i,r]
    # If U[i,0]=1, W[k*n+i,0]=1, then we need V[j*p+k,0] = 1 if j=0, else 0.
    # V has shape (1*7, 1) = (7,1).
    # Index is j*7+k. j is 0. So indices 0..6.
    # V[0..6, 0] should be 1.
    V3[:] = 1
    
    # W3: p=7, n=3. Row k*n + i.
    # W[k*3+i, 0] should be 1 for all i,k.
    W3[:] = 1
    
    # Now combine
    # U: (12, Rank)
    # V: (28, Rank)
    # W: (21, Rank)
    
    total_rank = U1.shape[1] + U2.shape[1] + U3.shape[1]
    
    U = np.zeros((12, total_rank))
    V = np.zeros((28, total_rank))
    W = np.zeros((21, total_rank))
    
    # Offset 0
    # U1: (9, 27). Maps to A[0:3, 0:3] -> rows 0..8.
    U[0:9, 0:27] = U1
    # V1: (21, 27). Maps to B[0:3, 0:3] -> rows 0..10? 
    # B is 4x7. Rows are j*7+k.
    # B[0:3, 0:3] corresponds to j=0,1,2 and k=0,1,2.
    # Indices:
    # j=0: k=0,1,2 -> 0,1,2
    # j=1: k=0,1,2 -> 7,8,9
    # j=2: k=0,1,2 -> 14,15,16
    # V1 rows are local j*3+k.
    # We need to scatter V1 into V.
    
    # Helper to scatter
    def scatter_U(local_U, local_n, local_m, global_n, global_m, row_offset, col_offset):
        # local_U is (local_n*local_m, rank)
        # Global U is (global_n*global_m, total_rank)
        # Map local row i*local_m + j to global row (row_offset + i)*global_m + (col_offset + j)
        # Here row_offset is 0, col_offset is 0 for U1.
        pass

    # It's easier to just loop and assign.
    
    # Sub 1: A[0:3, 0:3]
    for i in range(3):
        for j in range(3):
            local_row = i * 3 + j
            global_row = i * 4 + j
            U[global_row, 0:27] = U1[local_row, :]
            
    for j in range(3):
        for k in range(3):
            local_row = j * 3 + k
            global_row = j * 7 + k
            V[global_row, 0:27] = V1[local_row, :]
            
    for k in range(3):
        for i in range(3):
            local_row = k * 3 + i
            global_row = k * 3 + i
            W[global_row, 0:27] = W1[local_row, :]

    # Sub 2: A[0:3, 0:3] (same A part)
    # B[0:3, 3:7]
    # C[0:3, 3:7]
    
    rank_offset = 27
    
    for i in range(3):
        for j in range(3):
            local_row = i * 3 + j
            global_row = i * 4 + j
            U[global_row, rank_offset:rank_offset+36] = U2[local_row, :]
            
    for j in range(3):
        for k in range(4):
            local_row = j * 4 + k
            global_k = 3 + k # k offset in B
            global_row = j * 7 + global_k
            V[global_row, rank_offset:rank_offset+36] = V2[local_row, :]
            
    for k in range(4):
        for i in range(3):
            local_row = k * 3 + i
            global_k = 3 + k # k offset in C
            global_row = global_k * 3 + i
            W[global_row, rank_offset:rank_offset+36] = W2[local_row, :]

    # Sub 3: A[0:3, 3:4]
    # B[3:4, 0:7]
    # C[0:3, 0:7]
    
    rank_offset = 27 + 36
    
    for i in range(3):
        # j is 3
        local_row = i * 1 + 0
        global_row = i * 4 + 3
        U[global_row, rank_offset:rank_offset+1] = U3[local_row, :]
        
    for k in range(7):
        # j is 3
        local_row = 0 * 7 + k
        global_row = 3 * 7 + k
        V[global_row, rank_offset:rank_offset+1] = V3[local_row, :]
        
    for k in range(7):
        for i in range(3):
            local_row = k * 3 + i
            global_row = k * 3 + i
            W[global_row, rank_offset:rank_offset+1] = W3[local_row, :]

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
