import numpy as np

DIMENSIONS = [4, 4, 6]

def base_strassen_2x2x2():
    """
    Returns U, V, W for 2x2x2 matrix multiplication using Strassen's algorithm.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indices:
    A: 00, 01, 10, 11 -> 0, 1, 2, 3
    B: 00, 01, 10, 11 -> 0, 1, 2, 3
    M: 00, 01, 10, 11 -> 0, 1, 2, 3
    """
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    
    # P1 = (A11 - A00)(B00 + B10)
    U[3, 0] = 1; U[0, 0] = -1
    V[0, 0] = 1; V[2, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # P2 = (A00 + A11) B11
    U[0, 1] = 1; U[3, 1] = 1
    V[3, 1] = 1
    W[1, 1] = 1; W[2, 1] = 1
    
    # P3 = A00 (B01 - B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = -1
    
    # P4 = A11 (B00 + B10)
    U[3, 3] = 1
    V[0, 3] = 1; V[2, 3] = 1
    W[0, 3] = 1; W[3, 3] = 1
    
    # P5 = (A00 + A01) B10
    U[0, 4] = 1; U[1, 4] = 1
    V[2, 4] = 1
    W[1, 4] = 1; W[3, 4] = 1
    
    # P6 = (A10 - A00) (B01 + B11)
    U[2, 5] = 1; U[0, 5] = -1
    V[1, 5] = 1; V[3, 5] = 1
    W[2, 5] = 1
    
    # P7 = (A01 - A11) (B00 + B10)
    U[1, 6] = 1; U[3, 6] = -1
    V[0, 6] = 1; V[2, 6] = 1
    W[0, 6] = 1; W[3, 6] = 1
    
    return U, V, W

def strassen_square(n):
    """
    Returns U, V, W for nxnxn matrix multiplication using Strassen's algorithm.
    Assumes n is a power of 2.
    """
    if n == 2:
        return base_strassen_2x2x2()
    
    half = n // 2
    # Get decomposition for half x half x half
    Uh, Vh, Wh = strassen_square(half)
    
    # We need 7 copies of this decomposition
    # Total rank = 7 * rank(half)
    rank_h = Uh.shape[1]
    total_rank = 7 * rank_h
    
    U = np.zeros((n*n, total_rank), dtype=int)
    V = np.zeros((n*n, total_rank), dtype=int)
    W = np.zeros((n*n, total_rank), dtype=int)
    
    # Helper to place a block of U/V/W
    # Block indices for A, B, M are (r, c) where r,c in {0,1}
    # Global index for A[i,j] is i*n + j
    # In the half-size decomposition, the index is local_i*half + local_j
    
    def place_block(matrix, block_r, block_c, sub_matrix, col_offset):
        # block_r, block_c are 0 or 1
        # sub_matrix is (half*half, rank_h)
        for i in range(half):
            for j in range(half):
                global_i = block_r * half + i
                global_j = block_c * half + j
                global_idx = global_i * n + global_j
                local_idx = i * half + j
                matrix[global_idx, col_offset:col_offset+rank_h] += sub_matrix[local_idx, :]

    # Define the 7 Strassen products
    # P1 = (A11 - A00)(B00 + B10)
    # A part: A11 - A00. 
    # We can't just sum U blocks directly because the sub-decomposition expects a single matrix A.
    # We need to construct the "effective" A and B for each product.
    # Since the decomposition is linear, we can add the contributions.
    
    # For P1:
    # A_eff = A11 - A00
    # B_eff = B00 + B10
    # M_eff = M00 + M11 (with signs)
    
    # Actually, it's easier to just sum the U, V, W matrices with appropriate signs and permutations.
    
    # Let's define the linear combinations for A, B, M for each P_k.
    # A_combos[k] is a list of (block_r, block_c, sign)
    # B_combos[k] is a list of (block_r, block_c, sign)
    # M_combos[k] is a list of (block_r, block_c, sign)
    
    A_combos = [
        [(1,1,1), (0,0,-1)], # P1
        [(0,0,1), (1,1,1)],  # P2
        [(0,0,1)],           # P3
        [(1,1,1)],           # P4
        [(0,0,1), (0,1,1)],  # P5
        [(1,0,1), (0,0,-1)], # P6
        [(0,1,1), (1,1,-1)]  # P7
    ]
    
    B_combos = [
        [(0,0,1), (1,0,1)],  # P1
        [(1,1,1)],           # P2
        [(0,1,1), (1,1,-1)], # P3
        [(0,0,1), (1,0,1)],  # P4
        [(1,0,1)],           # P5
        [(0,1,1), (1,1,1)],  # P6
        [(0,0,1), (1,0,1)]   # P7
    ]
    
    M_combos = [
        [(0,0,1), (1,1,1)],  # P1
        [(0,1,1), (1,0,1)],  # P2
        [(0,1,1), (1,1,-1)], # P3
        [(0,0,1), (1,1,1)],  # P4
        [(0,1,1), (1,1,1)],  # P5
        [(1,0,1)],           # P6
        [(0,0,1), (1,1,1)]   # P7
    ]
    
    for k in range(7):
        col_offset = k * rank_h
        
        # Build U
        for br, bc, sign in A_combos[k]:
            place_block(U, br, bc, Uh, col_offset)
            # Multiply by sign
            U[:, col_offset:col_offset+rank_h] *= sign
            
        # Build V
        for br, bc, sign in B_combos[k]:
            place_block(V, br, bc, Vh, col_offset)
            V[:, col_offset:col_offset+rank_h] *= sign
            
        # Build W
        for br, bc, sign in M_combos[k]:
            place_block(W, br, bc, Wh, col_offset)
            W[:, col_offset:col_offset+rank_h] *= sign
            
    return U, V, W

def split_decomp(m, n, p):
    """
    Recursive decomposition by splitting dimensions.
    """
    if m == 2 and n == 2 and p == 2:
        return base_strassen_2x2x2()
    
    # Choose dimension to split
    # Prefer splitting the largest even dimension
    splits = []
    if m % 2 == 0:
        splits.append(('m', m//2))
    if n % 2 == 0:
        splits.append(('n', n//2))
    if p % 2 == 0:
        splits.append(('p', p//2))
        
    if not splits:
        # Fallback to schoolbook if no even dimension (should not happen for powers of 2)
        rank = m*n*p
        U = np.eye(m*n, rank)
        V = np.eye(n*p, rank)
        W = np.eye(p*m, rank)
        # Need to permute W for schoolbook
        # This is complex, so we assume inputs are powers of 2.
        pass

    # Pick the first available split
    dim, half = splits[0]
    
    if dim == 'm':
        m1, m2 = half, m - half
        U1, V1, W1 = split_decomp(m1, n, p)
        U2, V2, W2 = split_decomp(m2, n, p)
        
        # Combine
        # U: (m*n, rank). Split into top m1*n and bottom m2*n.
        # V: (n*p, rank). Same for both.
        # W: (p*m, rank). Split into top p*m1 and bottom p*m2.
        
        rank1 = U1.shape[1]
        rank2 = U2.shape[1]
        total_rank = rank1 + rank2
        
        U = np.zeros((m*n, total_rank), dtype=int)
        V = np.zeros((n*p, total_rank), dtype=int)
        W = np.zeros((p*m, total_rank), dtype=int)
        
        U[:m1*n, :rank1] = U1
        U[m1*n:, rank1:] = U2
        
        V[:, :rank1] = V1
        V[:, rank1:] = V2
        
        W[:p*m1, :rank1] = W1
        W[p*m1:, rank1:] = W2
        
        return U, V, W
        
    elif dim == 'n':
        n1, n2 = half, n - half
        U1, V1, W1 = split_decomp(m, n1, p)
        U2, V2, W2 = split_decomp(m, n2, p)
        
        rank1 = U1.shape[1]
        rank2 = U2.shape[1]
        total_rank = rank1 + rank2
        
        U = np.zeros((m*n, total_rank), dtype=int)
        V = np.zeros((n*p, total_rank), dtype=int)
        W = np.zeros((p*m, total_rank), dtype=int)
        
        # U: (m*n, rank). Split columns? No, U is indexed by i*m+j.
        # If we split n (j), then for a fixed i, j is split.
        # U1 corresponds to j in [0, n1).
        # U2 corresponds to j in [n1, n).
        # So for each i, U[i, :n1] comes from U1[i, :], U[i, n1:] comes from U2[i, :].
        
        for i in range(m):
            U[i*n : i*n+n1, :rank1] = U1[i*n1 : i*n1+n1, :rank1]
            U[i*n+n1 : (i+1)*n, rank1:] = U2[i*n2 : i*n2+n2, :rank2]
            
        # V: (n*p, rank). Split rows.
        # V1 corresponds to j in [0, n1).
        # V2 corresponds to j in [n1, n).
        V[:n1*p, :rank1] = V1
        V[n1*p:, rank1:] = V2
        
        # W: (p*m, rank). No change in structure, just sum.
        W[:, :rank1] = W1
        W[:, rank1:] = W2
        
        return U, V, W
        
    elif dim == 'p':
        p1, p2 = half, p - half
        U1, V1, W1 = split_decomp(m, n, p1)
        U2, V2, W2 = split_decomp(m, n, p2)
        
        rank1 = U1.shape[1]
        rank2 = U2.shape[1]
        total_rank = rank1 + rank2
        
        U = np.zeros((m*n, total_rank), dtype=int)
        V = np.zeros((n*p, total_rank), dtype=int)
        W = np.zeros((p*m, total_rank), dtype=int)
        
        # U: No change.
        U[:, :rank1] = U1
        U[:, rank1:] = U2
        
        # V: (n*p, rank). Split columns? No, V is indexed by j*p+k.
        # If we split p (k), then for a fixed j, k is split.
        # V1 corresponds to k in [0, p1).
        # V2 corresponds to k in [p1, p).
        for j in range(n):
            V[j*p : j*p+p1, :rank1] = V1[j*p1 : j*p1+p1, :rank1]
            V[j*p+p1 : (j+1)*p, rank1:] = V2[j*p2 : j*p2+p2, :rank2]
            
        # W: (p*m, rank). Split rows.
        # W1 corresponds to k in [0, p1).
        # W2 corresponds to k in [p1, p).
        W[:p1*m, :rank1] = W1
        W[p1*m:, rank1:] = W2
        
        return U, V, W

def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=6
    
    # Split p=6 into 4 and 2
    p1, p2 = 4, 2
    
    # Part 1: 4x4x4
    U1, V1, W1 = strassen_square(4)
    
    # Part 2: 4x4x2
    U2, V2, W2 = split_decomp(4, 4, 2)
    
    # Combine
    rank1 = U1.shape[1]
    rank2 = U2.shape[1]
    total_rank = rank1 + rank2
    
    U = np.zeros((n*m, total_rank), dtype=int)
    V = np.zeros((m*p, total_rank), dtype=int)
    W = np.zeros((p*n, total_rank), dtype=int)
    
    # U is same for both (A is 4x4)
    U[:, :rank1] = U1
    U[:, rank1:] = U2
    
    # V: (m*p, rank) = (4*6, rank)
    # V1 corresponds to k in [0, 4). Shape (4*4, rank1) = (16, rank1)
    # V2 corresponds to k in [4, 6). Shape (4*2, rank2) = (8, rank2)
    # Global V index: j*p + k.
    # For V1, j in [0,3], k in [0,3].
    # For V2, j in [0,3], k in [4,5].
    
    for j in range(m):
        # V1 part
        for k in range(p1):
            V[j*p + k, :rank1] = V1[j*p1 + k, :]
        # V2 part
        for k in range(p2):
            V[j*p + p1 + k, rank1:] = V2[j*p2 + k, :]
            
    # W: (p*n, rank) = (6*4, rank)
    # W1 corresponds to k in [0, 4). Shape (4*4, rank1) = (16, rank1)
    # W2 corresponds to k in [4, 6). Shape (2*4, rank2) = (8, rank2)
    # Global W index: k*n + i.
    # For W1, k in [0,3], i in [0,3].
    # For W2, k in [4,5], i in [0,3].
    
    for k in range(p1):
        for i in range(n):
            W[k*n + i, :rank1] = W1[k*p1 + i, :]
            
    for k in range(p2):
        for i in range(n):
            W[(p1+k)*n + i, rank1:] = W2[k*p2 + i, :]
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-START
# The construct function is defined above.
# EVOLVE-BLOCK-END
