"""Matrix multiplication 4x4x9 rank decomposition using Strassen + schoolbook."""
import time
import numpy as np

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def _build_strassen_2x2():
    """
    Build the rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    Returns U, V, W matrices of shape (4, 7) each.
    
    For C = A * B where A, B, C are 2x2:
    C[0,0] = M5 + M4 - M2 + M6
    C[0,1] = M1 + M4
    C[1,0] = M3 + M5
    C[1,1] = M5 + M4 - M1 + M7
    
    Where:
    M1 = (A00 + A11)(B00 + B11)
    M2 = (A10 + A11) * B00
    M3 = A00 * (B01 - B11)
    M4 = A11 * (B10 - B00)
    M5 = (A00 + A01) * B11
    M6 = (A10 - A00) * (B00 + B01)
    M7 = (A01 - A11) * (B10 + B11)
    """
    n = 2
    rank = 7
    U = np.zeros((n*n, rank))
    V = np.zeros((n*n, rank))
    W = np.zeros((n*n, rank))
    
    # Index convention: U[i*n+j, r], V[j*n+k, r], W[k*n+i, r]
    # For 2x2: U index = i*2+j, V index = j*2+k, W index = k*2+i
    
    # M1 = (A00 + A11)(B00 + B11)
    # U: A00 and A11 contribute
    U[0, 0] = 1  # A[0,0]
    U[3, 0] = 1  # A[1,1]
    # V: B00 and B11 contribute
    V[0, 0] = 1  # B[0,0]
    V[3, 0] = 1  # B[1,1]
    # W: contributes to C[0,0] and C[1,1]
    W[0, 0] = 1  # C[0,0] -> k=0,i=0 -> index 0*2+0=0
    W[3, 0] = 1  # C[1,1] -> k=1,i=1 -> index 1*2+1=3
    
    # M2 = (A10 + A11) * B00
    U[2, 1] = 1  # A[1,0]
    U[3, 1] = 1  # A[1,1]
    V[0, 1] = 1  # B[0,0]
    W[2, 1] = -1  # C[1,0] -> k=0,i=1 -> index 0*2+1=1... wait
    
    # Let me re-examine the W index convention: W[k*n+i, r]
    # C[i,k] where i is row, k is column
    # For C[0,0]: i=0, k=0, W index = 0*2+0 = 0
    # For C[0,1]: i=0, k=1, W index = 1*2+0 = 2
    # For C[1,0]: i=1, k=0, W index = 0*2+1 = 1
    # For C[1,1]: i=1, k=1, W index = 1*2+1 = 3
    
    # M1 = (A00 + A11)(B00 + B11)
    # C[0,0] += M1, C[1,1] += M1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A10 + A11) * B00
    # C[1,0] -= M2
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = -1  # C[1,0]
    
    # M3 = A00 * (B01 - B11)
    # C[1,0] += M3
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1  # B[0,1] - B[1,1]
    W[1, 2] = 1  # C[1,0]
    
    # M4 = A11 * (B10 - B00)
    # C[0,0] += M4, C[0,1] += M4, C[1,1] += M4
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1  # B[1,0] - B[0,0]
    W[0, 3] = 1; W[2, 3] = 1; W[3, 3] = 1  # C[0,0], C[0,1], C[1,1]
    
    # M5 = (A00 + A01) * B11
    # C[0,0] += M5, C[1,0] += M5, C[1,1] -= M5
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = 1; W[1, 4] = 1; W[3, 4] = -1  # C[0,0], C[1,0], C[1,1]
    
    # M6 = (A10 - A00) * (B00 + B01)
    # C[0,0] += M6
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[0, 5] = 1  # C[0,0]
    
    # M7 = (A01 - A11) * (B10 + B11)
    # C[0,1] -= M7
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[2, 6] = -1  # C[0,1]
    
    return U, V, W


def _build_4x4x4_strassen():
    """
    Build a rank-49 decomposition for 4x4x4 matrix multiplication using
    recursive Strassen. 4x4x4 can be decomposed into 7 blocks of 2x2x2.
    Each 2x2x2 has rank 7, so total rank = 7*7 = 49.
    
    Returns U, V, W of shape (16, 49), (16, 49), (16, 49).
    """
    n = 4
    rank = 49
    U = np.zeros((n*n, rank))
    V = np.zeros((n*n, rank))
    W = np.zeros((n*n, rank))
    
    # Get the 2x2 Strassen decomposition
    U2, V2, W2 = _build_strassen_2x2()
    
    # 4x4 matrix can be viewed as 4 blocks of 2x2:
    # A = [[A00, A01], [A10, A11]] where each Ai is 2x2
    # Similarly for B and C
    # 
    # Strassen for 4x4:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    #
    # Each Mi is a 2x2 matrix product, so each can be computed with rank 7
    # Total rank = 7 * 7 = 49
    
    # Block indices for 2x2 blocks in 4x4:
    # Block (bi, bj) covers rows 2*bi to 2*bi+1, cols 2*bj to 2*bj+1
    # U index for A[i,j]: i*4 + j
    # V index for B[j,k]: j*4 + k  
    # W index for C[k,i]: k*4 + i
    
    # For each of the 7 Strassen multipliers, we need to:
    # 1. Define how A blocks combine (using U2 pattern)
    # 2. Define how B blocks combine (using V2 pattern)  
    # 3. Define how the result contributes to C blocks (using W2 pattern)
    
    # The 7 Strassen combinations for 4x4 (block level):
    # Each combination specifies which 2x2 blocks of A and B to use
    # and with what signs
    
    strassen_blocks = [
        # (A_blocks, B_blocks, C_blocks)
        # Each block is (block_row, block_col)
        # A_blocks: list of (sign, (br, bc)) for A
        # B_blocks: list of (sign, (br, bc)) for B
        # C_blocks: list of (sign, (br, bc)) for C
        [
            # M1 = (A00 + A11)(B00 + B11)
            ([((1, (0,0)), (1, (1,1)))],
             [((1, (0,0)), (1, (1,1)))],
             [((1, (0,0)), (1, (1,1)))]),
            # M2 = (A10 + A11) * B00
            ([((1, (1,0)), (1, (1,1)))],
             [((1, (0,0)))],
             [((1, (1,0)))]),
            # M3 = A00 * (B01 - B11)
            ([((1, (0,0)))],
             [((1, (0,1)), (-1, (1,1)))],
             [((1, (1,0)))]),
            # M4 = A11 * (B10 - B00)
            ([((1, (1,1)))],
             [((1, (1,0)), (-1, (0,0)))],
             [((1, (0,0)), (1, (0,1)), (1, (1,1)))]),
            # M5 = (A00 + A01) * B11
            ([((1, (0,0)), (1, (0,1)))],
             [((1, (1,1)))],
             [((1, (0,0)), (1, (1,0)), (-1, (1,1)))]),
            # M6 = (A10 - A00) * (B00 + B01)
            ([((1, (1,0)), (-1, (0,0)))],
             [((1, (0,0)), (1, (0,1)))],
             [((1, (0,0)))]),
            # M7 = (A01 - A11) * (B10 + B11)
            ([((1, (0,1)), (-1, (1,1)))],
             [((1, (1,0)), (1, (1,1)))],
             [((-1, (0,1)))]),
        ]
    ]
    
    col = 0
    for s_idx, (a_blocks, b_blocks, c_blocks) in enumerate(strassen_blocks):
        for r2 in range(7):  # 7 ranks for each 2x2x2
            for (sa, (abr, abc)) in a_blocks:
                # A block (abr, abc) has 4 elements
                for di in range(2):
                    for dj in range(2):
                        i = 2*abr + di
                        j = 2*abc + dj
                        U[i*4 + j, col] += sa * U2[i*2 + j, r2]
            
            for (sb, (bbr, bbc)) in b_blocks:
                for dj in range(2):
                    for dk in range(2):
                        j = 2*bbr + dj
                        k = 2*bbc + dk
                        V[j*4 + k, col] += sb * V2[j*2 + k, r2]
            
            for (sc, (cbr, cbc)) in c_blocks:
                for dk in range(2):
                    for di in range(2):
                        k = 2*cbr + dk
                        i = 2*cbc + di
                        W[k*4 + i, col] += sc * W2[k*2 + i, r2]
            
            col += 1
    
    return U, V, W


def _build_4x4x5_schoolbook():
    """
    Build schoolbook decomposition for 4x4 * 4x5 = 4x5.
    Rank = 4*4*5 = 80.
    Returns U, V, W of shape (16, 80), (20, 80), (20, 80).
    """
    n = 4
    m = 4
    p = 5
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i*m + j
                v = j*p + k
                w = k*n + i
                U[u, col] = 1
                V[v, col] = 1
                W[w, col] = 1
                col += 1
    
    return U, V, W


def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that sum_r U[u,r]*V[v,r]*W[w,r] equals the matmul tensor."""
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i*m + j
                v = j*p + k
                w = k*n + i
                T[u, v, w] = 1
    
    # Compute sum_r U[u,r] * V[v,r] * W[w,r]
    # This is a tensor contraction
    # Result[u,v,w] = sum_r U[u,r] * V[v,r] * W[w,r]
    # We can compute this as: for each r, outer product of U[:,r], V[:,r], W[:,r]
    
    # Efficient computation using einsum
    # result = np.einsum('ur,vr,wr->uvw', U, V, W)
    # But this might be slow for large ranks. Let's do it in chunks.
    
    rank = U.shape[1]
    result = np.zeros((n*m, m*p, p*n))
    
    for r in range(rank):
        result += np.einsum('u,v,w->uvw', U[:, r], V[:, r], W[:, r])
    
    max_diff = np.max(np.abs(result - T))
    return max_diff < tol, max_diff


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    # Strategy: Split B (4x9) into B1 (4x4) and B2 (4x5)
    # Use Strassen 4x4x4 for A*B1 (rank 49)
    # Use schoolbook 4x4x5 for A*B2 (rank 80)
    # Total rank: 49 + 80 = 129
    
    # But we need to handle the index mapping carefully.
    # The full problem: A (4x4) * B (4x9) = C (4x9)
    # U shape: (n*m, rank) = (16, rank)
    # V shape: (m*p, rank) = (36, rank)  
    # W shape: (p*n, rank) = (36, rank)
    
    # For the Strassen part (first 4 columns of B):
    # A (4x4) * B1 (4x4) = C1 (4x4)
    # U1: (16, 49), V1: (16, 49), W1: (16, 49)
    # But in the full problem, V1 needs to map to the first 4 columns of B
    # V index in full problem: j*p + k where p=9
    # For B1, k ranges 0-3, so V1 indices are j*9 + k for k=0..3
    
    # For the schoolbook part (last 5 columns of B):
    # A (4x4) * B2 (4x5) = C2 (4x5)
    # U2: (16, 80), V2: (20, 80), W2: (20, 80)
    # But in the full problem, V2 needs to map to columns 4-8 of B
    # V2 index in full problem: j*9 + (4+k) for k=0..4
    # W2 index in full problem: (4+k)*4 + i for k=0..4, i=0..3
    
    # Build Strassen 4x4x4
    U_str, V_str, W_str = _build_4x4x4_strassen()
    
    # Build schoolbook 4x4x5
    U_sch, V_sch, W_sch = _build_4x4x5_schoolbook()
    
    rank_str = U_str.shape[1]  # 49
    rank_sch = U_sch.shape[1]  # 80
    total_rank = rank_str + rank_sch  # 129
    
    # Assemble full U, V, W
    U_full = np.zeros((n*m, total_rank))
    V_full = np.zeros((m*p, total_rank))
    W_full = np.zeros((p*n, total_rank))
    
    # Strassen part (columns 0-3 of B)
    U_full[:, :rank_str] = U_str
    
    # Map V_str indices: V_str[j*4+k, r] for j=0..3, k=0..3
    # In full problem: V[j*9+k, r] for j=0..3, k=0..3
    for j in range(m):
        for k in range(4):
            V_full[j*9 + k, :rank_str] = V_str[j*4 + k, :]
    
    # Map W_str indices: W_str[k*4+i, r] for k=0..3, i=0..3
    # In full problem: W[k*4+i, r] for k=0..3, i=0..3
    for k in range(4):
        for i in range(n):
            W_full[k*4 + i, :rank_str] = W_str[k*4 + i, :]
    
    # Schoolbook part (columns 4-8 of B)
    U_full[:, rank_str:] = U_sch
    
    # Map V_sch indices: V_sch[j*5+k, r] for j=0..3, k=0..4
    # In full problem: V[j*9 + (4+k), r] for j=0..3, k=0..4
    for j in range(m):
        for k in range(5):
            V_full[j*9 + (4 + k), rank_str:] = V_sch[j*5 + k, :]
    
    # Map W_sch indices: W_sch[k*4+i, r] for k=0..4, i=0..3
    # In full problem: W[(4+k)*4+i, r] for k=0..4, i=0..3
    for k in range(5):
        for i in range(n):
            W_full[(4 + k)*4 + i, rank_str:] = W_sch[k*4 + i, :]
    
    # Verify
    valid, max_diff = _verify_decomposition(U_full, V_full, W_full, n, m, p)
    
    if not valid:
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i*m + j
                    v = j*p + k
                    w = k*n + i
                    U[u, col] = 1
                    V[v, col] = 1
                    W[w, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
# EVOLVE-BLOCK-END
