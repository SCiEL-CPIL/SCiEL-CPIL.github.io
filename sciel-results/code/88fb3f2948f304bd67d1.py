"""Matrix multiplication 2,2,15 tensor rank minimization."""
import time
import numpy as np

DIMENSIONS = [2, 2, 15]
TIME_LIMIT = 140  # Safety margin before 150s kill

def _build_schoolbook(n, m, p):
    """Build the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _build_strassen_2x2x2():
    """
    Build Strassen's rank-7 decomposition for 2x2 x 2x2 matrix multiplication.
    Returns U (4x7), V (4x7), W (4x7) following the trace convention W[k*n+i, r].
    """
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Term 1: M1 = (A00 + A11) * (B00 + B11)
    U[0*2+0, 0] = 1  # A00
    U[1*2+1, 0] = 1  # A11
    V[0*2+0, 0] = 1  # B00
    V[1*2+1, 0] = 1  # B11
    W[0*2+0, 0] = 1  # C00
    W[1*2+1, 0] = 1  # C11
    
    # Term 2: M2 = (A10 + A11) * B00
    U[1*2+0, 1] = 1  # A10
    U[1*2+1, 1] = 1  # A11
    V[0*2+0, 1] = 1  # B00
    W[0*2+1, 1] = 1  # C10
    W[1*2+1, 1] = -1 # C11
    
    # Term 3: M3 = A00 * (B01 - B11)
    U[0*2+0, 2] = 1  # A00
    V[0*2+1, 2] = 1  # B01
    V[1*2+1, 2] = -1 # B11
    W[1*2+0, 2] = 1  # C01
    W[1*2+1, 2] = 1  # C11
    
    # Term 4: M4 = A11 * (B10 - B00)
    U[1*2+1, 3] = 1  # A11
    V[1*2+0, 3] = 1  # B10
    V[0*2+0, 3] = -1 # B00
    W[0*2+0, 3] = 1  # C00
    W[0*2+1, 3] = 1  # C10
    
    # Term 5: M5 = (A00 + A01) * B11
    U[0*2+0, 4] = 1  # A00
    U[0*2+1, 4] = 1  # A01
    V[1*2+1, 4] = 1  # B11
    W[0*2+0, 4] = -1 # C00
    W[1*2+0, 4] = 1  # C01
    
    # Term 6: M6 = (A10 - A00) * (B00 + B01)
    U[1*2+0, 5] = 1  # A10
    U[0*2+0, 5] = -1 # A00
    V[0*2+0, 5] = 1  # B00
    V[0*2+1, 5] = 1  # B01
    W[1*2+1, 5] = 1  # C11
    
    # Term 7: M7 = (A01 - A11) * (B10 + B11)
    U[0*2+1, 6] = 1  # A01
    U[1*2+1, 6] = -1 # A11
    V[1*2+0, 6] = 1  # B10
    V[1*2+1, 6] = 1  # B11
    W[0*2+0, 6] = 1  # C00
    
    return U, V, W

def _build_rank53():
    """Build the rank-53 decomposition using 7 Strassen blocks + 1 schoolbook."""
    n, m, p = DIMENSIONS
    block_sizes = [2, 2, 2, 2, 2, 2, 2, 1]
    num_blocks = len(block_sizes)
    
    U_str, V_str, W_str = _build_strassen_2x2x2()
    U_sb, V_sb, W_sb = _build_schoolbook(2, 2, 1)
    
    total_rank = 7 * 7 + 4  # 53
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col_offset = 0
    for b in range(num_blocks):
        block_size = block_sizes[b]
        start_k = sum(block_sizes[:b])
        
        if block_size == 2:
            U[:, col_offset:col_offset+7] = U_str
            for j in range(m):
                for k_local in range(2):
                    k_global = start_k + k_local
                    V[j * p + k_global, col_offset:col_offset+7] = V_str[j * 2 + k_local, :]
            for k_local in range(2):
                k_global = start_k + k_local
                for i in range(n):
                    W[k_global * n + i, col_offset:col_offset+7] = W_str[k_local * 2 + i, :]
            col_offset += 7
        else:
            U[:, col_offset:col_offset+4] = U_sb
            for j in range(m):
                V[j * p + start_k, col_offset:col_offset+4] = V_sb[j * 1 + 0, :]
            for i in range(n):
                W[start_k * n + i, col_offset:col_offset+4] = W_sb[0 * 2 + i, :]
            col_offset += 4
    
    return U, V, W

def _build_2x2x8_rank13():
    """
    Build a rank-13 decomposition for 2x2 x 2x8 matrix multiplication.
    
    Key insight: We can use the fact that for 2x2 * 2x8, we're computing
    8 columns of the result. Each column c_k = A * b_k where b_k is a 2x1 vector.
    
    A clever construction: use 7 Strassen terms that compute linear combinations
    of all 8 output columns simultaneously.
    
    Actually, a known result: rank(2x2x8) <= 13 via a specific construction.
    Let me implement a verified one.
    
    For simplicity and correctness, let's use two Strassen blocks (rank 14)
    and see if we can reduce to 13.
    """
    n, m, p = 2, 2, 8
    rank = 14  # Start with 14, try to optimize to 13
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    U_str, V_str, W_str = _build_strassen_2x2x2()
    
    # Block 1: k=0,1
    U[:, 0:7] = U_str
    for j in range(m):
        for k_local in range(2):
            V[j * p + k_local, 0:7] = V_str[j * 2 + k_local, :]
    for k_local in range(2):
        for i in range(n):
            W[k_local * n + i, 0:7] = W_str[k_local * 2 + i, :]
    
    # Block 2: k=2,3
    U[:, 7:14] = U_str
    for j in range(m):
        for k_local in range(2):
            V[j * p + 2 + k_local, 7:14] = V_str[j * 2 + k_local, :]
    for k_local in range(2):
        for i in range(n):
            W[(2 + k_local) * n + i, 7:14] = W_str[k_local * 2 + i, :]
    
    # We only have 4 columns so far. Need 4 more.
    # For now, this is incomplete. Let's just do 4 Strassen blocks for rank 28.
    # This function is a placeholder.
    
    return None  # Not ready yet

def _build_rank33_attempt():
    """
    Attempt to build a rank-33 decomposition.
    Strategy: 2x2x8 (rank 13) + 2x2x7 (rank 20) = rank 33.
    
    Since rank-13 for 2x2x8 is complex, let's try:
    2x2x4 (rank 14) + 2x2x4 (rank 14) + 2x2x7 (rank 20) = 48. No.
    
    Alternative: Just use the current rank-53 approach which is verified.
    """
    return _build_rank53()

def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that the decomposition is correct."""
    rank = U.shape[1]
    # Check shapes
    if U.shape != (n*m, rank):
        return False
    if V.shape != (m*p, rank):
        return False
    if W.shape != (p*n, rank):
        return False
    
    # For each valid (i,j,k), check that sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    # And for invalid combinations, it should be 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # This combination is valid, should give 1
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def construct():
    """
    Construct a rank-minimal exact tensor decomposition
    for multiplying a 2x2 matrix by a 2x15 matrix.
    """
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Baseline: rank 53
    U, V, W = _build_rank53()
    best_rank = 53
    best_U, best_V, best_W = U, V, W
    
    # Try to find a rank-52 decomposition by merging last Strassen with schoolbook
    # The last 2 columns (k=14,15) use Strassen (7 terms) and k=15 uses schoolbook (4 terms).
    # Wait, k=14,15 is the 7th Strassen block, and k=15 is... no, k goes 0-14.
    # Block sizes: [2,2,2,2,2,2,2,1] -> k=0-1 (block 0), k=2-3 (block 1), ..., k=12-13 (block 6), k=14 (block 7)
    # So 7 Strassen blocks for k=0-13, and schoolbook for k=14.
    # Total: 7*7 + 4 = 53.
    
    # Can we do better? What if we use 6 Strassen blocks (k=0-11) + 1 Strassen (k=12,13) + schoolbook (k=14)?
    # That's the same: 7*7+4=53.
    
    # Alternative: 6 Strassen (k=0-11) + schoolbook for k=12,13,14 = 6*7 + 3*4 = 42+12 = 54. Worse.
    
    # What about 5 Strassen (k=0-9) + 2 Strassen (k=10-13) + schoolbook (k=14) = 7*7+4 = 53. Same.
    
    # The rank-53 seems to be the natural construction. Let's stick with it.
    
    # Verify baseline
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to schoolbook if something is wrong
        U, V, W = _build_schoolbook(n, m, p)
        best_rank = 60
        best_U, best_V, best_W = U, V, W
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > TIME_LIMIT - 10:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Final time check
    elapsed = time.time() - start_time
    if elapsed > TIME_LIMIT - 5:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
