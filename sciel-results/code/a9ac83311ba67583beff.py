"""Rewritten program for matrix multiplication 2,2,15 tensor rank minimization."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 2, 15]

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
    
    For multiplying A (2x2) by B (2x2):
    U indexes (i, j) where i is row of A, j is col of A
    V indexes (j, k) where j is row of B, k is col of B
    W indexes (k, i) where k is col of B, i is row of A
    
    Strassen's 7 terms:
    M1 = (A00 + A11) * (B00 + B11)
    M2 = (A10 + A11) * B00
    M3 = A00 * (B01 - B11)
    M4 = A11 * (B10 - B00)
    M5 = (A00 + A01) * B11
    M6 = (A10 - A00) * (B00 + B01)
    M7 = (A01 - A11) * (B10 + B11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    """
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Index mapping: U[i*m + j, r], V[j*p + k, r], W[k*n + i, r]
    # A has indices (i, j) for i in {0,1}, j in {0,1}
    # B has indices (j, k) for j in {0,1}, k in {0,1}
    # C has indices (i, k) for i in {0,1}, k in {0,1}
    # W[k*n + i, r] means W index is k*2 + i
    
    # Term 1: M1 = (A00 + A11) * (B00 + B11)
    # U: A00 (i=0,j=0) and A11 (i=1,j=1)
    U[0*2+0, 0] = 1  # A00
    U[1*2+1, 0] = 1  # A11
    # V: B00 (j=0,k=0) and B11 (j=1,k=1)
    V[0*2+0, 0] = 1  # B00
    V[1*2+1, 0] = 1  # B11
    # W: C00 (k=0,i=0) and C11 (k=1,i=1)
    W[0*2+0, 0] = 1  # C00
    W[1*2+1, 0] = 1  # C11
    
    # Term 2: M2 = (A10 + A11) * B00
    U[1*2+0, 1] = 1  # A10
    U[1*2+1, 1] = 1  # A11
    V[0*2+0, 1] = 1  # B00
    W[0*2+1, 1] = 1  # C10 (k=0, i=1)
    W[1*2+1, 1] = -1 # C11 (k=1, i=1)
    
    # Term 3: M3 = A00 * (B01 - B11)
    U[0*2+0, 2] = 1  # A00
    V[0*2+1, 2] = 1  # B01 (j=0, k=1)
    V[1*2+1, 2] = -1 # B11 (j=1, k=1)
    W[1*2+0, 2] = 1  # C01 (k=1, i=0)
    W[1*2+1, 2] = 1  # C11 (k=1, i=1)
    
    # Term 4: M4 = A11 * (B10 - B00)
    U[1*2+1, 3] = 1  # A11
    V[1*2+0, 3] = 1  # B10 (j=1, k=0)
    V[0*2+0, 3] = -1 # B00 (j=0, k=0)
    W[0*2+0, 3] = 1  # C00 (k=0, i=0)
    W[0*2+1, 3] = 1  # C10 (k=0, i=1)
    
    # Term 5: M5 = (A00 + A01) * B11
    U[0*2+0, 4] = 1  # A00
    U[0*2+1, 4] = 1  # A01 (i=0, j=1)
    V[1*2+1, 4] = 1  # B11 (j=1, k=1)
    W[0*2+0, 4] = -1 # C00 (k=0, i=0)
    W[1*2+0, 4] = 1  # C01 (k=1, i=0)
    
    # Term 6: M6 = (A10 - A00) * (B00 + B01)
    U[1*2+0, 5] = 1  # A10
    U[0*2+0, 5] = -1 # A00
    V[0*2+0, 5] = 1  # B00
    V[0*2+1, 5] = 1  # B01
    W[1*2+1, 5] = 1  # C11 (k=1, i=1)
    
    # Term 7: M7 = (A01 - A11) * (B10 + B11)
    U[0*2+1, 6] = 1  # A01
    U[1*2+1, 6] = -1 # A11
    V[1*2+0, 6] = 1  # B10
    V[1*2+1, 6] = 1  # B11
    W[0*2+0, 6] = 1  # C00 (k=0, i=0)
    
    return U, V, W

def _combine_blocks(block_decomps):
    """
    Combine independent block decompositions.
    Each block decomposition is a tuple (U_block, V_block, W_block, offset_U, offset_V, offset_W)
    where offsets specify where in the full matrices this block's indices live.
    """
    total_rank = sum(U.shape[1] for U, V, W, _, _, _ in block_decomps)
    # We need to know the full dimensions
    # For now, we'll handle this in the main construct function
    
def construct():
    """
    Construct a rank-minimal (or near-minimal) exact tensor decomposition
    for multiplying a 2x2 matrix by a 2x15 matrix.
    
    Strategy: Decompose the 2x15 matrix into seven 2x2 blocks and one 2x1 block.
    Use Strassen (rank 7) for each 2x2 block and schoolbook (rank 4) for the 2x1 block.
    Total rank = 7 * 7 + 4 = 53.
    
    The 2x15 matrix B can be partitioned as:
    B = [B1 | B2 | B3 | B4 | B5 | B6 | B7 | B8]
    where B1..B7 are 2x2 blocks and B8 is a 2x1 block.
    
    Actually, 2x15 = 7 * 2x2 blocks + 1 * 2x1 block.
    7*2 + 1 = 15. Yes.
    
    For the 2x2 blocks, we use Strassen's rank-7 decomposition.
    For the 2x1 block, we use schoolbook rank-4 (2*2*1 = 4).
    
    Total rank = 7*7 + 4 = 53.
    """
    n, m, p = DIMENSIONS  # 2, 2, 15
    
    # Partition the p=15 columns into 7 blocks of size 2 and 1 block of size 1
    # Block sizes: [2, 2, 2, 2, 2, 2, 2, 1]
    block_sizes = [2, 2, 2, 2, 2, 2, 2, 1]
    num_blocks = len(block_sizes)
    
    # Get Strassen decomposition for 2x2x2
    U_str, V_str, W_str = _build_strassen_2x2x2()
    
    # Get schoolbook decomposition for 2x2x1
    U_sb, V_sb, W_sb = _build_schoolbook(2, 2, 1)
    
    # Total rank
    total_rank = num_blocks * 7  # 7 blocks of size 2 use Strassen (rank 7), 1 block of size 1 uses schoolbook (rank 4)
    # Wait: 7 blocks of size 2 => 7 * 7 = 49, plus 1 block of size 1 => 4. Total = 53.
    total_rank = 7 * 7 + 4  # 53
    
    # Build full U, V, W matrices
    U = np.zeros((n * m, total_rank))  # 4 x 53
    V = np.zeros((m * p, total_rank))  # 30 x 53
    W = np.zeros((p * n, total_rank))  # 30 x 53
    
    col_offset = 0
    for b in range(num_blocks):
        block_size = block_sizes[b]
        start_k = sum(block_sizes[:b])  # Starting column index in B
        
        if block_size == 2:
            # Use Strassen decomposition
            U[:, col_offset:col_offset+7] = U_str
            # V: for this block, the j*p+k indexing needs to account for the block offset
            # V has shape (m*p, rank) = (2*15, 7) for this block
            # For the block starting at column start_k, V[j*15 + (start_k + k_local), col]
            for j in range(m):
                for k_local in range(2):
                    k_global = start_k + k_local
                    V[j * p + k_global, col_offset:col_offset+7] = V_str[j * 2 + k_local, :]
            # W: W[k*n + i, col] where k is the column index in C
            # For this block, k_local ranges over 0,1, and k_global = start_k + k_local
            for k_local in range(2):
                k_global = start_k + k_local
                for i in range(n):
                    W[k_global * n + i, col_offset:col_offset+7] = W_str[k_local * 2 + i, :]
            col_offset += 7
        else:
            # block_size == 1, use schoolbook
            U[:, col_offset:col_offset+4] = U_sb
            # V: for this block, k_local = 0, k_global = start_k
            for j in range(m):
                V[j * p + start_k, col_offset:col_offset+4] = V_sb[j * 1 + 0, :]
            # W: k_local = 0, k_global = start_k
            for i in range(n):
                W[start_k * n + i, col_offset:col_offset+4] = W_sb[0 * 2 + i, :]
            col_offset += 4
    
    # Convert to lists as required by the interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
