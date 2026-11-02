import numpy as np
import time

DIMENSIONS = [3, 5, 6]

def _get_strassen_2x2():
    """
    Returns U, V, W for 2x2x2 matrix multiplication using Strassen's algorithm.
    U: 4x7, V: 4x7, W: 4x7
    Indices: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    rank = 7
    U = np.zeros((4, rank))
    V = np.zeros((4, rank))
    W = np.zeros((4, rank))
    
    # A indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # B indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # C indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    # M1 = A00 + A11
    U[0, 0] = 1
    U[3, 0] = 1
    V[0, 0] = 1
    V[3, 0] = 1
    W[0, 0] = 1   # C00 += M1
    W[3, 0] = 1   # C11 += M1
    
    # M2 = A10 + A11
    U[2, 1] = 1
    U[3, 1] = 1
    V[0, 1] = 1
    V[3, 1] = 1
    W[3, 1] = -1  # C11 -= M2
    
    # M3 = A00 * B01
    U[0, 2] = 1
    V[1, 2] = 1
    W[1, 2] = 1   # C01 += M3
    W[3, 2] = 1   # C11 += M3
    
    # M4 = A11 * B10
    U[3, 3] = 1
    V[2, 3] = 1
    W[0, 3] = 1   # C00 += M4
    W[2, 3] = 1   # C10 += M4
    
    # M5 = (A00 + A11) * B11
    U[0, 4] = 1
    U[3, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1  # C00 -= M5
    W[1, 4] = 1   # C01 += M5
    
    # M6 = (A10 - A00) * (B00 + B01)
    U[2, 5] = 1
    U[0, 5] = -1
    V[0, 5] = 1
    V[1, 5] = 1
    W[2, 5] = 1   # C10 += M6
    W[3, 5] = 1   # C11 += M6
    
    # M7 = (A01 + A11) * (B10 - B11)
    U[1, 6] = 1
    U[3, 6] = 1
    V[2, 6] = 1
    V[3, 6] = -1
    W[0, 6] = 1   # C00 += M7
    
    return U, V, W

def _get_trivial_1x2x2():
    """
    Returns U, V, W for 1x2x2 matrix multiplication.
    U: 2x1, V: 4x1, W: 2x1
    Indices: U[i*2+j, c] -> U[j, c] for i=0
    V[j*2+k, c]
    W[k*1+i, c] -> W[k, c] for i=0
    """
    rank = 1
    U = np.zeros((2, rank))
    V = np.zeros((4, rank))
    W = np.zeros((2, rank))
    
    # C0 = A0*B0 + A1*B1
    # U[0]=1 (A0), U[1]=1 (A1)
    # V[0]=1 (B00), V[1]=1 (B01), V[2]=1 (B10), V[3]=1 (B11)
    # W[0]=1 (C0), W[1]=1 (C1)
    
    U[0, 0] = 1
    U[1, 0] = 1
    V[0, 0] = 1
    V[1, 0] = 1
    V[2, 0] = 1
    V[3, 0] = 1
    W[0, 0] = 1
    W[1, 0] = 1
    
    return U, V, W

def _get_block_3x2x2():
    """
    Combines Strassen (for rows 0,1) and Trivial (for row 2) to form 3x2x2.
    U: 6x8, V: 4x8, W: 6x8
    U indices: i*2+j
    V indices: j*2+k
    W indices: k*3+i
    """
    U_s, V_s, W_s = _get_strassen_2x2()
    U_t, V_t, W_t = _get_trivial_1x2x2()
    
    rank = 8
    U = np.zeros((6, rank))
    V = np.zeros((4, rank))
    W = np.zeros((6, rank))
    
    # Place Strassen (rank 7)
    # U_s is 4x7 (rows 0,1 of A)
    U[0:4, 0:7] = U_s
    V[0:4, 0:7] = V_s
    # W_s is 4x7 (cols 0,1 of C, rows 0,1)
    # W indices for C[i,k] is k*3+i
    # W_s[k*2+i, c] corresponds to C[i,k] for i in 0,1
    # We need to map W_s[k*2+i, c] to W[k*3+i, c]
    for k in range(2):
        for i in range(2):
            W[k*3+i, 0:7] = W_s[k*2+i, 0:7]
            
    # Place Trivial (rank 1)
    # U_t is 2x1 (row 2 of A)
    U[4:6, 7] = U_t
    V[0:4, 7] = V_t
    # W_t is 2x1 (cols 0,1 of C, row 2)
    # W_t[k, c] corresponds to C[2,k]
    # Map to W[k*3+2, c]
    for k in range(2):
        W[k*3+2, 7] = W_t[k, 0]
        
    return U, V, W

def _get_block_3x1x6():
    """
    Returns U, V, W for 3x1x6 matrix multiplication.
    U: 3x1, V: 6x1, W: 18x1
    """
    rank = 1
    U = np.zeros((3, rank))
    V = np.zeros((6, rank))
    W = np.zeros((18, rank))
    
    # C[i,k] = A[i,0] * B[0,k]
    U[0, 0] = 1
    U[1, 0] = 1
    U[2, 0] = 1
    V[0, 0] = 1
    V[1, 0] = 1
    V[2, 0] = 1
    V[3, 0] = 1
    V[4, 0] = 1
    V[5, 0] = 1
    for k in range(6):
        for i in range(3):
            W[k*3+i, 0] = 1
            
    return U, V, W

def construct():
    start_time = time.time()
    
    # Get base blocks
    U_322, V_322, W_322 = _get_block_3x2x2() # Rank 8
    U_316, V_316, W_316 = _get_block_3x1x6() # Rank 1
    
    # Target: 3x5x6
    # Split inner dim 5 into 1 + 4
    # Part 1: 3x1x6 (Rank 1)
    # Part 2: 3x4x6
    # Split 4 into 2 + 2
    # Part 2a: 3x2x6
    # Split 6 into 2 + 2 + 2
    # Part 2a1: 3x2x2 (Rank 8)
    # Part 2a2: 3x2x2 (Rank 8)
    # Part 2a3: 3x2x2 (Rank 8)
    # Total Rank = 1 + 8 + 8 + 8 + 8 + 8 = 41?
    # Wait.
    # 3x4x6 = 2 * (3x2x6)
    # 3x2x6 = 3 * (3x2x2)
    # So 3x4x6 = 6 * (3x2x2).
    # Rank = 6 * 8 = 48.
    # Total Rank = 1 + 48 = 49.
    
    total_rank = 49
    
    U = np.zeros((15, total_rank))
    V = np.zeros((30, total_rank))
    W = np.zeros((18, total_rank))
    
    col_offset = 0
    
    # Add 3x1x6 block
    U[0:3, col_offset:col_offset+1] = U_316
    V[0:6, col_offset:col_offset+1] = V_316
    W[0:18, col_offset:col_offset+1] = W_316
    col_offset += 1
    
    # Add 6 blocks of 3x2x2
    # Indices for A (3x4):
    # Rows 0-2. Cols 1-4 (since col 0 was used in 3x1x6).
    # Block 1: Cols 1-2 (indices 1,2 in A)
    # Block 2: Cols 3-4 (indices 3,4 in A)
    # Wait, the 4x6 matrix B has rows 1-4 (indices 1,2,3,4).
    # We split B's 6 columns into 3 pairs.
    # We split B's 4 rows into 2 pairs.
    
    # Let's iterate through the 2 row-pairs of B and 3 col-pairs of B
    # B rows: 1,2 and 3,4
    # B cols: 0,1 and 2,3 and 4,5
    
    # A rows: 0,1,2
    # A cols: 1,2 and 3,4
    
    # Block (b_row_pair, b_col_pair):
    # A uses cols corresponding to b_row_pair.
    # B uses rows corresponding to b_row_pair, cols corresponding to b_col_pair.
    # C uses cols corresponding to b_col_pair.
    
    b_row_pairs = [(1, 2), (3, 4)]
    b_col_pairs = [(0, 1), (2, 3), (4, 5)]
    
    for br1, br2 in b_row_pairs:
        a_col_1 = br1 - 1 # A col index (0-based)
        a_col_2 = br2 - 1
        for bc1, bc2 in b_col_pairs:
            # Extract 3x2x2 block
            # U: Rows 0,1,2. Cols a_col_1, a_col_2.
            # V: Rows br1, br2. Cols bc1, bc2.
            # W: Cols bc1, bc2. Rows 0,1,2.
            
            # U_322 is 6x8. Indices i*2+j.
            # We need to map U_322[i*2+j] to U[i*5 + a_col_j]
            for i in range(3):
                U[i*5 + a_col_1, col_offset:col_offset+8] = U_322[i*2+0, :]
                U[i*5 + a_col_2, col_offset:col_offset+8] = U_322[i*2+1, :]
                
            # V_322 is 4x8. Indices j*2+k.
            # We need to map V_322[j*2+k] to V[br_j*6 + bc_k]
            # j=0 corresponds to row br1, j=1 to br2.
            # k=0 corresponds to col bc1, k=1 to bc2.
            V[br1*6 + bc1, col_offset:col_offset+8] = V_322[0*2+0, :]
            V[br1*6 + bc2, col_offset:col_offset+8] = V_322[0*2+1, :]
            V[br2*6 + bc1, col_offset:col_offset+8] = V_322[1*2+0, :]
            V[br2*6 + bc2, col_offset:col_offset+8] = V_322[1*2+1, :]
            
            # W_322 is 6x8. Indices k*3+i.
            # We need to map W_322[k*3+i] to W[bc_k*3 + i]
            W[bc1*3 + 0, col_offset:col_offset+8] = W_322[0*3+0, :]
            W[bc1*3 + 1, col_offset:col_offset+8] = W_322[0*3+1, :]
            W[bc1*3 + 2, col_offset:col_offset+8] = W_322[0*3+2, :]
            W[bc2*3 + 0, col_offset:col_offset+8] = W_322[1*3+0, :]
            W[bc2*3 + 1, col_offset:col_offset+8] = W_322[1*3+1, :]
            W[bc2*3 + 2, col_offset:col_offset+8] = W_322[1*3+2, :]
            
            col_offset += 8
            
    if col_offset != total_rank:
        print(f"Warning: col_offset {col_offset} != total_rank {total_rank}")
        
    # Convert to lists for return
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

if __name__ == "__main__":
    res = construct()
    print("Constructed")
