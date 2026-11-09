"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def _get_strassen_2x2x2():
    """Returns U, V, W for 2x2 by 2x2 matrix multiplication using Strassen."""
    # U shape: (4, 7), V shape: (4, 7), W shape: (4, 7)
    # Indices: i*m+j for U, j*p+k for V, k*n+i for W (Wait, W is p*n? No, W is n*p? 
    # Task says W[k*n+i, r]. Here n=2, p=2. So W is 4x7.
    # A is 2x2, B is 2x2, C is 2x2.
    # A indices: 0..3. B indices: 0..3. C indices: 0..3.
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A00+A11)(B00+B11)
    U[0,0] = 1; U[3,0] = 1
    V[0,0] = 1; V[3,0] = 1
    W[0,0] = 1; W[1,0] = 1; W[2,0] = 1; W[3,0] = 1
    
    # M2 = (A10+A11)B00
    U[2,1] = 1; U[3,1] = 1
    V[0,1] = 1
    W[2,1] = 1; W[3,1] = -1
    
    # M3 = A00(B01-B11)
    U[0,2] = 1
    V[1,2] = 1; V[3,2] = -1
    W[1,2] = 1; W[3,2] = 1
    
    # M4 = A11(B10-B00)
    U[3,3] = 1
    V[2,3] = 1; V[0,3] = -1
    W[2,3] = 1; W[3,3] = 1
    
    # M5 = (A00+A01)B11
    U[0,4] = 1; U[1,4] = 1
    V[3,4] = 1
    W[0,4] = -1; W[1,4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U[2,5] = 1; U[0,5] = -1
    V[0,5] = 1; V[1,5] = 1
    W[3,5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1,6] = 1; U[3,6] = -1
    V[2,6] = 1; V[3,6] = 1
    W[0,6] = 1; W[3,6] = -1
    
    return U, V, W

def _get_trivial_2x2x1():
    """Returns U, V, W for 2x2 by 2x1 matrix multiplication."""
    # A 2x2, B 2x1, C 2x1.
    # U shape: (4, 4), V shape: (2, 4), W shape: (2, 4)
    # C0 = A00*B00 + A01*B10
    # C1 = A10*B00 + A11*B10
    
    U = np.zeros((4, 4))
    V = np.zeros((2, 4))
    W = np.zeros((2, 4))
    
    # Term 1: A00 B00 -> C0
    U[0,0] = 1
    V[0,0] = 1
    W[0,0] = 1
    
    # Term 2: A01 B10 -> C0
    U[1,1] = 1
    V[1,1] = 1
    W[0,1] = 1
    
    # Term 3: A10 B00 -> C1
    U[2,2] = 1
    V[0,2] = 1
    W[1,2] = 1
    
    # Term 4: A11 B10 -> C1
    U[3,3] = 1
    V[1,3] = 1
    W[1,3] = 1
    
    return U, V, W

def _get_trivial_2x1x9():
    """Returns U, V, W for 2x1 by 1x9 matrix multiplication."""
    # A 2x1, B 1x9, C 2x9.
    # U shape: (2, 18), V shape: (9, 18), W shape: (18, 18)
    # C_i,k = A_i,0 * B_0,k
    
    U = np.zeros((2, 18))
    V = np.zeros((9, 18))
    W = np.zeros((18, 18))
    
    col = 0
    for i in range(2):
        for k in range(9):
            U[i, col] = 1
            V[k, col] = 1
            W[k*2 + i, col] = 1
            col += 1
            
    return U, V, W

def _combine_decomps(decomps, offsets):
    """
    Combines a list of decompositions.
    decomps: list of (U, V, W)
    offsets: list of (row_offset_U, row_offset_V, row_offset_W)
    Returns combined U, V, W.
    """
    total_rank = sum(U.shape[1] for U, V, W in decomps)
    
    # Determine max dimensions
    max_U_rows = max(U.shape[0] for U, V, W in decomps)
    max_V_rows = max(V.shape[0] for U, V, W in decomps)
    max_W_rows = max(W.shape[0] for U, V, W in decomps)
    
    U_full = np.zeros((max_U_rows, total_rank))
    V_full = np.zeros((max_V_rows, total_rank))
    W_full = np.zeros((max_W_rows, total_rank))
    
    col_start = 0
    for (U, V, W), (off_U, off_V, off_W) in zip(decomps, offsets):
        r = U.shape[1]
        U_full[off_U:off_U+U.shape[0], col_start:col_start+r] = U
        V_full[off_V:off_V+V.shape[0], col_start:col_start+r] = V
        W_full[off_W:off_W+W.shape[0], col_start:col_start+r] = W
        col_start += r
        
    return U_full, V_full, W_full

def construct():
    start_time = time.time()
    deadline = start_time + 110
    n, m, p = DIMENSIONS # 2, 3, 9
    
    # We are building a decomposition for 2x3 by 3x9.
    # U shape: (2*3, rank) = (6, rank)
    # V shape: (3*9, rank) = (27, rank)
    # W shape: (9*2, rank) = (18, rank)
    
    # Split A (2x3) into A1 (2x2) and A2 (2x1).
    # A1 corresponds to rows 0..5 of U? No.
    # U indices: i*m+j.
    # A1 (cols 0,1): i in 0..1, j in 0..1. Indices: 0,1,2,3.
    # A2 (col 2): i in 0..1, j=2. Indices: 4,5.
    
    # Split B (3x9) into B1 (2x9) and B2 (1x9).
    # B1 (rows 0,1): j in 0..1, k in 0..8.
    # V indices: j*p+k.
    # B1: j=0, k=0..8 -> 0..8. j=1, k=0..8 -> 9..17.
    # B2 (row 2): j=2, k=0..8. Indices: 18..26.
    
    # Decomposition parts:
    # 1. A1 * B1_0 (2x2 by 2x3, B1 cols 0,1,2)
    # 2. A1 * B1_1 (2x2 by 2x3, B1 cols 3,4,5)
    # 3. A1 * B1_2 (2x2 by 2x3, B1 cols 6,7,8)
    # 4. A2 * B2 (2x1 by 1x9)
    
    # Get base decompositions
    U_s, V_s, W_s = _get_strassen_2x2x2()
    U_t, V_t, W_t = _get_trivial_2x2x1()
    U_21, V_21, W_21 = _get_trivial_2x1x9()
    
    # Construct 2x2 by 2x3 decomposition
    # U is 4x11. V is 6x11 (2x3). W is 6x11 (2x3).
    # Strassen part: U_s (4x7), V_s (4x7) [maps to B cols 0,1], W_s (4x7) [maps to C cols 0,1]
    # Trivial part: U_t (4x4), V_t (2x4) [maps to B col 2], W_t (2x4) [maps to C col 2]
    
    # We need to adjust V_s and W_s indices to match the 2x3 block.
    # In 2x2x2, V is 2x2. In 2x2x3, V is 2x3.
    # Strassen uses B cols 0,1. So V_s fits in V[0:2, 0:2] of the 2x3 block?
    # Yes, if we place it correctly.
    
    # Let's build the 2x2x3 block decomposition manually to be safe.
    U_block = np.zeros((4, 11))
    V_block = np.zeros((6, 11)) # 2x3
    W_block = np.zeros((6, 11)) # 2x3
    
    # Copy Strassen
    U_block[:, 0:7] = U_s
    # V_s is 4x7. Indices j*p+k. p=2.
    # In 2x3 block, p=3.
    # V_s[0] is j=0,k=0. V_s[1] is j=0,k=1. V_s[2] is j=1,k=0. V_s[3] is j=1,k=1.
    # In 2x3, j=0,k=0 is 0. j=0,k=1 is 1. j=1,k=0 is 3. j=1,k=1 is 4.
    # So we need to map indices.
    V_block[0, 0:7] = V_s[0, 0:7] # j=0,k=0
    V_block[1, 0:7] = V_s[1, 0:7] # j=0,k=1
    V_block[3, 0:7] = V_s[2, 0:7] # j=1,k=0
    V_block[4, 0:7] = V_s[3, 0:7] # j=1,k=1
    
    # W_s is 4x7. Indices k*n+i. n=2.
    # In 2x3 block, n=2.
    # W_s[0] is k=0,i=0. W_s[1] is k=0,i=1. W_s[2] is k=1,i=0. W_s[3] is k=1,i=1.
    # In 2x3, k=0,i=0 is 0. k=0,i=1 is 1. k=1,i=0 is 2. k=1,i=1 is 3.
    # So W_s fits directly in W_block[0:4, 0:7].
    W_block[0:4, 0:7] = W_s
    
    # Copy Trivial (2x2 by 2x1)
    U_block[:, 7:11] = U_t
    # V_t is 2x4. Indices j*p+k. p=1.
    # V_t[0] is j=0,k=0. V_t[1] is j=1,k=0.
    # In 2x3 block, this corresponds to col 2.
    # j=0,k=2 is 2. j=1,k=2 is 5.
    V_block[2, 7:11] = V_t[0, 7:11]
    V_block[5, 7:11] = V_t[1, 7:11]
    
    # W_t is 2x4. Indices k*n+i. n=2.
    # W_t[0] is k=0,i=0. W_t[1] is k=0,i=1.
    # In 2x3 block, this corresponds to col 2.
    # k=2,i=0 is 4. k=2,i=1 is 5.
    W_block[4, 7:11] = W_t[0, 7:11]
    W_block[5, 7:11] = W_t[1, 7:11]
    
    # Now we have a 2x2 by 2x3 decomposition (U_block, V_block, W_block).
    # We need to use this 3 times for the 3 blocks of B1.
    
    decomps = []
    offsets = []
    
    # Block 0: B1 cols 0,1,2.
    # U offset: 0 (A1)
    # V offset: 0 (B1 rows 0,1, cols 0,1,2)
    # W offset: 0 (C cols 0,1,2)
    decomps.append((U_block, V_block, W_block))
    offsets.append((0, 0, 0))
    
    # Block 1: B1 cols 3,4,5.
    # U offset: 0
    # V offset: 3 (cols 3,4,5)
    # W offset: 3 (cols 3,4,5)
    decomps.append((U_block, V_block, W_block))
    offsets.append((0, 3, 3))
    
    # Block 2: B1 cols 6,7,8.
    # U offset: 0
    # V offset: 6 (cols 6,7,8)
    # W offset: 6 (cols 6,7,8)
    decomps.append((U_block, V_block, W_block))
    offsets.append((0, 6, 6))
    
    # Part 4: A2 * B2 (2x1 by 1x9)
    # U offset: 4 (A2 is rows 4,5 of A)
    # V offset: 18 (B2 is row 2 of B, which is indices 18..26 in V)
    # W offset: 0? No, W is 18x18.
    # W_21 is 18x18.
    # It writes to all C columns?
    # A2 B2 produces C[0:2, 0:9].
    # So W offset 0.
    decomps.append((U_21, V_21, W_21))
    offsets.append((4, 18, 0))
    
    # Combine
    U_full, V_full, W_full = _combine_decomps(decomps, offsets)
    
    # Convert to lists
    return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
# EVOLVE-BLOCK-END
