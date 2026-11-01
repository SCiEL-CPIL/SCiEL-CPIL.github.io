import time
import numpy as np

DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START

def _build_strassen_222():
    """
    Returns U, V, W for M_{2,2,2} with rank 7.
    U: 4x7, V: 4x7, W: 4x7
    """
    U = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ])
    V = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ])
    W = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ])
    return U, V, W

def _build_school_221():
    """
    Returns U, V, W for M_{2,2,1} with rank 4.
    U: 4x4, V: 2x4, W: 2x4
    """
    U = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    V = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1]
    ])
    W = np.array([
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ])
    return U, V, W

def _build_223():
    """
    Builds M_{2,2,3} by direct sum of M_{2,2,2} and M_{2,2,1}.
    Rank = 7 + 4 = 11.
    U: 4x11
    V: 6x11 (rows 0,1 for 222, row 2 for 221)
    W: 6x11 (cols 0,1 for 222, col 2 for 221)
    """
    U1, V1, W1 = _build_strassen_222()
    U2, V2, W2 = _build_school_221()
    
    # Concatenate U, V, W
    # U is 4x7 and 4x4 -> 4x11
    U = np.hstack([U1, U2])
    
    # V is 4x7 and 2x4.
    # V1 rows 0,1 correspond to local 0,1.
    # V2 row 0 corresponds to local 2? 
    # Wait, V for M_{2,2,1} has 2 rows (j=0,1).
    # In M_{2,2,3}, the 3rd column (k=2) corresponds to the 2x1 multiplication.
    # So V2 (2x4) should be placed at rows 2? 
    # No, V index is j*p + k.
    # For M_{2,2,3}, p=3.
    # V1 is for k=0,1. Rows j*3+k.
    # If we treat it as a block, local V indices are 0..5.
    # M_{2,2,2} uses k=0,1. V indices:
    # j=0: k=0 -> 0, k=1 -> 1
    # j=1: k=0 -> 3, k=1 -> 4
    # M_{2,2,1} uses k=2. V indices:
    # j=0: k=2 -> 2
    # j=1: k=2 -> 5
    # So V1 (4x7) needs to be scattered into rows 0,1,3,4.
    # V2 (2x4) needs to be scattered into rows 2,5.
    
    V = np.zeros((6, 11), dtype=int)
    # Place V1
    V[0, :7] = V1[0, :] # j=0, k=0
    V[1, :7] = V1[1, :] # j=0, k=1
    V[3, :7] = V1[2, :] # j=1, k=0
    V[4, :7] = V1[3, :] # j=1, k=1
    
    # Place V2
    V[2, 7:11] = V2[0, :] # j=0, k=2
    V[5, 7:11] = V2[1, :] # j=1, k=2
    
    # W is 4x7 and 2x4.
    # W index is k*n + i.
    # M_{2,2,2} uses k=0,1.
    # k=0: i=0,1 -> 0,1
    # k=1: i=0,1 -> 2,3
    # M_{2,2,1} uses k=2.
    # k=2: i=0,1 -> 4,5
    W = np.zeros((6, 11), dtype=int)
    W[0, :7] = W1[0, :] # k=0, i=0
    W[1, :7] = W1[1, :] # k=0, i=1
    W[2, :7] = W1[2, :] # k=1, i=0
    W[3, :7] = W1[3, :] # k=1, i=1
    
    W[4, 7:11] = W2[0, :] # k=2, i=0
    W[5, 7:11] = W2[1, :] # k=2, i=1
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    n, m, p = DIMENSIONS
    # 6 blocks of 222, 1 block of 223
    rank = 6 * 7 + 11
    R = rank
    
    U = np.zeros((n*m, R), dtype=int)
    V = np.zeros((m*p, R), dtype=int)
    W = np.zeros((p*n, R), dtype=int)
    
    # Fill 6 blocks of M_{2,2,2}
    col_offset = 0
    for b in range(6):
        U_b, V_b, W_b = _build_strassen_222()
        
        # U: same for all blocks
        U[:, col_offset:col_offset+7] = U_b
        
        # V: indices j*p + k
        # Block b uses k = 2b, 2b+1
        # V_b rows 0,1 are for j=0 (k=0,1)
        # V_b rows 2,3 are for j=1 (k=0,1)
        # Global V rows:
        # j=0: 0*p + 2b = 2b, 0*p + 2b+1 = 2b+1
        # j=1: 1*p + 2b = p+2b, 1*p + 2b+1 = p+2b+1
        V[2*b, col_offset:col_offset+7] = V_b[0, :]
        V[2*b+1, col_offset:col_offset+7] = V_b[1, :]
        V[p + 2*b, col_offset:col_offset+7] = V_b[2, :]
        V[p + 2*b+1, col_offset:col_offset+7] = V_b[3, :]
        
        # W: indices k*n + i
        # Block b uses k = 2b, 2b+1
        # W_b rows 0,1 are for k=0 (i=0,1)
        # W_b rows 2,3 are for k=1 (i=0,1)
        # Global W rows:
        # k=2b: 2b*n, 2b*n+1
        # k=2b+1: (2b+1)*n, (2b+1)*n+1
        W[2*b*n, col_offset:col_offset+7] = W_b[0, :]
        W[2*b*n+1, col_offset:col_offset+7] = W_b[1, :]
        W[(2*b+1)*n, col_offset:col_offset+7] = W_b[2, :]
        W[(2*b+1)*n+1, col_offset:col_offset+7] = W_b[3, :]
        
        col_offset += 7
        
    # Fill 1 block of M_{2,2,3}
    U_c, V_c, W_c = _build_223()
    
    U[:, col_offset:col_offset+11] = U_c
    
    # V: indices j*p + k
    # Block 6 uses k = 12, 13, 14
    # V_c rows 0,1 are for j=0 (k=0,1 local -> 12,13 global)
    # V_c row 2 is for j=0 (k=2 local -> 14 global)
    # V_c row 3 is for j=1 (k=0 local -> p+12 global)
    # V_c row 4 is for j=1 (k=1 local -> p+13 global)
    # V_c row 5 is for j=1 (k=2 local -> p+14 global)
    
    V[12, col_offset:col_offset+11] = V_c[0, :]
    V[13, col_offset:col_offset+11] = V_c[1, :]
    V[14, col_offset:col_offset+11] = V_c[2, :]
    V[p+12, col_offset:col_offset+11] = V_c[3, :]
    V[p+13, col_offset:col_offset+11] = V_c[4, :]
    V[p+14, col_offset:col_offset+11] = V_c[5, :]
    
    # W: indices k*n + i
    # Block 6 uses k = 12, 13, 14
    # W_c rows 0,1 are for k=0 (i=0,1 local -> 12n, 12n+1 global)
    # W_c rows 2,3 are for k=1 (i=0,1 local -> 13n, 13n+1 global)
    # W_c rows 4,5 are for k=2 (i=0,1 local -> 14n, 14n+1 global)
    
    W[12*n, col_offset:col_offset+11] = W_c[0, :]
    W[12*n+1, col_offset:col_offset+11] = W_c[1, :]
    W[13*n, col_offset:col_offset+11] = W_c[2, :]
    W[13*n+1, col_offset:col_offset+11] = W_c[3, :]
    W[14*n, col_offset:col_offset+11] = W_c[4, :]
    W[14*n+1, col_offset:col_offset+11] = W_c[5, :]
    
    if time.time() > deadline:
        # Fallback
        rank_fb = n * m * p
        U_fb = [[0] * rank_fb for _ in range(n * m)]
        V_fb = [[0] * rank_fb for _ in range(m * p)]
        W_fb = [[0] * rank_fb for _ in range(p * n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * m * p + j * p + k
                    U_fb[i * m + j][r] = 1
                    V_fb[j * p + k][r] = 1
                    W_fb[k * n + i][r] = 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
        
    # Convert to list of lists if required, but numpy arrays are usually accepted or converted
    # The prompt says "factor matrices of shapes ... with integer/half-integer entries"
    # Returning numpy arrays is standard.
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
