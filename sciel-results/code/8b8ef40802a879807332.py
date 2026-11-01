import time
import numpy as np

def construct():
    """
    Constructs a rank-minimal exact tensor decomposition for multiplying
    a 2x2 matrix by a 2x8 matrix.
    
    The strategy uses the block-diagonal structure of the problem.
    The 2x8 matrix B can be viewed as four 2x2 blocks.
    The multiplication A * B is equivalent to four independent multiplications
    of A (2x2) by each 2x2 block of B.
    
    The minimal rank for 2x2 matrix multiplication is 7 (Strassen's algorithm).
    Therefore, the minimal rank for 2x2 * 2x8 is 4 * 7 = 28.
    
    The output follows the trace convention W[k*n + i, r].
    """
    
    # Deadline guard: 120s limit, use 150s wall-clock with safety margin
    # Since this is a deterministic construction, it completes in microseconds.
    # We just ensure we don't accidentally enter a long-running path.
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    n, m, p = 2, 2, 8
    rank = 28
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen factors for 2x2 x 2x2 -> 2x2
    # A indices: 00, 01, 10, 11  (row-major: A[0,0], A[0,1], A[1,0], A[1,1])
    # B indices: 00, 01, 10, 11
    # C indices: 00, 01, 10, 11
    
    # U: Coefficients for A (rows indexed by A entries, columns by Strassen products)
    U_strassen = [
        [1, 0, 0, 1],   # P1 = (A00+A11)*(B00+B11)
        [0, 0, 1, 1],   # P2 = (A10+A11)*B00
        [1, 0, 0, 0],   # P3 = A00*(B01-B11)
        [0, 0, 0, 1],   # P4 = A11*(B10-B00)
        [1, 1, 0, 0],   # P5 = (A00+A01)*B11
        [-1, 0, 1, 0],  # P6 = (A10-A00)*(B00+B01)
        [0, 1, 0, -1]   # P7 = (A01-A11)*(B10+B11)
    ]
    
    # V: Coefficients for B
    V_strassen = [
        [1, 0, 0, 1],   # P1
        [1, 0, 0, 0],   # P2
        [0, 1, 0, -1],  # P3
        [-1, 0, 1, 0],  # P4
        [0, 0, 0, 1],   # P5
        [1, 1, 0, 0],   # P6
        [0, 0, 1, 1]    # P7
    ]
    
    # W: Coefficients for C
    W_strassen = [
        [1, 0, 0, 1],   # P1
        [0, 0, 1, -1],  # P2
        [0, 1, 0, 1],   # P3
        [1, 0, 1, 0],   # P4
        [-1, 1, 0, 0],  # P5
        [0, 0, 0, 1],   # P6
        [1, 0, 0, 0]    # P7
    ]
    
    num_blocks = p // 2  # 4 blocks of 2 columns
    
    col = 0
    for t in range(num_blocks):
        for r in range(7):
            # Fill U: Same for all blocks
            for i in range(4):
                U[i, col] = U_strassen[r][i]
                
            # Fill V: Indices for B[row, col]
            # B is 2x8. Global index = row * 8 + col_idx.
            # Block t corresponds to B columns 2t and 2t+1.
            # Local B indices 0,1,2,3 map to:
            # 0: B[0, 2t]   -> 0*8 + 2t
            # 1: B[0, 2t+1] -> 0*8 + 2t+1
            # 2: B[1, 2t]   -> 1*8 + 2t
            # 3: B[1, 2t+1] -> 1*8 + 2t+1
            v_indices = [2*t, 2*t+1, 8+2*t, 8+2*t+1]
            for l in range(4):
                V[v_indices[l], col] = V_strassen[r][l]
                
            # Fill W: Indices for C[i, k]
            # C is 2x8. Convention W[k*n + i, r].
            # Block t corresponds to C columns 2t and 2t+1.
            # Local C indices 0,1,2,3 map to:
            # 0: C[0, 2t]   -> i=0, k=2t   -> 2t*2 + 0
            # 1: C[0, 2t+1] -> i=0, k=2t+1 -> (2t+1)*2 + 0
            # 2: C[1, 2t]   -> i=1, k=2t   -> 2t*2 + 1
            # 3: C[1, 2t+1] -> i=1, k=2t+1 -> (2t+1)*2 + 1
            w_indices = [4*t, 4*t+2, 4*t+1, 4*t+3]
            for l in range(4):
                W[w_indices[l], col] = W_strassen[r][l]
                
            col += 1
            
            # Deadline check (should never trigger for deterministic construction)
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
