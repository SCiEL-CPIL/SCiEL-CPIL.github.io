"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Strassen decomposition for 2x2 (Rank 7)
    # Indices: 0:00, 1:01, 2:10, 3:11
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # M1 = (A00 + A11)(B00 + B11)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[1, 0] = 1; W2[2, 0] = 1; W2[3, 0] = 1
    
    # M2 = (A10 + A11) B00
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1; W2[3, 1] = 1
    
    # M3 = A00 (B01 - B11)
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1
    
    # M4 = A11 (B10 - B00)
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[3, 3] = 1
    
    # M5 = (A00 + A01) B11
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[1, 4] = 1
    
    # M6 = (A10 - A00) (B00 + B01)
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1
    
    # M7 = (A01 - A11) (B10 + B11)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[1, 6] = 1; W2[2, 6] = -1

    # Schoolbook decomposition for 3x3 (Rank 27)
    # Indices: 0:00, 1:01, 2:02, 3:10, 4:11, 5:12, 6:20, 7:21, 8:22
    U3 = np.zeros((9, 27))
    V3 = np.zeros((9, 27))
    W3 = np.zeros((9, 27))
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U3[i * 3 + k, col] = 1
                V3[k * 3 + j, col] = 1
                W3[i * 3 + j, col] = 1
                col += 1

    # Combine using Kronecker product logic
    # Rank = 27 * 7 = 189
    rank = 27 * 7
    U = np.zeros((36, rank))
    V = np.zeros((36, rank))
    W = np.zeros((36, rank))
    
    # Indices for 6x6:
    # Block index (i, j) in 0..2
    # Inner index (u, v) in 0..1
    # Global index = i * 2 + u (for rows/cols of A/B)
    # Wait, A is 6x6. Row i is 0..5.
    # If we view as 3x3 blocks of 2x2:
    # Row r = i * 2 + u.
    # Col c = j * 2 + v.
    
    # U rows: (i, u) -> i*2+u
    # U cols: (c3, c2) -> c3 * 7 + c2
    for i in range(3):
        for u in range(2):
            row_u = i * 2 + u
            for c3 in range(27):
                for c2 in range(7):
                    col_u = c3 * 7 + c2
                    U[row_u, col_u] = U3[i, c3] * U2[u, c2]
                    
    for j in range(3):
        for v in range(2):
            row_v = j * 2 + v
            for c3 in range(27):
                for c2 in range(7):
                    col_v = c3 * 7 + c2
                    V[row_v, col_v] = V3[j, c3] * V2[v, c2]
                    
    for i in range(3):
        for u in range(2):
            row_w = i * 2 + u
            for c3 in range(27):
                for c2 in range(7):
                    col_w = c3 * 7 + c2
                    # W index convention: W[k*n+i, r]
                    # In our 3x3 x 2x2 decomposition:
                    # Outer index i (block row of C), Inner index u (row in block)
                    # So W row is i*2+u.
                    W[row_w, col_w] = W3[i, c3] * W2[u, c2]

    # Convert to list of lists for output
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
