import time
import sys

def construct():
    start_time = time.time()
    timeout = 110.0  # Safety margin
    
    n, m, p = 2, 4, 4
    # Rank 28 decomposition
    # U shape: (n*m, rank) = (8, 28)
    # V shape: (m*p, rank) = (16, 28)
    # W shape: (p*n, rank) = (8, 28)
    
    rank = 28
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Strassen-like decomposition for 2x2 blocks
    # We have 4 blocks of 2x2x2 multiplication
    # Block (block_a_row, block_b_col)
    # A is split into Left (cols 0,1) and Right (cols 2,3)
    # B is split into Top-Left, Top-Right, Bottom-Left, Bottom-Right
    
    def add_strassen_block(col_offset, a_rows, b_rows, c_rows):
        """
        Adds a 2x2x2 Strassen decomposition.
        a_rows: indices in U for the 4 elements of the 2x2 A block
                [A00, A01, A10, A11]
        b_rows: indices in V for the 4 elements of the 2x2 B block
                [B00, B01, B10, B11]
        c_rows: indices in W for the 4 elements of the 2x2 C block
                [C00, C01, C10, C11]
        col_offset: starting column index in U, V, W
        """
        # M1 = (A00 + A11) * (B00 + B11)
        U[a_rows[0]][col_offset + 0] += 1.0
        U[a_rows[3]][col_offset + 0] += 1.0
        V[b_rows[0]][col_offset + 0] += 1.0
        V[b_rows[3]][col_offset + 0] += 1.0
        W[c_rows[0]][col_offset + 0] += 1.0
        W[c_rows[3]][col_offset + 0] += 1.0
        
        # M2 = A10 * (B00 + B01)
        U[a_rows[2]][col_offset + 1] += 1.0
        V[b_rows[0]][col_offset + 1] += 1.0
        V[b_rows[1]][col_offset + 1] += 1.0
        W[c_rows[2]][col_offset + 1] += 1.0
        W[c_rows[3]][col_offset + 1] += 1.0
        
        # M3 = A00 * (B10 - B11)
        U[a_rows[0]][col_offset + 2] += 1.0
        V[b_rows[2]][col_offset + 2] += 1.0
        V[b_rows[3]][col_offset + 2] -= 1.0
        W[c_rows[0]][col_offset + 2] += 1.0
        W[c_rows[1]][col_offset + 2] += 1.0
        
        # M4 = (A10 - A00) * B00
        U[a_rows[2]][col_offset + 3] += 1.0
        U[a_rows[0]][col_offset + 3] -= 1.0
        V[b_rows[0]][col_offset + 3] += 1.0
        W[c_rows[2]][col_offset + 3] += 1.0
        W[c_rows[0]][col_offset + 3] += 1.0
        
        # M5 = (A01 + A11) * B10
        U[a_rows[1]][col_offset + 4] += 1.0
        U[a_rows[3]][col_offset + 4] += 1.0
        V[b_rows[2]][col_offset + 4] += 1.0
        W[c_rows[1]][col_offset + 4] += 1.0
        W[c_rows[3]][col_offset + 4] += 1.0
        
        # M6 = (A00 - A10) * (B01 + B11)
        U[a_rows[0]][col_offset + 5] += 1.0
        U[a_rows[2]][col_offset + 5] -= 1.0
        V[b_rows[1]][col_offset + 5] += 1.0
        V[b_rows[3]][col_offset + 5] += 1.0
        W[c_rows[0]][col_offset + 5] += 1.0
        W[c_rows[3]][col_offset + 5] += 1.0
        
        # M7 = (A11 - A01) * (B00 - B10)
        U[a_rows[3]][col_offset + 6] += 1.0
        U[a_rows[1]][col_offset + 6] -= 1.0
        V[b_rows[0]][col_offset + 6] += 1.0
        V[b_rows[2]][col_offset + 6] -= 1.0
        W[c_rows[1]][col_offset + 6] += 1.0
        W[c_rows[2]][col_offset + 6] += 1.0

    # Mapping of matrix elements to indices in U, V, W
    # U corresponds to A (2x4). Index: i*m + j
    # V corresponds to B (4x4). Index: j*p + k
    # W corresponds to C (2x4). Index: k*n + i ? 
    # Task says: W[k*n+i, r]. 
    # So W index is k*n + i.
    
    # Block 1: A[:, 0:2] * B[0:2, 0:2] -> C[:, 0:2]
    # A indices: 0,1,4,5 (A00, A01, A10, A11)
    # B indices: 0,1,8,9 (B00, B01, B10, B11)
    # C indices: 0,2,1,3 (C00, C01, C10, C11) 
    # C00: k=0,i=0 -> 0
    # C01: k=1,i=0 -> 2
    # C10: k=0,i=1 -> 1
    # C11: k=1,i=1 -> 3
    a1 = [0, 1, 4, 5]
    b1 = [0, 1, 8, 9]
    c1 = [0, 2, 1, 3]
    add_strassen_block(0, a1, b1, c1)

    # Block 2: A[:, 2:4] * B[0:2, 2:4] -> C[:, 2:4]
    # A indices: 2,3,6,7 (A02, A03, A12, A13)
    # B indices: 2,3,10,11 (B02, B03, B12, B13)
    # C indices: 4,6,5,7 (C02, C03, C12, C13)
    # C02: k=2,i=0 -> 4
    # C03: k=3,i=0 -> 6
    # C12: k=2,i=1 -> 5
    # C13: k=3,i=1 -> 7
    a2 = [2, 3, 6, 7]
    b2 = [2, 3, 10, 11]
    c2 = [4, 6, 5, 7]
    add_strassen_block(7, a2, b2, c2)

    # Block 3: A[:, 0:2] * B[2:4, 0:2] -> C[:, 0:2]
    # A indices: 0,1,4,5
    # B indices: 4,5,12,13 (B20, B21, B30, B31)
    # C indices: 0,2,1,3 (same as Block 1)
    a3 = [0, 1, 4, 5]
    b3 = [4, 5, 12, 13]
    c3 = [0, 2, 1, 3]
    add_strassen_block(14, a3, b3, c3)

    # Block 4: A[:, 2:4] * B[2:4, 2:4] -> C[:, 2:4]
    # A indices: 2,3,6,7
    # B indices: 6,7,14,15 (B22, B23, B32, B33)
    # C indices: 4,6,5,7 (same as Block 2)
    a4 = [2, 3, 6, 7]
    b4 = [6, 7, 14, 15]
    c4 = [4, 6, 5, 7]
    add_strassen_block(21, a4, b4, c4)

    # Convert to integers if possible for cleanliness, though floats are fine
    # The task says "integer/half-integer entries".
    for i in range(len(U)):
        for j in range(rank):
            U[i][j] = int(round(U[i][j] * 2)) / 2.0
    for i in range(len(V)):
        for j in range(rank):
            V[i][j] = int(round(V[i][j] * 2)) / 2.0
    for i in range(len(W)):
        for j in range(rank):
            W[i][j] = int(round(W[i][j] * 2)) / 2.0

    return {"U": U, "V": V, "W": W}
