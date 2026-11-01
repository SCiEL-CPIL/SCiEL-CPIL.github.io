import numpy as np

DIMENSIONS = [2, 2, 12]

def construct():
    n, m, p = DIMENSIONS
    # We use 6 blocks of size 2 for p=12
    num_blocks = p // 2
    rank_per_block = 7
    total_rank = num_blocks * rank_per_block
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 multiplication
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # A indices: 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
    # B indices: 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
    # C indices: 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
    
    # U rows: 4 (A entries)
    # V rows: 24 (B entries, 12 cols * 2 rows)
    # W rows: 24 (C entries, 12 cols * 2 rows, but W is p*n so 12*2=24)
    # Note: W index is k*n + i. k is col (0..11), i is row (0..1).
    
    # Strassen U contributions (per block)
    # M1: A11(0), A22(3) -> +1, +1
    # M2: A21(2), A22(3) -> +1, +1
    # M3: A11(0) -> +1
    # M4: A22(3) -> +1
    # M5: A11(0), A12(1) -> +1, +1
    # M6: A21(2), A11(0) -> +1, -1
    # M7: A12(1), A22(3) -> +1, -1
    
    U_block = [
        [1, 0, 1, 0, 1, 1, 0], # A11 (idx 0)
        [0, 0, 0, 0, 1, 0, 1], # A12 (idx 1)
        [0, 1, 0, 0, 0, 1, 0], # A21 (idx 2)
        [1, 1, 0, 1, 0, 0, -1] # A22 (idx 3)
    ]
    
    # Strassen V contributions (per block)
    # B indices in block: 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
    # M1: B11(0), B22(3) -> +1, +1
    # M2: B11(0) -> +1
    # M3: B12(1), B22(3) -> +1, -1
    # M4: B21(2), B11(0) -> +1, -1
    # M5: B22(3) -> +1
    # M6: B11(0), B12(1) -> +1, +1
    # M7: B21(2), B22(3) -> +1, +1
    
    V_block = [
        [1, 1, 0, -1, 0, 1, 0], # B11 (idx 0)
        [0, 0, 1, 0, 0, 1, 0], # B12 (idx 1)
        [0, 0, 0, 1, 0, 0, 1], # B21 (idx 2)
        [1, 0, -1, 0, 1, 0, 1] # B22 (idx 3)
    ]
    
    # Strassen W contributions (per block)
    # C indices in block: 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    W_block = [
        [1, 0, 0, 1, -1, 0, 1], # C11 (idx 0)
        [0, 0, 1, 0, 1, 0, 0], # C12 (idx 1)
        [0, 1, 0, 1, 0, 0, 0], # C21 (idx 2)
        [1, -1, 1, 0, 0, 1, 0] # C22 (idx 3)
    ]
    
    for block in range(num_blocks):
        col_offset = block * rank_per_block
        # Map block indices to global indices
        # B global index: j * p + k
        # j is 0 or 1. k is 2*block + 0 or 2*block + 1
        # V_block row 0 corresponds to B[0, 2*block] -> global row 0 * 12 + 2*block
        # V_block row 1 corresponds to B[0, 2*block+1] -> global row 0 * 12 + 2*block + 1
        # V_block row 2 corresponds to B[1, 2*block] -> global row 1 * 12 + 2*block
        # V_block row 3 corresponds to B[1, 2*block+1] -> global row 1 * 12 + 2*block + 1
        
        # W global index: k * n + i
        # k is 2*block + 0 or 2*block + 1. i is 0 or 1.
        # W_block row 0 corresponds to C[0, 2*block] -> global row (2*block) * 2 + 0
        # W_block row 1 corresponds to C[0, 2*block+1] -> global row (2*block+1) * 2 + 0
        # W_block row 2 corresponds to C[1, 2*block] -> global row (2*block) * 2 + 1
        # W_block row 3 corresponds to C[1, 2*block+1] -> global row (2*block+1) * 2 + 1
        
        for r in range(rank_per_block):
            col = col_offset + r
            # U
            for i in range(4):
                U[i][col] = U_block[i][r]
            # V
            V[0 * p + (2 * block)][col] = V_block[0][r]
            V[0 * p + (2 * block + 1)][col] = V_block[1][r]
            V[1 * p + (2 * block)][col] = V_block[2][r]
            V[1 * p + (2 * block + 1)][col] = V_block[3][r]
            # W
            W[(2 * block) * n + 0][col] = W_block[0][r]
            W[(2 * block + 1) * n + 0][col] = W_block[1][r]
            W[(2 * block) * n + 1][col] = W_block[2][r]
            W[(2 * block + 1) * n + 1][col] = W_block[3][r]
            
    return {"U": U, "V": V, "W": W}
