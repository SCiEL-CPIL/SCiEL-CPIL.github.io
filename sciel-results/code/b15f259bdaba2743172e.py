import time
import numpy as np

DIMENSIONS = [2, 2, 16]

def construct():
    """
    Rank-56 decomposition for multiplying a 2x2 matrix A by a 2x16 matrix B.
    This is achieved by viewing B as 8 independent 2x2 blocks and applying
    Strassen's algorithm (rank 7) to each block.
    Total rank = 8 * 7 = 56.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    n, m, p = DIMENSIONS  # n=2, m=2, p=16
    
    # Strassen coefficients for 2x2x2 matrix multiplication
    # U corresponds to linear combinations of A entries
    # V corresponds to linear combinations of B entries
    # W corresponds to linear combinations of C entries
    
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    
    # Strassen Formulas:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * B10
    # M7 = (A01 - A11) * (B10 + B11)
    
    # C00 = M1 - M4 + M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # U: 7 rows (for 7 multiplications), 4 cols (A00, A01, A10, A11)
    strassen_U = np.array([
        [1, 0, 0, 1],  # M1: A00 + A11
        [0, 0, 1, 1],  # M2: A10 + A11
        [1, 0, 0, 0],  # M3: A00
        [0, 0, 0, 1],  # M4: A11
        [1, 1, 0, 0],  # M5: A00 + A01
        [-1, 0, 1, 0], # M6: A10 - A00
        [0, 1, 0, -1]  # M7: A01 - A11
    ], dtype=float)
    
    # V: 7 rows, 4 cols (B00, B01, B10, B11)
    strassen_V = np.array([
        [1, 0, 0, 1],  # M1: B00 + B11
        [1, 0, 0, 0],  # M2: B00
        [0, 1, 0, -1], # M3: B01 - B11
        [-1, 0, 1, 0], # M4: B10 - B00
        [0, 0, 0, 1],  # M5: B11
        [1, 1, 0, 0],  # M6: B00 + B01
        [0, 0, 1, 1]   # M7: B10 + B11
    ], dtype=float)
    
    # W: 7 rows, 4 cols (C00, C01, C10, C11)
    # Note: The trace convention is W[k*n+i, r].
    # For 2x2, indices are:
    # C00 -> k=0, i=0 -> idx 0
    # C01 -> k=1, i=0 -> idx 1
    # C10 -> k=0, i=1 -> idx 2
    # C11 -> k=1, i=1 -> idx 3
    # Wait, the prompt says W[k*n+i, r].
    # n=2.
    # C[i,k] in standard math.
    # Prompt: "output factor follows the official trace convention W[k*n+i,r]"
    # Usually C[i,k] is stored.
    # Let's check the previous code's mapping.
    # Previous code: c_rows = [4*l, 4*l+2, 4*l+1, 4*l+3]
    # This mapped C00->0, C01->2, C10->1, C11->3?
    # Let's re-verify the index mapping for W.
    # W index = k * n + i.
    # If C is 2x16.
    # For block l (columns 2l, 2l+1).
    # C[0, 2l] -> k=2l, i=0 -> idx 2l*2 + 0 = 4l.
    # C[1, 2l] -> k=2l, i=1 -> idx 2l*2 + 1 = 4l+1.
    # C[0, 2l+1] -> k=2l+1, i=0 -> idx (2l+1)*2 + 0 = 4l+2.
    # C[1, 2l+1] -> k=2l+1, i=1 -> idx (2l+1)*2 + 1 = 4l+3.
    
    # So the local indices for C00, C10, C01, C11 in the 2x2 block are:
    # C00 (i=0, k=0) -> 0
    # C10 (i=1, k=0) -> 1
    # C01 (i=0, k=1) -> 2
    # C11 (i=1, k=1) -> 3
    
    # My strassen_W above was defined for C00, C01, C10, C11 order?
    # Let's check the formulas again.
    # C00 = M1 - M4 + M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We need to map these to the W indices:
    # W[0] corresponds to C00
    # W[1] corresponds to C10
    # W[2] corresponds to C01
    # W[3] corresponds to C11
    
    # So row 0 of strassen_W (for M1) should have:
    # C00: 1, C10: 0, C01: 0, C11: 1
    # Row 1 (M2):
    # C00: 0, C10: 1, C01: 0, C11: -1
    # Row 2 (M3):
    # C00: 0, C10: 0, C01: 1, C11: 1
    # Row 3 (M4):
    # C00: -1, C10: 1, C01: 0, C11: 0
    # Row 4 (M5):
    # C00: 1, C10: 0, C01: 1, C11: 0
    # Row 5 (M6):
    # C00: 0, C10: 0, C01: 0, C11: 1
    # Row 6 (M7):
    # C00: 1, C10: 0, C01: 0, C11: 0
    
    strassen_W = np.array([
        [1, 0, 0, 1],  # M1: C00+1, C11+1
        [0, 1, 0, -1], # M2: C10+1, C11-1
        [0, 0, 1, 1],  # M3: C01+1, C11+1
        [-1, 1, 0, 0], # M4: C00-1, C10+1
        [1, 0, 1, 0],  # M5: C00+1, C01+1
        [0, 0, 0, 1],  # M6: C11+1
        [1, 0, 0, 0]   # M7: C00+1
    ], dtype=float)
    
    rank = 56
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Fill U, V, W
    for l in range(8):
        for s in range(7):
            col = 7 * l + s
            
            # U: same for all blocks
            U[:, col] = strassen_U[s]
            
            # V: local B indices mapped to global
            # B00 -> 2*l, B01 -> 2*l+1, B10 -> 16+2*l, B11 -> 16+2*l+1
            b_rows = [2*l, 2*l+1, 16+2*l, 16+2*l+1]
            for b_idx in range(4):
                V[b_rows[b_idx], col] = strassen_V[s, b_idx]
            
            # W: local C indices mapped to global
            # C00 -> 4*l, C10 -> 4*l+1, C01 -> 4*l+2, C11 -> 4*l+3
            c_rows = [4*l, 4*l+1, 4*l+2, 4*l+3]
            for c_idx in range(4):
                W[c_rows[c_idx], col] = strassen_W[s, c_idx]
    
    # Verification
    def verify(U, V, W, n, m, p):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i * m + j
                    v = j * p + k
                    w = k * n + i
                    val = 0
                    for s in range(rank):
                        val += U[u, s] * V[v, s] * W[w, s]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    if verify(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook
    rank_school = n * m * p
    U = np.zeros((n * m, rank_school), dtype=float)
    V = np.zeros((m * p, rank_school), dtype=float)
    W = np.zeros((p * n, rank_school), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
