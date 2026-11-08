import numpy as np
import time

def construct():
    start_time = time.time()
    n, m, p = 4, 4, 8

    # Base case: 2x2x2 Strassen
    # A, B, C are 2x2.
    # Indices: A[i,j] -> i*2+j, B[j,k] -> j*2+k, C[k,i] -> k*2+i
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=int)
    
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=int)
    
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=int)
    
    # Helper to build block-structured Strassen
    # For 4x4x4, we treat A, B, C as 2x2 blocks of 2x2 matrices.
    # The "super" Strassen operates on these blocks.
    # The "sub" Strassen operates on the elements within the blocks.
    
    # Specs for the 7 super-steps (same as 2x2 Strassen logic but on blocks)
    # A_eff = sum coeff * A_block
    # B_eff = sum coeff * B_block
    # C_eff = sum coeff * C_block
    
    # Block indices: 00, 01, 10, 11
    # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
    
    A_specs = [
        [(1, 0), (1, 3)], # P0: A00+A11
        [(1, 2), (1, 3)], # P1: A10+A11
        [(1, 0)],         # P2: A00
        [(1, 3)],         # P3: A11
        [(1, 0), (1, 1)], # P4: A00+A01
        [(1, 2), (-1, 0)],# P5: A10-A00
        [(1, 1), (-1, 3)] # P6: A01-A11
    ]
    
    B_specs = [
        [(1, 0), (1, 3)], # P0: B00+B11
        [(1, 0)],         # P1: B00
        [(1, 1), (-1, 3)],# P2: B01-B11
        [(1, 2), (-1, 0)],# P3: B10-B00
        [(1, 3)],         # P4: B11
        [(1, 0), (1, 1)], # P5: B00+B01
        [(1, 2), (1, 3)]  # P6: B10+B11
    ]
    
    C_specs = [
        [(1, 0), (1, 3)], # P0: C00+C11
        [(1, 2), (-1, 3)],# P1: C10-C11
        [(1, 1), (1, 3)], # P2: C01+C11
        [(1, 0), (1, 2)], # P3: C00+C10
        [(-1, 0), (1, 1)],# P4: -C00+C01
        [(1, 3)],         # P5: C11
        [(1, 0)]          # P6: C00
    ]
    
    # Build 4x4x4 decomposition (Rank 49)
    # U4: 16x49, V4: 16x49, W4: 16x49
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    for kp in range(7):
        for ms in range(7):
            col = kp * 7 + ms
            
            # U: A is 4x4. Index i*4+j.
            # Block (bi, bj) covers rows 2*bi..2*bi+1, cols 2*bj..2*bj+1.
            # Within block, local index r*2+c.
            # Global index: (2*bi+r)*4 + (2*bj+c).
            for coeff, block_idx in A_specs[kp]:
                bi, bj = divmod(block_idx, 2)
                for r in range(2):
                    for c in range(2):
                        gi = 2 * bi + r
                        gj = 2 * bj + c
                        U4[gi * 4 + gj, col] += coeff * U2[r * 2 + c, ms]
                        
            # V: B is 4x4. Index j*4+k.
            for coeff, block_idx in B_specs[kp]:
                bi, bj = divmod(block_idx, 2)
                for r in range(2):
                    for c in range(2):
                        gj = 2 * bi + r
                        gk = 2 * bj + c
                        V4[gj * 4 + gk, col] += coeff * V2[r * 2 + c, ms]
                        
            # W: C is 4x4. Index k*4+i.
            for coeff, block_idx in C_specs[kp]:
                bi, bj = divmod(block_idx, 2)
                for r in range(2):
                    for c in range(2):
                        gi = 2 * bi + r
                        gk = 2 * bj + c
                        W4[gk * 4 + gi, col] += coeff * W2[r * 2 + c, ms]
                        
    # Extend to 4x4x8
    # B is 4x8. Split into B_left (4x4) and B_right (4x4).
    # C is 4x8. Split into C_left (4x4) and C_right (4x4).
    # The multiplication is [A * B_left, A * B_right].
    # We can use the same A decomposition for both halves.
    # Rank = 49 (for left) + 49 (for right) = 98.
    
    rank = 98
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # U is the same for both halves
    U[:, :49] = U4
    U[:, 49:] = U4
    
    # V: B is 4x8. Index j*8+k.
    # Left half: k in [0, 4). V4 index j*4+k.
    # Right half: k in [4, 8). V4 index j*4+(k-4).
    for j in range(4):
        for k_local in range(4):
            # Left half
            V[j * 8 + k_local, :49] = V4[j * 4 + k_local, :]
            # Right half
            V[j * 8 + (4 + k_local), 49:] = V4[j * 4 + k_local, :]
            
    # W: C is 4x8. Index k*4+i. (Trace convention: W[k*n+i, r])
    # Here n=4. So index is k*4+i.
    # Left half: k in [0, 4). W4 index k*4+i.
    # Right half: k in [4, 8). W4 index (k-4)*4+i.
    for k_local in range(4):
        for i in range(4):
            # Left half
            W[k_local * 4 + i, :49] = W4[k_local * 4 + i, :]
            # Right half
            W[(4 + k_local) * 4 + i, 49:] = W4[k_local * 4 + i, :]
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
