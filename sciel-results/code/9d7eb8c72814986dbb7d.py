import numpy as np
import time

def _matmul22_7():
    """Return U,V (4x7) and W (4x7) for 2x2 * 2x2 matmul, rank 7 (Strassen/Winograd)."""
    # U rows: x00, x01, x10, x11
    # V rows: y00, y01, y10, y11
    # W rows: z00, z01, z10, z11 (k*n+i = k*2+i)
    
    U = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 1, 0, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 1, 1],
        [1, 1, 0, 1, 0, 0, 0]
    ], dtype=float)
    
    V = np.array([
        [1, 1, -1, 0, 0, 1, 0],
        [1, 0, 0, 1, 0, 1, 0],
        [1, 0, 1, 0, 0, 0, 1],
        [1, 0, 0, -1, 1, 0, 1]
    ], dtype=float)
    
    W = np.array([
        [1, 0, -1, 0, -1, -1, 0], # z00
        [0, 0, 0, 1, 1, -1, 0],   # z01
        [0, 1, 1, 0, -1, 0, 1],   # z10
        [1, -1, 0, 1, 0, 0, 1]    # z11
    ], dtype=float)
    
    return U, V, W

def construct():
    n, m, p = 2, 4, 6
    t0 = time.time()
    deadline = t0 + 140.0

    U22, V22, W22 = _matmul22_7()
    
    # Build 2x4 * 4x6
    # U: 8 rows
    # V: 24 rows
    # W: 12 rows
    
    # We have 2x2 * 2x6 for left and right.
    # 2x2 * 2x6 is 3 blocks of 2x2 * 2x2.
    # Total 6 blocks of 2x2 * 2x2.
    # Rank = 6 * 7 = 42.
    
    U = np.zeros((8, 42))
    V = np.zeros((24, 42))
    W = np.zeros((12, 42))
    
    col_offset = 0
    # Left part: A_left (rows 0,1,4,5 of U) * B_top (rows 0..5 of V)
    # B_top is 2x6. Split into 3 2x2 blocks.
    # Block 0: B_top cols 0,1 -> V rows 0,1 (y00,y01) and 6,7 (y10,y11)? 
    # V indexing for 2x2: j*2+k. 
    # For 2x6, j is 0,1. k is 0..5.
    # Block 0: k=0,1. V rows 0,1 (j=0) and 6,7 (j=1)? No.
    # V rows are j*p+k. Here p=2 for the sub-block?
    # In the full 4x6, V rows are j*6+k.
    # For B_top (rows 0,1 of B), j=0,1. k=0..5.
    # Block 0 (cols 0,1): V rows 0,1 (j=0,k=0,1) and 6,7 (j=1,k=0,1).
    # Wait, V row index is j*p+k. p=6.
    # Row 0: j=0, k=0.
    # Row 1: j=0, k=1.
    # Row 6: j=1, k=0.
    # Row 7: j=1, k=1.
    
    for block in range(3):
        # U rows for A_left: 0,1 (x00,x01) and 4,5 (x10,x11)
        u_rows = [0, 1, 4, 5]
        # V rows for B_top block:
        # j=0, k=2*block, 2*block+1 -> rows 2*block, 2*block+1
        # j=1, k=2*block, 2*block+1 -> rows 6+2*block, 6+2*block+1
        v_rows = [2*block, 2*block+1, 6+2*block, 6+2*block+1]
        # W rows for Z_top (rows 0,1 of Z):
        # k=2*block, 2*block+1. i=0,1.
        # W index k*n+i. n=2.
        # k=2*block, i=0 -> 4*block
        # k=2*block, i=1 -> 4*block+1
        # k=2*block+1, i=0 -> 4*block+2
        # k=2*block+1, i=1 -> 4*block+3
        w_rows = [4*block, 4*block+1, 4*block+2, 4*block+3]
        
        for c in range(7):
            for r_u, r_v, r_w in zip(u_rows, v_rows, w_rows):
                U[r_u, col_offset+c] = U22[r_u, c]
                V[r_v, col_offset+c] = V22[r_v, c]
                W[r_w, col_offset+c] = W22[r_w, c]
        col_offset += 7
        
    # Right part: A_right (rows 2,3,6,7 of U) * B_bottom (rows 12..23 of V)
    # B_bottom is rows 2,3 of B. j=2,3.
    # V rows: j*6+k.
    # j=2: 12, 13, 14, 15, 16, 17
    # j=3: 18, 19, 20, 21, 22, 23
    # Block 0: k=0,1. V rows 12,13 (j=2) and 18,19 (j=3).
    
    for block in range(3):
        u_rows = [2, 3, 6, 7]
        v_rows = [12+2*block, 12+2*block+1, 18+2*block, 18+2*block+1]
        # W rows for Z_bottom (rows 2,3 of Z? No, Z is 2x6. Rows 0,1).
        # Wait, Z is 2x6. i=0,1.
        # The output of A_right * B_bottom is added to Z.
        # So W rows are the same: 4*block .. 4*block+3.
        w_rows = [4*block, 4*block+1, 4*block+2, 4*block+3]
        
        for c in range(7):
            for r_u, r_v, r_w in zip(u_rows, v_rows, w_rows):
                U[r_u, col_offset+c] += U22[r_u, c]
                V[r_v, col_offset+c] += V22[r_v, c]
                W[r_w, col_offset+c] += W22[r_w, c]
        col_offset += 7

    # Verification
    rng = np.random.RandomState(0)
    X = rng.randn(2, 4)
    Y = rng.randn(4, 6)
    Z_true = X @ Y

    x = X.reshape(8)
    y = Y.reshape(24)
    Z_decomp = np.zeros(12)
    for c in range(42):
        uc = U[:, c]
        vc = V[:, c]
        wc = W[:, c]
        a = np.einsum('r,r->', uc, x)
        b = np.einsum('r,r->', vc, y)
        Z_decomp += wc * (a * b)
    Z_decomp = Z_decomp.reshape(6, 2).T

    if not np.allclose(Z_decomp, Z_true, atol=1e-9, rtol=1e-9):
        # Fallback
        U48 = np.zeros((8, 48))
        V48 = np.zeros((24, 48))
        W48 = np.zeros((12, 48))
        col = 0
        for i in range(2):
            for j in range(4):
                for k in range(6):
                    U48[i*4+j, col] = 1.0
                    V48[j*6+k, col] = 1.0
                    W48[k*2+i, col] = 1.0
                    col += 1
        return {"U": U48.tolist(), "V": V48.tolist(), "W": W48.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
