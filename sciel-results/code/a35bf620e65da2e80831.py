"""Matrix multiplication 2x3 x 3x10 - rank-54 decomposition"""
import numpy as np
import time

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-54 exact tensor decomposition for 2x3 x 3x10 matrix multiplication.
    
    Strategy: Split m=3 into m1=1, m2=2.
    - Part 1 (rank 10): A_top(1x1) * B_left(1x10) via schoolbook.
    - Part 2 (rank 44): A_bottom(2x2) * B_right(2x10) via optimized Strassen blocks.
    
    For the 2x2 x 2x10 part, we use 5 Strassen blocks of rank 7 each = 35,
    plus we can potentially save ranks by sharing products between adjacent blocks.
    
    Actually, let me try rank 54 by using a more efficient decomposition.
    
    Alternative: Use rank 7 Strassen for 2x2x2, and for the 2x2x10 tensor,
    the rank is at most 5*7 = 35 (independent blocks).
    
    So total = 10 + 35 = 45. Wait, that's already better!
    
    Let me verify: 2x1 x 1x10 needs rank 10 (schoolbook).
    2x2 x 2x10: split into 5 independent 2x2x2 blocks, each rank 7, total 35.
    Grand total: 10 + 35 = 45.
    """
    n, m, p = DIMENSIONS  # 2, 3, 10
    rank = 45
    
    U = np.zeros((n * m, rank))  # (6, 45)
    V = np.zeros((m * p, rank))  # (30, 45)
    W = np.zeros((p * n, rank))  # (20, 45)
    
    col = 0
    
    # Part 1: A_top (A[0,:], shape 1x3) * B_left (B[:,0], shape 3x1)
    # Wait, I need to reconsider the split.
    
    # Let me split m=3 into m1=1 (row 0) and m2=2 (rows 1,2).
    # A is 2x3, B is 3x10.
    # C = A @ B (2x10)
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # Split j: j=0 for top, j=1,2 for bottom.
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    #
    # Part 1: A_col0 (A[:,0], 2x1) * B_row0 (B[0,:], 1x10) -> rank 10
    # Part 2: A_bottom (A[:,1:3], 2x2) * B_bottom (B[1:3,:], 2x10) -> rank 35 (5 Strassen)
    
    # Part 1: A[:,0] * B[0,:]
    # A[i,0] indices: i*3+0 = 0, 3
    # B[0,k] indices: 0*10+k = k for k=0..9
    # C[i,k] = W[k*2+i]
    for i in range(n):
        a_idx = i * m + 0  # 0 or 3
        for k in range(p):
            b_idx = 0 * p + k  # 0..9
            c_idx = k * n + i  # 2k+i
            U[a_idx, col] = 1.0
            V[b_idx, col] = 1.0
            W[c_idx, col] = 1.0
            col += 1
    
    # Part 2: A[:,1:3] (2x2) * B[1:3,:] (2x10)
    # A[i,1] indices: i*3+1 = 1, 4
    # A[i,2] indices: i*3+2 = 2, 5
    # B[1,k] indices: 1*10+k = 10+k
    # B[2,k] indices: 2*10+k = 20+k
    
    x00 = 1  # A[0,1]
    x01 = 2  # A[0,2]
    x10 = 4  # A[1,1]
    x11 = 5  # A[1,2]
    
    for j in range(5):
        # B block j covers columns 2j, 2j+1
        y00 = 10 + 2 * j       # B[1, 2j]
        y01 = 10 + 2 * j + 1   # B[1, 2j+1]
        y10 = 20 + 2 * j       # B[2, 2j]
        y11 = 20 + 2 * j + 1   # B[2, 2j+1]
        
        # C output for this block
        z00 = (2 * j) * n + 0      # C[0, 2j]
        z10 = (2 * j) * n + 1      # C[1, 2j]
        z01 = (2 * j + 1) * n + 0  # C[0, 2j+1]
        z11 = (2 * j + 1) * n + 1  # C[1, 2j+1]
        
        # Strassen's 7 products for 2x2 * 2x2
        # M1 = (X00 + X11) * (Y00 + Y11)
        U[x00, col] += 1.0
        U[x11, col] += 1.0
        V[y00, col] += 1.0
        V[y11, col] += 1.0
        W[z00, col] += 1.0
        W[z11, col] += 1.0
        col += 1
        
        # M2 = (X10 + X11) * Y00
        U[x10, col] += 1.0
        U[x11, col] += 1.0
        V[y00, col] += 1.0
        W[z10, col] += 1.0
        W[z11, col] -= 1.0
        col += 1
        
        # M3 = X00 * (Y01 - Y11)
        U[x00, col] += 1.0
        V[y01, col] += 1.0
        V[y11, col] -= 1.0
        W[z01, col] += 1.0
        W[z11, col] += 1.0
        col += 1
        
        # M4 = X11 * (Y10 - Y00)
        U[x11, col] += 1.0
        V[y10, col] += 1.0
        V[y00, col] -= 1.0
        W[z00, col] += 1.0
        W[z10, col] += 1.0
        col += 1
        
        # M5 = (X00 + X01) * Y11
        U[x00, col] += 1.0
        U[x01, col] += 1.0
        V[y11, col] += 1.0
        W[z00, col] -= 1.0
        W[z01, col] += 1.0
        col += 1
        
        # M6 = (X10 - X00) * (Y00 + Y01)
        U[x10, col] += 1.0
        U[x00, col] -= 1.0
        V[y00, col] += 1.0
        V[y01, col] += 1.0
        W[z11, col] += 1.0
        col += 1
        
        # M7 = (X01 - X11) * (Y10 + Y11)
        U[x01, col] += 1.0
        U[x11, col] -= 1.0
        V[y10, col] += 1.0
        V[y11, col] += 1.0
        W[z00, col] += 1.0
        col += 1
    
    if col != rank:
        raise ValueError(f"Rank mismatch: expected {rank}, got {col}")
    
    # Validate with random data
    np.random.seed(42)
    A = np.random.rand(n, m)
    B = np.random.rand(m, p)
    C_expected = A @ B
    
    C_calc = np.zeros((n, p))
    for r in range(rank):
        u_vec = U[:, r].reshape(n, m)
        v_vec = V[:, r].reshape(m, p)
        P_r = u_vec @ v_vec
        for i in range(n):
            for k in range(p):
                C_calc[i, k] += P_r[i, k] * W[k * n + i, r]
    
    if not np.allclose(C_calc, C_expected, atol=1e-9):
        raise ValueError("Decomposition validation failed")
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

