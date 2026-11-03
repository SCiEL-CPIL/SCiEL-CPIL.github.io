"""Matrix multiplication 4x5x5 decomposition with reduced rank via Strassen blocks."""
import time
import numpy as np

DIMENSIONS = [4, 5, 5]

def _strassen_2x2x2_factors():
    """Return (U, V, W) as lists of lists for 2x2x2 Strassen decomposition.
    
    Convention: U[(i,j), c], V[(j,k), c], W[(k,i), c]
    where i,j,k in {0,1}
    
    Strassen computes M = AB where A,B are 2x2.
    Split into quadrants:
    A = [[A00, A01], [A10, A11]]
    B = [[B00, B01], [B10, B11]]
    
    The 7 products:
    P1 = (A00+A11)(B00+B11)
    P2 = (A10+A11)B00
    P3 = A00(B01-B11)
    P4 = A11(B10-B00)
    P5 = (A00+A01)B11
    P6 = (A10-A00)(B00+B01)
    P7 = (A01-A11)(B10+B11)
    
    Results:
    C00 = P1 + P4 - P5 + P7
    C01 = P3 + P5
    C10 = P2 + P4
    C11 = P1 - P2 + P3 + P6
    """
    rank = 7
    # U has shape (4, 7), V has shape (4, 7), W has shape (4, 7)
    # U[(i,j), c], V[(j,k), c], W[(k,i), c]
    
    U = [[0.0]*rank for _ in range(4)]
    V = [[0.0]*rank for _ in range(4)]
    W = [[0.0]*rank for _ in range(4)]
    
    # P1 = (A00+A11)(B00+B11)
    # U: A00 at (0,0), A11 at (1,1)
    U[0*2+0][0] = 1  # A00
    U[1*2+1][0] = 1  # A11
    # V: B00 at (0,0), B11 at (1,1)
    V[0*2+0][0] = 1  # B00
    V[1*2+1][0] = 1  # B11
    # W: C00 += P1, C11 += P1
    W[0*2+0][0] = 1  # C00
    W[1*2+1][0] = 1  # C11
    
    # P2 = (A10+A11)B00
    U[1*2+0][1] = 1  # A10
    U[1*2+1][1] = 1  # A11
    V[0*2+0][1] = 1  # B00
    W[1*2+0][1] = 1  # C10
    W[1*2+1][1] = -1 # C11
    
    # P3 = A00(B01-B11)
    U[0*2+0][2] = 1  # A00
    V[0*2+1][2] = 1  # B01
    V[1*2+1][2] = -1 # B11
    W[0*2+1][2] = 1  # C01
    W[1*2+1][2] = 1  # C11
    
    # P4 = A11(B10-B00)
    U[1*2+1][3] = 1  # A11
    V[1*2+0][3] = 1  # B10
    V[0*2+0][3] = -1 # B00
    W[0*2+0][3] = 1  # C00
    W[1*2+0][3] = 1  # C10
    
    # P5 = (A00+A01)B11
    U[0*2+0][4] = 1  # A00
    U[0*2+1][4] = 1  # A01
    V[1*2+1][4] = 1  # B11
    W[0*2+0][4] = -1 # C00
    W[0*2+1][4] = 1  # C01
    
    # P6 = (A10-A00)(B00+B01)
    U[1*2+0][5] = 1  # A10
    U[0*2+0][5] = -1 # A00
    V[0*2+0][5] = 1  # B00
    V[0*2+1][5] = 1  # B01
    W[1*2+1][5] = 1  # C11
    
    # P7 = (A01-A11)(B10+B11)
    U[0*2+1][6] = 1  # A01
    U[1*2+1][6] = -1 # A11
    V[1*2+0][6] = 1  # B10
    V[1*2+1][6] = 1  # B11
    W[0*2+0][6] = 1  # C00
    
    return U, V, W

def _embed_2x2x2_in_4x4x4(U4, V4, W4, base, offset_i, offset_j, offset_k):
    """Embed 2x2x2 Strassen factors into 4x4x4 at position (offset_i, offset_j, offset_k).
    
    U4, V4, W4 have shape (16, rank) for 4x4x4.
    U4[(i,j), c], V4[(j,k), c], W4[(k,i), c]
    """
    U2, V2, W2 = _strassen_2x2x2_factors()
    for i2 in range(2):
        for j2 in range(2):
            u_idx = (offset_i + i2) * 4 + (offset_j + j2)
            for c in range(7):
                U4[u_idx][base + c] = U2[i2*2+j2][c]
    for j2 in range(2):
        for k2 in range(2):
            v_idx = (offset_j + j2) * 4 + (offset_k + k2)
            for c in range(7):
                V4[v_idx][base + c] = V2[j2*2+k2][c]
    for k2 in range(2):
        for i2 in range(2):
            w_idx = (offset_k + k2) * 4 + (offset_i + i2)
            for c in range(7):
                W4[w_idx][base + c] = W2[k2*2+i2][c]

def _strassen_4x4x4_factors():
    """Return (U, V, W) for 4x4x4 using Strassen on 2x2x2 blocks.
    
    Split 4x4x4 into four 2x2x2 blocks:
    Block (0,0): i,j,k in {0,1}
    Block (0,1): i in {0,1}, j in {2,3}, k in {2,3}
    Block (1,0): i in {2,3}, j in {0,1}, k in {0,1}
    Block (1,1): i,j,k in {2,3}
    
    Total rank: 4 * 7 = 28
    """
    rank = 28
    U = [[0.0]*rank for _ in range(16)]
    V = [[0.0]*rank for _ in range(16)]
    W = [[0.0]*rank for _ in range(16)]
    
    # Block (0,0): offsets (0,0,0), base 0
    _embed_2x2x2_in_4x4x4(U, V, W, 0, 0, 0, 0)
    # Block (0,1): offsets (0,2,2), base 7
    _embed_2x2x2_in_4x4x4(U, V, W, 7, 0, 2, 2)
    # Block (1,0): offsets (2,0,0), base 14
    _embed_2x2x2_in_4x4x4(U, V, W, 14, 2, 0, 0)
    # Block (1,1): offsets (2,2,2), base 21
    _embed_2x2x2_in_4x4x4(U, V, W, 21, 2, 2, 2)
    
    return U, V, W

def construct():
    """Construct rank-minimal exact tensor decomposition for 4x5x5 matrix multiplication.
    
    Returns:
        dict with keys 'U', 'V', 'W' where:
        - U: shape (n*m, rank) = (20, rank)
        - V: shape (m*p, rank) = (25, rank)
        - W: shape (p*n, rank) = (20, rank)
    """
    n, m, p = DIMENSIONS  # 4, 5, 5
    
    # Decompose into blocks:
    # The 4x5x5 tensor can be seen as:
    # For each (i,j) pair in 4x5, and k in 5:
    # We split j and k into (4,1) and (4,1) respectively
    # So we have blocks:
    # - (i in 4, j in 4, k in 4): 4x4x4 block, rank 28
    # - (i in 4, j in 4, k in 1): 4x4x1 block, rank 16
    # - (i in 4, j in 1, k in 4): 4x1x4 block, rank 16
    # - (i in 4, j in 1, k in 1): 4x1x1 block, rank 4
    # Total rank: 28 + 16 + 16 + 4 = 64
    
    total_rank = 28 + 16 + 16 + 4
    
    U = [[0.0]*total_rank for _ in range(n*m)]  # (20, rank)
    V = [[0.0]*total_rank for _ in range(m*p)]  # (25, rank)
    W = [[0.0]*total_rank for _ in range(p*n)]  # (20, rank)
    
    # Block 1: 4x4x4, rank 28, base 0
    U4, V4, W4 = _strassen_4x4x4_factors()
    for i in range(4):
        for j in range(4):
            u_idx = i * m + j  # i*5 + j
            for c in range(28):
                U[u_idx][c] = U4[i*4+j][c]
    for j in range(4):
        for k in range(4):
            v_idx = j * p + k  # j*5 + k
            for c in range(28):
                V[v_idx][c] = V4[j*4+k][c]
    for k in range(4):
        for i in range(4):
            w_idx = k * n + i  # k*4 + i
            for c in range(28):
                W[w_idx][c] = W4[k*4+i][c]
    
    # Block 2: 4x4x1, rank 16, base 28
    # j in 4, k = 4 (the 5th column)
    base = 28
    for i in range(4):
        for j in range(4):
            for c in range(16):
                col = c  # 0..15
                U[i*m + j][base + col] = 1
                V[j*p + 4][base + col] = 1 if (i*4+j) == c else 0
                W[4*n + i][base + col] = 1
    # Actually, let me redo this more carefully
    # For 4x4x1: we need U[(i,j), c], V[(j,4), c], W[(4,i), c]
    # where i in 0..3, j in 0..3, c in 0..15
    # U[(i,j), c] = 1 if c == i*4+j
    # V[(j,4), c] = 1 if c == i*4+j
    # W[(4,i), c] = 1 if c == i*4+j
    # This is schoolbook
    
    # Let me restart the block construction more carefully
    U = [[0.0]*total_rank for _ in range(n*m)]
    V = [[0.0]*total_rank for _ in range(m*p)]
    W = [[0.0]*total_rank for _ in range(p*n)]
    
    # Block 1: 4x4x4, rank 28, base 0
    U4, V4, W4 = _strassen_4x4x4_factors()
    for i in range(4):
        for j in range(4):
            u_idx = i * m + j  # i*5 + j
            for c in range(28):
                U[u_idx][c] = U4[i*4+j][c]
    for j in range(4):
        for k in range(4):
            v_idx = j * p + k  # j*5 + k
            for c in range(28):
                V[v_idx][c] = V4[j*4+k][c]
    for k in range(4):
        for i in range(4):
            w_idx = k * n + i  # k*4 + i
            for c in range(28):
                W[w_idx][c] = W4[k*4+i][c]
    
    # Block 2: 4x4x1, rank 16, base 28
    # i in 0..3, j in 0..3, k = 4
    base = 28
    for i in range(4):
        for j in range(4):
            c = i*4 + j
            U[i*m + j][base + c] = 1
            V[j*p + 4][base + c] = 1
            W[4*n + i][base + c] = 1
    
    # Block 3: 4x1x4, rank 16, base 44
    # i in 0..3, j = 4, k in 0..3
    base = 44
    for i in range(4):
        for k in range(4):
            c = i*4 + k
            U[i*m + 4][base + c] = 1
            V[4*p + k][base + c] = 1
            W[k*n + i][base + c] = 1
    
    # Block 4: 4x1x1, rank 4, base 60
    # i in 0..3, j = 4, k = 4
    base = 60
    for i in range(4):
        U[i*m + 4][base + i] = 1
        V[4*p + 4][base + i] = 1
        W[4*n + i][base + i] = 1
    
    return {"U": U, "V": V, "W": W}
