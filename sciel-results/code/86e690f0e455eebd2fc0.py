import time
import numpy as np

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """Find a rank-minimal exact tensor decomposition for 2x3 * 3x8 matrix multiplication."""
    n, m, p = 2, 3, 8
    
    def verify(U, V, W, n, m, p):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        r = U.shape[1]
        T = np.zeros((n, m, p))
        for c in range(r):
            u = U[:, c].reshape(n, m)
            v = V[:, c].reshape(m, p)
            w = W[:, c].reshape(p, n)
            T += np.einsum('ij,jk,ki->ijk', u, v, w)
        return np.allclose(T, np.ones((n, m, p)), atol=1e-10)

    # Attempt Rank 42 Decomposition
    # Structure:
    # 1. A[:,0] (2x1) * B[0,:] (1x8) -> Rank 16
    # 2. A[:,1:3] (2x2) * B[1:3,0:4] (2x4) -> Rank 13
    # 3. A[:,1:3] (2x2) * B[1:3,4:8] (2x4) -> Rank 13
    # Total Rank: 42
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: A[:, 0] * B[0, :]
    # A indices: 0, 3
    # B indices: 0..7
    # C indices: k*2 + i
    for i in range(2):
        for k in range(8):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            
    # Helper to add a 2x2x4 rank-13 decomposition
    def add_2x2x4_rank13(a_offset, b_offset, w_offset):
        nonlocal col
        # Local A indices: 0,1,2,3 -> Global: a_offset+0, a_offset+1, a_offset+2, a_offset+3
        # Local B indices: 0..7 -> Global: b_offset+0 .. b_offset+7
        # Local W indices: 0..7 -> Global: w_offset+0 .. w_offset+7
        
        # The 13 terms for M_{2,2,4}
        # a00, a01, a10, a11
        # b00, b01, b02, b03, b10, b11, b12, b13
        # c00, c01, c02, c03, c10, c11, c12, c13
        
        # Term 1: (a00+a11)(b00+b12)
        U[a_offset+0][col] = 1; U[a_offset+3][col] = 1
        V[b_offset+0][col] = 1; V[b_offset+6][col] = 1
        W[w_offset+0][col] = 1; W[w_offset+6][col] = 1
        col += 1
        
        # Term 2: (a10+a11)(b00+b10)
        U[a_offset+2][col] = 1; U[a_offset+3][col] = 1
        V[b_offset+0][col] = 1; V[b_offset+4][col] = 1
        W[w_offset+4][col] = 1; W[w_offset+5][col] = 1
        col += 1
        
        # Term 3: a00(b01-b11)
        U[a_offset+0][col] = 1
        V[b_offset+1][col] = 1; V[b_offset+5][col] = -1
        W[w_offset+1][col] = 1; W[w_offset+7][col] = 1
        col += 1
        
        # Term 4: a11(b12-b02)
        U[a_offset+3][col] = 1
        V[b_offset+6][col] = 1; V[b_offset+2][col] = -1
        W[w_offset+0][col] = 1; W[w_offset+2][col] = 1
        col += 1
        
        # Term 5: (a00+a01)(b12+b13)
        U[a_offset+0][col] = 1; U[a_offset+1][col] = 1
        V[b_offset+6][col] = 1; V[b_offset+7][col] = 1
        W[w_offset+2][col] = 1; W[w_offset+3][col] = 1
        col += 1
        
        # Term 6: (a10-a00)(b00+b01)
        U[a_offset+2][col] = 1; U[a_offset+0][col] = -1
        V[b_offset+0][col] = 1; V[b_offset+1][col] = 1
        W[w_offset+5][col] = 1
        col += 1
        
        # Term 7: (a01-a11)(b10+b11)
        U[a_offset+1][col] = 1; U[a_offset+3][col] = -1
        V[b_offset+4][col] = 1; V[b_offset+5][col] = 1
        W[w_offset+0][col] = 1
        col += 1
        
        # Term 8: a01(b02-b12)
        U[a_offset+1][col] = 1
        V[b_offset+2][col] = 1; V[b_offset+6][col] = -1
        W[w_offset+3][col] = 1; W[w_offset+7][col] = 1
        col += 1
        
        # Term 9: a10(b12-b02)
        U[a_offset+2][col] = 1
        V[b_offset+6][col] = 1; V[b_offset+2][col] = -1
        W[w_offset+6][col] = 1; W[w_offset+2][col] = 1
        col += 1
        
        # Term 10: (a00+a10)(b02+b03)
        U[a_offset+0][col] = 1; U[a_offset+2][col] = 1
        V[b_offset+2][col] = 1; V[b_offset+3][col] = 1
        W[w_offset+2][col] = 1; W[w_offset+3][col] = 1
        col += 1
        
        # Term 11: (a01+a11)(b10+b13)
        U[a_offset+1][col] = 1; U[a_offset+3][col] = 1
        V[b_offset+4][col] = 1; V[b_offset+7][col] = 1
        W[w_offset+3][col] = 1; W[w_offset+7][col] = 1
        col += 1
        
        # Term 12: a00(b10-b13)
        U[a_offset+0][col] = 1
        V[b_offset+4][col] = 1; V[b_offset+7][col] = -1
        W[w_offset+4][col] = 1
        col += 1
        
        # Term 13: a11(b03-b13)
        U[a_offset+3][col] = 1
        V[b_offset+3][col] = 1; V[b_offset+7][col] = -1
        W[w_offset+3][col] = 1; W[w_offset+7][col] = 1
        col += 1

    # Apply to Block 1: A[:, 1:3] * B[1:3, 0:4]
    # A indices for cols 1,2: 1,2 (row 0), 4,5 (row 1)
    # Local A mapping: 0->1, 1->2, 2->4, 3->5
    # B indices for rows 1,2 and cols 0-3:
    # Row 1 (idx 1): 1*8+0=8, 9, 10, 11
    # Row 2 (idx 2): 2*8+0=16, 17, 18, 19
    # Local B mapping: 0->8, 1->9, 2->10, 3->11, 4->16, 5->17, 6->18, 7->19
    # W indices for cols 0-3: 0..7
    add_2x2x4_rank13(a_offset=1, b_offset=8, w_offset=0)
    
    # Apply to Block 2: A[:, 1:3] * B[1:3, 4:8]
    # A indices same: 1,2,4,5
    # B indices for rows 1,2 and cols 4-7:
    # Row 1: 1*8+4=12, 13, 14, 15
    # Row 2: 2*8+4=20, 21, 22, 23
    # Local B mapping: 0->12, 1->13, 2->14, 3->15, 4->20, 5->21, 6->22, 7->23
    # W indices for cols 4-7: 8..15
    add_2x2x4_rank13(a_offset=1, b_offset=12, w_offset=8)

    if verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}

    # Fallback: Rank 44 Strassen Decomposition
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: A[:, 0] * B[0, :]
    for i in range(2):
        for k in range(8):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            
    # Part 2: A[:, 1:3] * B[1:3, :] using Strassen on 2x2x2 blocks
    for block in range(4):
        b00 = 1 * p + block * 2
        b01 = 1 * p + block * 2 + 1
        b10 = 2 * p + block * 2
        b11 = 2 * p + block * 2 + 1
        
        c00 = (block * 2) * n + 0
        c01 = (block * 2 + 1) * n + 0
        c10 = (block * 2) * n + 1
        c11 = (block * 2 + 1) * n + 1
        
        # Strassen steps
        U[1][col] = 1; U[5][col] = 1
        V[b00][col] = 1; V[b11][col] = 1
        W[c00][col] = 1; W[c11][col] = 1
        col += 1
        
        U[4][col] = 1; U[5][col] = 1
        V[b00][col] = 1
        W[c10][col] = 1; W[c11][col] = -1
        col += 1
        
        U[1][col] = 1
        V[b01][col] = 1; V[b11][col] = -1
        W[c01][col] = 1; W[c11][col] = 1
        col += 1
        
        U[5][col] = 1
        V[b10][col] = 1; V[b00][col] = -1
        W[c00][col] = 1; W[c10][col] = 1
        col += 1
        
        U[1][col] = 1; U[2][col] = 1
        V[b11][col] = 1
        W[c00][col] = -1; W[c01][col] = 1
        col += 1
        
        U[4][col] = 1; U[1][col] = -1
        V[b00][col] = 1; V[b01][col] = 1
        W[c11][col] = 1
        col += 1
        
        U[2][col] = 1; U[5][col] = -1
        V[b10][col] = 1; V[b11][col] = 1
        W[c00][col] = 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
