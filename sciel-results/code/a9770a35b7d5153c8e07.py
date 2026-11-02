"""Construct rank-minimal exact tensor decomposition for 3x3 x 3x8 matrix multiplication."""
DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 8
    # Strategy: Split A into 2x2, 2x1, 1x2, 1x1.
    # Split B into 2x8, 1x8.
    # C_top (2x8) = A_22 B_28 + v b
    # C_bot (1x8) = w^T B_28 + a b
    
    # Rank calculation:
    # A_22 B_28: 4 blocks of 2x2x2. Each 2x2x2 uses Strassen (rank 7). Total 28.
    # v b: v is 2x1, b is 1x8. 2*8 = 16 terms.
    # w^T B_28: w is 1x2, B_28 is 2x8. 2*8 = 16 terms.
    # a b: a is 1x1, b is 1x8. 1*8 = 8 terms.
    # Total: 28 + 16 + 16 + 8 = 68.
    
    rank = 68
    U = [[0]*rank for _ in range(n*m)]  # 9 rows
    V = [[0]*rank for _ in range(m*p)]  # 24 rows
    W = [[0]*rank for _ in range(p*n)]  # 24 rows
    
    col = 0
    
    # Helper to add Strassen 2x2x2 decomposition
    # A is 2x2, B is 2x2, C is 2x2
    # Indices in U: (i*m + j) for A
    # Indices in V: (j*p + k) for B
    # Indices in W: (k*n + i) for C
    
    # We need to map the 2x2 blocks to the global indices.
    # A_22 is A[0:2, 0:2] -> indices 0,1,3,4 in U (row i, col j)
    # B_28 is B[0:2, 0:8] -> indices 0..15 in V
    # C_top is C[0:2, 0:8] -> indices 0..15 in W
    
    # Strassen for 2x2x2:
    # m1 = (a00+a11)(b00+b11)
    # m2 = (a10+a11)b00
    # m3 = a00(b01-b11)
    # m4 = a11(b10-b00)
    # m5 = (a00+a01)b11
    # m6 = (a10-a00)(b00+b01)
    # m7 = (a01-a11)(b10+b11)
    
    # We will add these 7 terms for each 2x2 block of B_28.
    # B_28 has 4 blocks of 2x2: cols 0-1, 2-3, 4-5, 6-7.
    
    for block_idx in range(4):
        k_start = block_idx * 2
        # A_22 indices:
        # a00: U[0] (i=0, j=0)
        # a01: U[1] (i=0, j=1)
        # a10: U[3] (i=1, j=0)
        # a11: U[4] (i=1, j=1)
        
        # B_block indices:
        # b00: V[0*p + k_start] = V[k_start]
        # b01: V[0*p + k_start+1] = V[k_start+1]
        # b10: V[1*p + k_start] = V[p + k_start]
        # b11: V[1*p + k_start+1] = V[p + k_start+1]
        
        # C_block indices (C_top):
        # c00: W[k_start*n + 0] = W[k_start]
        # c01: W[(k_start+1)*n + 0] = W[3*k_start+3]
        # c10: W[k_start*n + 1] = W[k_start+1]
        # c11: W[(k_start+1)*n + 1] = W[3*k_start+4]
        
        # Term 1: (a00+a11)(b00+b11) -> c00 += 1, c01 += 1, c10 += -1, c11 += 1
        U[0][col] = 1; U[4][col] = 1
        V[k_start][col] = 1; V[p+k_start+1][col] = 1
        W[k_start][col] = 1; W[3*k_start+3][col] = 1; W[k_start+1][col] = -1; W[3*k_start+4][col] = 1
        col += 1
        
        # Term 2: (a10+a11)b00 -> c10 += 1, c11 += 1
        U[3][col] = 1; U[4][col] = 1
        V[k_start][col] = 1
        W[k_start+1][col] = 1; W[3*k_start+4][col] = 1
        col += 1
        
        # Term 3: a00(b01-b11) -> c01 += 1, c11 += -1
        U[0][col] = 1
        V[k_start+1][col] = 1; V[p+k_start+1][col] = -1
        W[3*k_start+3][col] = 1; W[3*k_start+4][col] = -1
        col += 1
        
        # Term 4: a11(b10-b00) -> c10 += 1, c11 += -1
        U[4][col] = 1
        V[p+k_start][col] = 1; V[k_start][col] = -1
        W[k_start+1][col] = 1; W[3*k_start+4][col] = -1
        col += 1
        
        # Term 5: (a00+a01)b11 -> c01 += 1, c11 += 1
        U[0][col] = 1; U[1][col] = 1
        V[p+k_start+1][col] = 1
        W[3*k_start+3][col] = 1; W[3*k_start+4][col] = 1
        col += 1
        
        # Term 6: (a10-a00)(b00+b01) -> c10 += 1, c11 += -1
        U[3][col] = 1; U[0][col] = -1
        V[k_start][col] = 1; V[k_start+1][col] = 1
        W[k_start+1][col] = 1; W[3*k_start+4][col] = -1
        col += 1
        
        # Term 7: (a01-a11)(b10+b11) -> c00 += -1, c01 += 1, c10 += -1, c11 += 1
        U[1][col] = 1; U[4][col] = -1
        V[p+k_start][col] = 1; V[p+k_start+1][col] = 1
        W[k_start][col] = -1; W[3*k_start+3][col] = 1; W[k_start+1][col] = -1; W[3*k_start+4][col] = 1
        col += 1

    # Now handle v b (2x8)
    # v is A[0:2, 2] -> U[2] (i=0, j=2), U[5] (i=1, j=2)
    # b is B[2, 0:8] -> V[2*p + k] = V[16+k]
    # C_top is C[0:2, 0:8] -> W[k*n + i]
    
    for k in range(8):
        # v0 * b_k -> C[0, k]
        U[2][col] = 1
        V[16+k][col] = 1
        W[k*3 + 0][col] = 1
        col += 1
        # v1 * b_k -> C[1, k]
        U[5][col] = 1
        V[16+k][col] = 1
        W[k*3 + 1][col] = 1
        col += 1

    # Now handle w^T B_28 (1x8)
    # w is A[2, 0:2] -> U[6] (i=2, j=0), U[7] (i=2, j=1)
    # B_28 is B[0:2, 0:8] -> V[0*p + k], V[1*p + k]
    # C_bot is C[2, 0:8] -> W[k*n + 2]
    
    for k in range(8):
        # w0 * B[0,k] -> C[2, k]
        U[6][col] = 1
        V[k][col] = 1
        W[k*3 + 2][col] = 1
        col += 1
        # w1 * B[1,k] -> C[2, k]
        U[7][col] = 1
        V[3+k][col] = 1
        W[k*3 + 2][col] = 1
        col += 1

    # Now handle a b (1x8)
    # a is A[2,2] -> U[8]
    # b is B[2, 0:8] -> V[16+k]
    # C_bot is C[2, 0:8] -> W[k*n + 2]
    
    for k in range(8):
        U[8][col] = 1
        V[16+k][col] = 1
        W[k*3 + 2][col] = 1
        col += 1

    return {"U": U, "V": V, "W": W}
