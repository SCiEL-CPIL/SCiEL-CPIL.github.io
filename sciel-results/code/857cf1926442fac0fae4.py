"""Tensor product decomposition: Laderman(2x2) ⊗ Schoolbook(3x3) for 6x6 matrix mult."""
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 145  # safety margin before 150s kill
    n, m, p = DIMENSIONS  # 6, 6, 6
    n2, n3 = 2, 3  # block sizes: 6 = 2 * 3
    
    # Laderman's rank-7 decomposition for 2x2 matrix multiplication
    # U2: (2*2, 7), V2: (2*2, 7), W2: (2*2, 7)
    # Indexing: U2[i*m2+j, c], V2[j*m2+k, c], W2[k*n2+i, c]
    U2 = [
        [1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 1, 0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0, 0, 1],
    ]
    V2 = [
        [1, 1, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 1, 1],
    ]
    W2 = [
        [1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1],
    ]
    
    # Schoolbook rank-3 decomposition for 3x3 matrix multiplication
    # U3: (3*3, 3), V3: (3*3, 3), W3: (3*3, 3)
    U3 = [[0]*3 for _ in range(9)]
    V3 = [[0]*3 for _ in range(9)]
    W3 = [[0]*3 for _ in range(9)]
    for i in range(3):
        for j in range(3):
            c = i * 3 + j  # This won't work directly; schoolbook has rank 9 for 3x3
            pass
    # Actually schoolbook for 3x3 has rank 27. Let me reconsider.
    # For 3x3 schoolbook: rank = 3*3*3 = 27
    # Tensor product: rank = 7 * 27 = 189. Still better than 216.
    
    # Wait, let me reconsider. The schoolbook for nxn has rank n^3.
    # For 3x3: rank 27. For 2x2: rank 8 (schoolbook) or 7 (Laderman).
    # Tensor product rank: 7 * 27 = 189.
    
    # Let me build the full 6x6 decomposition using the Kronecker product approach.
    # 6x6 = (2x3) x (2x3) blocks... actually let's think of it as:
    # A[6,6] * B[6,6] = C[6,6]
    # Group rows/cols: row index = a*3 + b where a in {0,1,2}, b in {0,1,2}? 
    # No, let's use the block structure: 6 = 2*3, so 3 blocks of size 2.
    
    # Better approach: Use the fact that if T1 is a decomposition of M_{2,2,2} with rank r1
    # and T2 is a decomposition of M_{3,3,3} with rank r2, then T1 ⊗ T2 is a decomposition
    # of M_{6,6,6} with rank r1*r2.
    
    # Schoolbook for 3x3: rank 27
    rank3 = 27
    U3 = [[0]*rank3 for _ in range(9)]
    V3 = [[0]*rank3 for _ in range(9)]
    W3 = [[0]*rank3 for _ in range(9)]
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U3[i*3+j][col] = 1
                V3[j*3+k][col] = 1
                W3[k*3+i][col] = 1
                col += 1
    
    rank1 = 7
    total_rank = rank1 * rank3  # 189
    
    # Build U, V, W of shapes (36, 189), (36, 189), (36, 189)
    U = [[0]*total_rank for _ in range(36)]
    V = [[0]*total_rank for _ in range(36)]
    W = [[0]*total_rank for _ in range(36)]
    
    for c1 in range(rank1):
        for c3 in range(rank3):
            c = c1 * rank3 + c3
            u1 = U2[:, c1] if isinstance(U2, list) else None
            u3 = U3[:, c3] if isinstance(U3, list) else None
            
            # U2 is 4x7, U3 is 9x27
            # For the tensor product: U[i1*n3+i2, c] = U2[i1][c1] * U3[i2][c3]
            # where i1 in {0,1} (2x2 part), i2 in {0,1,2,3,4,5} (3x3 part flattened)
            # Wait, the indexing needs care.
            
            # Full 6x6: row index for U is i_full*m_full + j_full where i_full in 0..5, j_full in 0..5
            # Decompose: i_full = i2*2 + i1, j_full = j2*2 + j1 where i1,j1 in {0,1}, i2,j2 in {0,1,2}
            # So U_full[i2*2+i1, j2*2+j1] = U2[i1, j1; c1] * U3[i2, j2; c3]
            
            for i2 in range(3):
                for i1 in range(2):
                    row_u = (i2*2 + i1) * 6 + 0  # This is wrong
                    pass
            
            # Let me redo the indexing carefully.
            # U has shape (n*m, rank) = (36, 189)
            # Row index = i*m + j where i,j in 0..5
            # i = i2*2 + i1, j = j2*2 + j1
            # U[i,j] = U2[i1,j1] * U3[i2,j2]
            
            for i2 in range(3):
                for j2 in range(3):
                    u3_val = U3[i2*3 + j2][c3]
                    if u3_val == 0:
                        continue
                    for i1 in range(2):
                        for j1 in range(2):
                            u2_val = U2[i1*2 + j1][c1]
                            if u2_val == 0:
                                continue
                            i_full = i2*2 + i1
                            j_full = j2*2 + j1
                            row = i_full * 6 + j_full
                            U[row][c] = u2_val * u3_val
            
            # V has shape (m*p, rank) = (36, 189)
            # Row index = j*m + k where j,k in 0..5
            # j = j2*2 + j1, k = k2*2 + k1
            # V[j,k] = V2[j1,k1] * V3[j2,k2]
            
            for j2 in range(3):
                for k2 in range(3):
                    v3_val = V3[j2*3 + k2][c3]
                    if v3_val == 0:
                        continue
                    for j1 in range(2):
                        for k1 in range(2):
                            v2_val = V2[j1*2 + k1][c1]
                            if v2_val == 0:
                                continue
                            j_full = j2*2 + j1
                            k_full = k2*2 + k1
                            row = j_full * 6 + k_full
                            V[row][c] = v2_val * v3_val
            
            # W has shape (p*n, rank) = (36, 189)
            # Row index = k*n + i where k,i in 0..5 (trace convention W[k*n+i, r])
            # k = k2*2 + k1, i = i2*2 + i1
            # W[k,i] = W2[k1,i1] * W3[k2,i2]
            
            for k2 in range(3):
                for i2 in range(3):
                    w3_val = W3[k2*3 + i2][c3]
                    if w3_val == 0:
                        continue
                    for k1 in range(2):
                        for i1 in range(2):
                            w2_val = W2[k1*2 + i1][c1]
                            if w2_val == 0:
                                continue
                            k_full = k2*2 + k1
                            i_full = i2*2 + i1
                            row = k_full * 6 + i_full
                            W[row][c] = w2_val * w3_val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
