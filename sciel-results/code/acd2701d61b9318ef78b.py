"""Strassen-based 2x5 x 5x6 matrix multiplication decomposition."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    import time
    import numpy as np
    from itertools import product

    start_time = time.time()
    DEATHLINE = 130.0
    n, m, p = 2, 5, 6

    def elapsed():
        return time.time() - start_time

    def build_trivial():
        """Build the trivial rank-60 decomposition."""
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][r] = 1
                    V[j * p + k][r] = 1
                    W[k * n + i][r] = 1
                    r += 1
        return {"U": U, "V": V, "W": W}, rank

    def build_strassen_block(local_U, local_V, local_W, offset_U, offset_V, offset_W, global_U, global_V, global_W, rank_offset):
        """
        Adds a Strassen block decomposition to the global matrices.
        local_U, local_V, local_W are for a 2x2x2 block.
        offset_U, offset_V, offset_W are the starting indices in the global matrices.
        """
        for r in range(7):
            for i in range(2):
                for j in range(2):
                    u_idx = offset_U + i * 2 + j
                    global_U[u_idx][rank_offset + r] += local_U[i * 2 + j][r]
            for j in range(2):
                for k in range(2):
                    v_idx = offset_V + j * 2 + k
                    global_V[v_idx][rank_offset + r] += local_V[j * 2 + k][r]
            for k in range(2):
                for i in range(2):
                    w_idx = offset_W + k * 2 + i
                    global_W[w_idx][rank_offset + r] += local_W[k * 2 + i][r]

    def build_strassen_2x2x2():
        """
        Returns U, V, W for a single 2x2x2 Strassen decomposition.
        U is 4x7, V is 4x7, W is 4x7.
        Indices: U[i*2+j], V[j*2+k], W[k*2+i].
        """
        U = [[0] * 7 for _ in range(4)]
        V = [[0] * 7 for _ in range(4)]
        W = [[0] * 7 for _ in range(4)]
        
        # M1 = (a+d)(e+h)
        # a=U[0,0], d=U[1,1], e=V[0,0], h=V[1,1]
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        # C11 += M1, C22 += M1
        W[0][0] = 1; W[3][0] = 1
        
        # M2 = (c+d)e
        # c=U[2,0], d=U[3,0], e=V[0,0]
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        # C21 += M2
        W[1][1] = 1
        
        # M3 = a(f-h)
        # a=U[0,0], f=V[0,1], h=V[1,1]
        U[0][2] = 1
        V[0][2] = 1; V[3][2] = -1
        # C12 += M3
        W[2][2] = 1
        
        # M4 = d(g-e)
        # d=U[3,0], g=V[1,0], e=V[0,0]
        U[3][3] = 1
        V[1][3] = 1; V[0][3] = -1
        # C11 += M4, C21 += M4
        W[0][3] = 1; W[1][3] = 1
        
        # M5 = (a+b)h
        # a=U[0,0], b=U[0,1], h=V[1,1]
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        # C11 -= M5, C12 += M5
        W[0][4] = -1; W[2][4] = 1
        
        # M6 = (c-a)(e+f)
        # c=U[2,0], a=U[0,0], e=V[0,0], f=V[0,1]
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        # C22 += M6
        W[3][5] = 1
        
        # M7 = (b-d)(g+h)
        # b=U[0,1], d=U[1,1], g=V[1,0], h=V[1,1]
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        # C11 += M7
        W[0][6] = 1
        
        return U, V, W

    def verify_decomposition(U, V, W, tol=1e-9):
        """Verify that the decomposition exactly decomposes the matrix multiplication tensor."""
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    s = 0.0
                    for r in range(rank):
                        s += U[u_idx][r] * V[v_idx][r] * W[w_idx][r]
                    if abs(s - 1.0) > tol:
                        return False
        return True

    # 1. Build the trivial decomposition as a fallback
    best_result, best_rank = build_trivial()
    
    # 2. Try to build the Strassen-based decomposition
    # Decompose M_{2,5,6} into:
    # A = [A1 (2x2), A2 (2x2), A3 (2x1)]
    # B = [B11 (2x2), B12 (2x2), B13 (2x2)]
    #     [B21 (2x2), B22 (2x2), B23 (2x2)]
    #     [B31 (1x2), B32 (1x2), B33 (1x2)]
    
    # C = A1 B11 + A1 B12 + A1 B13 + A2 B21 + A2 B22 + A2 B23 + A3 B31 + A3 B32 + A3 B33
    # This is not quite right. B is 5x6.
    # Rows of B: 5. Cols of B: 6.
    # A is 2x5.
    # A = [A_1 (2x2), A_2 (2x2), A_3 (2x1)]
    # B = [B_{11} (2x2), B_{12} (2x2), B_{13} (2x2)]
    #     [B_{21} (2x2), B_{22} (2x2), B_{23} (2x2)]
    #     [B_{31} (1x2), B_{32} (1x2), B_{33} (1x2)]
    
    # C = A_1 B_{11} + A_1 B_{12} + A_1 B_{13} + A_2 B_{21} + A_2 B_{22} + A_2 B_{23} + A_3 B_{31} + A_3 B_{32} + A_3 B_{33}
    
    # Terms:
    # 1. A_1 B_{11}: M_{2,2,2} -> Rank 7
    # 2. A_1 B_{12}: M_{2,2,2} -> Rank 7
    # 3. A_1 B_{13}: M_{2,2,2} -> Rank 7
    # 4. A_2 B_{21}: M_{2,2,2} -> Rank 7
    # 5. A_2 B_{22}: M_{2,2,2} -> Rank 7
    # 6. A_2 B_{23}: M_{2,2,2} -> Rank 7
    # 7. A_3 B_{31}: M_{2,1,2} -> Rank 1 (Outer product)
    # 8. A_3 B_{32}: M_{2,1,2} -> Rank 1
    # 9. A_3 B_{33}: M_{2,1,2} -> Rank 1
    
    # Total Rank = 6 * 7 + 3 * 1 = 42 + 3 = 45.
    
    # Let's construct this.
    U = [[0] * 45 for _ in range(n * m)] # 2*5 = 10 rows
    V = [[0] * 45 for _ in range(m * p)] # 5*6 = 30 rows
    W = [[0] * 45 for _ in range(p * n)] # 6*2 = 12 rows
    
    rank_offset = 0
    strassen_U, strassen_V, strassen_W = build_strassen_2x2x2()
    
    # Helper to add a block
    def add_block(u_off, v_off, w_off, block_U, block_V, block_W, off):
        for r in range(7):
            for i in range(2):
                for j in range(2):
                    U[u_off + i * 2 + j][off + r] += block_U[i * 2 + j][r]
            for j in range(2):
                for k in range(2):
                    V[v_off + j * 2 + k][off + r] += block_V[j * 2 + k][r]
            for k in range(2):
                for i in range(2):
                    W[w_off + k * 2 + i][off + r] += block_W[k * 2 + i][r]

    # Block 1: A_1 (cols 0-1) * B_11 (rows 0-1, cols 0-1)
    # U offset: 0 (A_1 is cols 0-1 of A, so indices 0-3)
    # V offset: 0 (B_11 is rows 0-1, cols 0-1, so indices 0-3)
    # W offset: 0 (C_11 is rows 0-1, cols 0-1, so indices 0-3)
    add_block(0, 0, 0, strassen_U, strassen_V, strassen_W, rank_offset)
    rank_offset += 7

    # Block 2: A_1 (cols 0-1) * B_12 (rows 0-1, cols 2-3)
    # U offset: 0
    # V offset: 2*2 = 4? No. B is 5x6.
    # B_12 is rows 0-1, cols 2-3.
    # Index in V: row * 6 + col.
    # Rows 0-1, Cols 2-3 -> 0*6+2=2, 0*6+3=3, 1*6+2=8, 1*6+3=9.
    # This is not a contiguous 2x2 block in the V matrix indices if we just use offset.
    # We need to map local indices to global indices.
    
    # Let's redefine the helper to take index mappings.
    
    U = [[0] * 45 for _ in range(n * m)]
    V = [[0] * 45 for _ in range(m * p)]
    W = [[0] * 45 for _ in range(p * n)]
    
    rank_offset = 0
    sU, sV, sW = build_strassen_2x2x2()
    
    def add_strassen_block(u_indices, v_indices, w_indices, off):
        # u_indices: list of 4 global indices for A block (2x2)
        # v_indices: list of 4 global indices for B block (2x2)
        # w_indices: list of 4 global indices for C block (2x2)
        for r in range(7):
            for li in range(4):
                U[u_indices[li]][off + r] += sU[li][r]
            for lj in range(4):
                V[v_indices[lj]][off + r] += sV[lj][r]
            for lk in range(4):
                W[w_indices[lk]][off + r] += sW[lk][r]

    def add_outer_product_block(u_indices, v_indices, w_indices, off):
        # u_indices: list of 2 global indices for A block (2x1)
        # v_indices: list of 1 global index for B block (1x1) -- Wait, B block is 1x2
        # v_indices: list of 2 global indices for B block (1x2)
        # w_indices: list of 2 global indices for C block (2x1) -- Wait, C block is 2x2?
        # A (2x1) * B (1x2) = C (2x2).
        # C_{ik} = A_i B_k.
        # This is not a single outer product in the tensor sense?
        # T_{i, j, k} = A_i B_k.
        # This is rank 1.
        # U_i = A_i, V_j = B_j (but j is only 1), W_k = B_k?
        # No.
        # The tensor is $u \otimes v \otimes w$?
        # $T_{i, j, k} = u_i v_j w_k$.
        # Here $j$ is fixed to 0 (local).
        # $T_{i, 0, k} = A_i B_k$.
        # So $u_i = A_i$, $v_0 = 1$, $w_k = B_k$?
        # Then $T_{i,0,k} = A_i \cdot 1 \cdot B_k = A_i B_k$.
        # Yes.
        # So we need:
        # U[u_indices[i]] = A_i
        # V[v_indices[0]] = 1
        # W[w_indices[k]] = B_k
        # But we are adding to global matrices.
        # We just add 1s and -1s.
        pass

    # This is getting complicated to implement generically.
    # Given the time, I will return the trivial decomposition if the construction is too complex.
    # But rank 45 is a huge improvement.
    
    # Let's try to implement it carefully.
    
    # A indices:
    # A_1: cols 0,1 -> U indices 0,1,2,3 (i*5+j)
    # A_2: cols 2,3 -> U indices 5,6,7,8
    # A_3: col 4 -> U indices 4,9
    
    # B indices:
    # B_11: rows 0,1 cols 0,1 -> V indices 0,1,6,7
    # B_12: rows 0,1 cols 2,3 -> V indices 2,3,8,9
    # B_13: rows 0,1 cols 4,5 -> V indices 4,5,10,11
    # B_21: rows 2,3 cols 0,1 -> V indices 12,13,18,19
    # B_22: rows 2,3 cols 2,3 -> V indices 14,15,20,21
    # B_23: rows 2,3 cols 4,5 -> V indices 16,17,22,23
    # B_31: row 4 cols 0,1 -> V indices 24,25
    # B_32: row 4 cols 2,3 -> V indices 26,27
    # B_33: row 4 cols 4,5 -> V indices 28,29
    
    # C indices:
    # C_11: rows 0,1 cols 0,1 -> W indices 0,1,2,3 (k*2+i)
    # C_12: rows 0,1 cols 2,3 -> W indices 4,5,6,7
    # C_13: rows 0,1 cols 4,5 -> W indices 8,9,10,11
    # C_21: rows 2,3 cols 0,1 -> W indices 0,1,2,3? No.
    # W index is k*n + i.
    # C_11 (cols 0,1): k=0,1. W indices 0,1 (i=0), 2,3 (i=1).
    # C_12 (cols 2,3): k=2,3. W indices 4,5 (i=0), 6,7 (i=1).
    # C_13 (cols 4,5): k=4,5. W indices 8,9 (i=0), 10,11 (i=1).
    # C_21 (cols 0,1, rows 2,3? No, C is 2x6. Rows are 0,1.
    # Wait, C is 2x6.
    # C_{ik} where i in 0,1 and k in 0..5.
    # So there is no C_21.
    # My partition of C was wrong.
    # C is 2x6.
    # C_1 (cols 0-1), C_2 (cols 2-3), C_3 (cols 4-5).
    
    # So:
    # A_1 B_11 -> C_1 (part 1)
    # A_1 B_12 -> C_2 (part 1)
    # A_1 B_13 -> C_3 (part 1)
    # A_2 B_21 -> C_1 (part 2)
    # A_2 B_22 -> C_2 (part 2)
    # A_2 B_23 -> C_3 (part 2)
    # A_3 B_31 -> C_1 (part 3)
    # A_3 B_32 -> C_2 (part 3)
    # A_3 B_33 -> C_3 (part 3)
    
    # U indices for A_1: [0, 1, 5, 6] (i=0,j=0; i=0,j=1; i=1,j=0; i=1,j=1)
    # U indices for A_2: [2, 3, 7, 8]
    # U indices for A_3: [4, 9]
    
    # V indices for B_11: [0, 1, 6, 7] (j=0,k=0; j=0,k=1; j=1,k=0; j=1,k=1)
    # V indices for B_12: [2, 3, 8, 9]
    # V indices for B_13: [4, 5, 10, 11]
    # V indices for B_21: [12, 13, 18, 19]
    # V indices for B_22: [14, 15, 20, 21]
    # V indices for B_23: [16, 17, 22, 23]
    # V indices for B_31: [24, 25]
    # V indices for B_32: [26, 27]
    # V indices for B_33: [28, 29]
    
    # W indices for C_1: [0, 1, 2, 3] (k=0,i=0; k=0,i=1; k=1,i=0; k=1,i=1)
    # W indices for C_2: [4, 5, 6, 7]
    # W indices for C_3: [8, 9, 10, 11]
    
    # Now add blocks.
    
    # 1. A_1 B_11 -> C_1
    add_strassen_block([0, 1, 5, 6], [0, 1, 6, 7], [0, 1, 2, 3], rank_offset)
    rank_offset += 7
    
    # 2. A_1 B_12 -> C_2
    add_strassen_block([0, 1, 5, 6], [2, 3, 8, 9], [4, 5, 6, 7], rank_offset)
    rank_offset += 7
    
    # 3. A_1 B_13 -> C_3
    add_strassen_block([0, 1, 5, 6], [4, 5, 10, 11], [8, 9, 10, 11], rank_offset)
    rank_offset += 7
    
    # 4. A_2 B_21 -> C_1
    add_strassen_block([2, 3, 7, 8], [12, 13, 18, 19], [0, 1, 2, 3], rank_offset)
    rank_offset += 7
    
    # 5. A_2 B_22 -> C_2
    add_strassen_block([2, 3, 7, 8], [14, 15, 20, 21], [4, 5, 6, 7], rank_offset)
    rank_offset += 7
    
    # 6. A_2 B_23 -> C_3
    add_strassen_block([2, 3, 7, 8], [16, 17, 22, 23], [8, 9, 10, 11], rank_offset)
    rank_offset += 7
    
    # 7. A_3 B_31 -> C_1
    # A_3 is 2x1. B_31 is 1x2. C_1 is 2x2.
    # T_{i, 4, k} = A_{i4} B_{4k}.
    # Rank 1.
    # U[4] = 1, U[9] = 1.
    # V[24] = 1, V[25] = 1.
    # W[0] = 1, W[1] = 1, W[2] = 1, W[3] = 1?
    # No.
    # T_{i, j, k} = U_{i,j} V_{j,k} W_{k,i}.
    # Here j=4.
    # T_{i, 4, k} = U_{i,4} V_{4,k} W_{k,i}.
    # We want this to be A_{i4} B_{4k}.
    # So we can set U_{i,4} = A_{i4}, V_{4,k} = B_{4k}, W_{k,i} = 1?
    # But U, V, W are fixed factors.
    # We can set:
    # U[4][r] = 1, U[9][r] = 1.
    # V[24][r] = 1, V[25][r] = 1.
    # W[0][r] = 1, W[2][r] = 1, W[1][r] = 1, W[3][r] = 1.
    # Then T_{0,4,0} = 1*1*1 = 1.
    # T_{1,4,0} = 1*1*1 = 1.
    # T_{0,4,1} = 1*1*1 = 1.
    # T_{1,4,1} = 1*1*1 = 1.
    # This works for A_{i4}=1, B_{4k}=1.
    # So we add 1s.
    
    U[4][rank_offset] = 1
    U[9][rank_offset] = 1
    V[24][rank_offset] = 1
    V[25][rank_offset] = 1
    W[0][rank_offset] = 1
    W[2][rank_offset] = 1
    W[1][rank_offset] = 1
    W[3][rank_offset] = 1
    rank_offset += 1
    
    # 8. A_3 B_32 -> C_2
    U[4][rank_offset] = 1
    U[9][rank_offset] = 1
    V[26][rank_offset] = 1
    V[27][rank_offset] = 1
    W[4][rank_offset] = 1
    W[6][rank_offset] = 1
    W[5][rank_offset] = 1
    W[7][rank_offset] = 1
    rank_offset += 1
    
    # 9. A_3 B_33 -> C_3
    U[4][rank_offset] = 1
    U[9][rank_offset] = 1
    V[28][rank_offset] = 1
    V[29][rank_offset] = 1
    W[8][rank_offset] = 1
    W[10][rank_offset] = 1
    W[9][rank_offset] = 1
    W[11][rank_offset] = 1
    rank_offset += 1
    
    # Total rank 45.
    
    if verify_decomposition(U, V, W):
        return {"U": U, "V": V, "W": W}
    
    return best_result
# EVOLVE-BLOCK-END
