"""Program for matrix multiplication tensor decomposition of 5x5 by 5x5."""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # Strategy: Use Strassen-based decomposition for the 4x4x4 block (rank 56 via 8x Strassen 2x2)
    # and schoolbook for the remaining blocks.
    # Total rank: 56 + 16 + 16 + 4 + 16 + 4 + 4 + 1 = 117.
    # This is an improvement over the schoolbook rank 125.
    
    def get_strassen_2x2():
        """
        Returns U, V, W for 2x2x2 matrix multiplication using Strassen's algorithm (rank 7).
        U, V, W have shape (4, 7).
        Indices: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r].
        """
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # Strassen's 7 multiplications:
        # M1 = (A00 + A11) * (B00 + B11)
        # M2 = (A10 + A11) * B00
        # M3 = A00 * (B10 - B11)
        # M4 = A11 * (B01 - B00)
        # M5 = (A00 + A10) * B11
        # M6 = (A01 - A00) * (B00 + B10)
        # M7 = (A10 - A11) * (B01 + B11)
        
        # C00 = M1 - M4 + M5 - M7
        # C01 = M3 + M4
        # C10 = M2 + M7
        # C11 = M1 - M2 + M3 + M6
        
        # We define the rank-1 terms (u, v, w) such that sum(u_ij v_jk w_ki) = C_ik
        
        # Term 1: M1
        # u: A00=1, A11=1 => U[0,0]=1, U[3,0]=1
        # v: B00=1, B11=1 => V[0,0]=1, V[3,0]=1
        # w: C00=1, C11=1 => W[0,0]=1, W[3,0]=1
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # Term 2: M2
        # u: A10=1, A11=1 => U[2,1]=1, U[3,1]=1
        # v: B00=1 => V[0,1]=1
        # w: C10=1 => W[2,1]=1 (C10 is at index k=1, i=0? No, C10 is i=1, k=0. W index is k*n+i. n=2. W[0*2+1]=W[1]. Wait.
        # C_ik. i=1, k=0. W index is k*2+i = 0*2+1 = 1. So W[1,1]=1.
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[1, 1] = 1
        
        # Term 3: M3
        # u: A00=1 => U[0,2]=1
        # v: B10=1, B11=-1 => V[2,2]=1, V[3,2]=-1
        # w: C01=1 => W[2,2]=1 (i=0, k=1. W[1*2+0]=W[2])
        U[0, 2] = 1
        V[2, 2] = 1; V[3, 2] = -1
        W[2, 2] = 1
        
        # Term 4: M4
        # u: A11=1 => U[3,3]=1
        # v: B01=1, B00=-1 => V[1,3]=1, V[0,3]=-1
        # w: C01=1 (from +M4), C00=-1 (from -M4)
        # C01: i=0, k=1 => W[2,3]=1
        # C00: i=0, k=0 => W[0,3]=-1
        U[3, 3] = 1
        V[1, 3] = 1; V[0, 3] = -1
        W[2, 3] = 1; W[0, 3] = -1
        
        # Term 5: M5
        # u: A00=1, A10=1 => U[0,4]=1, U[2,4]=1
        # v: B11=1 => V[3,4]=1
        # w: C00=1 => W[0,4]=1
        U[0, 4] = 1; U[2, 4] = 1
        V[3, 4] = 1
        W[0, 4] = 1
        
        # Term 6: M6
        # u: A01=1, A00=-1 => U[1,5]=1, U[0,5]=-1
        # v: B00=1, B10=1 => V[0,5]=1, V[2,5]=1
        # w: C11=1 => W[3,5]=1 (i=1, k=1. W[1*2+1]=W[3])
        U[1, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[2, 5] = 1
        W[3, 5] = 1
        
        # Term 7: M7
        # u: A10=1, A11=-1 => U[2,6]=1, U[3,6]=-1
        # v: B01=1, B11=1 => V[1,6]=1, V[3,6]=1
        # w: C10=1 (from +M7), C00=-1 (from -M7)
        # C10: i=1, k=0 => W[1,6]=1
        # C00: i=0, k=0 => W[0,6]=-1
        U[2, 6] = 1; U[3, 6] = -1
        V[1, 6] = 1; V[3, 6] = 1
        W[1, 6] = 1; W[0, 6] = -1
        
        return U, V, W
    
    def schoolbook_decomp(n, m_, p_, rank_local):
        """Standard schoolbook decomposition for n x m x p multiplication."""
        U = np.zeros((n * m_, rank_local))
        V = np.zeros((m_ * p_, rank_local))
        W = np.zeros((p_ * n, rank_local))
        col = 0
        for i in range(n):
            for j in range(m_):
                for k in range(p_):
                    U[i * m_ + j, col] = 1
                    V[j * p_ + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W
    
    # Define blocks and their ranks
    # (bn, bm, bp, rank)
    blocks = [
        (4, 4, 4, 56),  # 8 Strassen 2x2
        (4, 4, 1, 16),  # Schoolbook
        (4, 1, 4, 16),  # Schoolbook
        (4, 1, 1, 4),   # Schoolbook
        (1, 4, 4, 16),  # Schoolbook
        (1, 4, 1, 4),   # Schoolbook
        (1, 1, 4, 4),   # Schoolbook
        (1, 1, 1, 1),   # Schoolbook
    ]
    
    total_rank = sum(b[3] for b in blocks)
    
    U = np.zeros((25, total_rank))
    V = np.zeros((25, total_rank))
    W = np.zeros((25, total_rank))
    
    col_offset = 0
    
    strassen_U, strassen_V, strassen_W = get_strassen_2x2()
    
    for (bn, bm, bp, rank) in blocks:
        if bn == 4 and bm == 4 and bp == 4:
            # Use 8 Strassen decompositions
            # Blocks of 4x4 matrix:
            # A00 (0:2, 0:2), A01 (0:2, 2:4), A10 (2:4, 0:2), A11 (2:4, 2:4)
            # B00 (0:2, 0:2), B01 (0:2, 2:4), B10 (2:4, 0:2), B11 (2:4, 2:4)
            # C00 (0:2, 0:2), C01 (0:2, 2:4), C10 (2:4, 0:2), C11 (2:4, 2:4)
            
            # Mapping of (row_block, col_block) to local 2x2 indices
            # A indices: (i, j) -> (i_block, j_block)
            # i in 0..3, j in 0..3
            # i_block = i // 2, j_block = j // 2
            
            # We need to add 8 Strassen decompositions.
            # Each Strassen decomposition computes C_block = A_block * B_block
            # The terms are:
            # C00 += A00*B00
            # C00 += A01*B10
            # C01 += A00*B01
            # C01 += A01*B11
            # C10 += A10*B00
            # C10 += A11*B10
            # C11 += A10*B01
            # C11 += A11*B11
            
            # For each term, we use the Strassen vectors.
            # The Strassen vectors are defined for 2x2 matrices.
            # We need to map the local 2x2 indices (li, lj, lk) to global 4x4 indices.
            
            # List of (A_block, B_block, C_block)
            # A_block: (row_start, col_start)
            # B_block: (row_start, col_start)
            # C_block: (row_start, col_start)
            
            terms = [
                ((0, 0), (0, 0), (0, 0)), # A00 B00 -> C00
                ((0, 2), (2, 0), (0, 0)), # A01 B10 -> C00
                ((0, 0), (0, 2), (0, 2)), # A00 B01 -> C01
                ((0, 2), (2, 2), (0, 2)), # A01 B11 -> C01
                ((2, 0), (0, 0), (2, 0)), # A10 B00 -> C10
                ((2, 2), (2, 0), (2, 0)), # A11 B10 -> C10
                ((2, 0), (0, 2), (2, 2)), # A10 B01 -> C11
                ((2, 2), (2, 2), (2, 2)), # A11 B11 -> C11
            ]
            
            for t_idx, (a_block, b_block, c_block) in enumerate(terms):
                # a_block: (ar, ac)
                # b_block: (br, bc)
                # c_block: (cr, cc)
                
                ar, ac = a_block
                br, bc = b_block
                cr, cc = c_block
                
                # For each of the 7 Strassen terms
                for s in range(7):
                    # Add to U, V, W
                    # U indices: (i, j) in 4x4
                    # V indices: (j, k) in 4x4
                    # W indices: (k, i) in 4x4
                    
                    # Strassen U has 1s at specific (li, lj)
                    # We need to map (li, lj) to (i, j)
                    # i = ar + li
                    # j = ac + lj
                    
                    # Strassen V has 1s at specific (lj, lk)
                    # Map (lj, lk) to (j, k)
                    # j = br + lj  (Note: j in V is the column of A, row of B)
                    # k = bc + lk
                    
                    # Strassen W has 1s at specific (lk, li)
                    # Map (lk, li) to (k, i)
                    # k = cr + lk
                    # i = cc + li
                    
                    # Iterate over all 4x4 indices to add contributions?
                    # No, we can just iterate over the non-zeros in Strassen vectors.
                    # But it's easier to iterate over all 16 entries and add if non-zero.
                    
                    for li in range(2):
                        for lj in range(2):
                            u_val = strassen_U[li * 2 + lj, s]
                            if u_val != 0:
                                i = ar + li
                                j = ac + lj
                                U[i * 4 + j, col_offset + t_idx * 7 + s] += u_val
                            
                            for lk in range(2):
                                v_val = strassen_V[lj * 2 + lk, s]
                                if v_val != 0:
                                    j = br + lj
                                    k = bc + lk
                                    V[j * 4 + k, col_offset + t_idx * 7 + s] += v_val
                            
                            for lk in range(2):
                                for li in range(2):
                                    w_val = strassen_W[lk * 2 + li, s]
                                    if w_val != 0:
                                        k = cr + lk
                                        i = cc + li
                                        W[k * 4 + i, col_offset + t_idx * 7 + s] += w_val
        else:
            # Schoolbook decomposition
            local_U, local_V, local_W = schoolbook_decomp(bn, bm, bp, rank)
            
            for li in range(bn):
                gi = li if bn == 4 else 4
                for lj in range(bm):
                    gj = lj if bm == 4 else 4
                    for lk in range(bp):
                        gk = lk if bp == 4 else 4
                        
                        local_col = li * bm * bp + lj * bp + lk
                        
                        U[gi * 5 + gj, col_offset + local_col] = local_U[li * bm + lj, local_col]
                        V[gj * 5 + gk, col_offset + local_col] = local_V[lj * bp + lk, local_col]
                        W[gk * 5 + gi, col_offset + local_col] = local_W[lk * bn + li, local_col]
        
        col_offset += rank
    
    # Check time budget
    if time.time() > deadline:
        # Fall back to trivial schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
